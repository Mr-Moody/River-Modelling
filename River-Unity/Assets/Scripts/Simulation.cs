using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

/// <summary>
/// The main simulation controller class.
/// Runs physics simulation directly on the river mesh using river-local coordinates.
/// </summary>
public class SimulationController : MonoBehaviour
{
    // --- Integration Component ---
    [Header("Geometry Integration")]
    [Tooltip("Reference to the RiverGeometry GameObject containing CSVGeometryLoader component.")]
    public GameObject RiverGeometry;
    
    [Header("River Geometry Update")]
    [Tooltip("If enabled, the river geometry will be updated based on simulation results (erosion/deposition).")]
    public bool UpdateRiverGeometry = true;
    
    [Tooltip("Scale factor for applying bed elevation changes to river geometry. Higher values make changes more visible.")]
    [Range(0.1f, 10f)]
    public float RiverGeometryElevationScale = 1.0f;
    
    [Tooltip("Enable horizontal bank migration visualization. When enabled, river width changes are visible as banks migrate. Disable if mesh disappears.")]
    public bool EnableHorizontalBankMigration = false;
    
    public enum VisualizationMode { Velocity, Erosion }
    [Header("Visualization")]
    [Tooltip("Visualization mode: Velocity shows flow speed, Erosion shows bed elevation changes and bank migration.")]
    public VisualizationMode visualizationMode = VisualizationMode.Erosion;
    
    [Tooltip("Erosion threshold percentile for visualization. Only erosion rates above this percentile will show as red. Higher values = less red (more selective). Recommended: 0.7-0.9")]
    [Range(0.0f, 1.0f)]
    public float ErosionVisualizationThreshold = 0.85f; // Only show top 15% of erosion rates as red
    
    [Header("Terrain")]
    [Tooltip("If enabled, generates and updates 3D terrain mesh around the river.")]
    public bool EnableTerrain = false; // Disabled by default since we're not using grid anymore
    
    private CSVGeometryLoader geometryLoader;
    private TerrainMesh terrainMesh;

    // --- Configuration Parameters ---
    [Header("Simulation Control")]
    [Tooltip("Toggle to start/stop the simulation. When off, mesh will display but simulation steps won't run.")]
    public bool RunSimulation = false;
    
    private bool lastRunSimulationState = false;

    [Header("Time")]
    [Range(0.001f, 0.1f)]
    [Tooltip("Time step for simulation. Should be small (0.01 or less) for stability. Larger values cause numerical instability.")]
    public float TimeStep = 0.01f;
    public int IterationsPerFrame = 1;
    public float InitialWaterDepth = 0.5f;

    // --- Physics Parameters (Passed to Solver) ---
    [Header("Physics Parameters")]
    public double Nu = 1e-6;
    public double Rho = 1000.0;
    public double G = 9.81;
    public double SedimentDensity = 2650.0;
    public double Porosity = 0.4;
    public double CriticalShear = 0.05;
    public double TransportCoefficient = 0.1;
    
    [Header("Bank Erosion Parameters")]
    [Tooltip("Critical shear stress for bank erosion. Lower values = easier to erode (faster migration). Recommended: 0.05-0.15")]
    public double BankCriticalShear = 0.05;  // Lowered from 0.15 for faster erosion
    
    [Tooltip("Bank erosion rate multiplier. Higher values = faster bank erosion and migration. Recommended: 1.0-5.0 for visible migration")]
    public double BankErosionRate = 2.0;  // Increased from 0.3 for faster migration
    
    [Tooltip("Erosion threshold before bank migrates (meters). Lower = faster migration. Recommended: 0.001-0.01")]
    public double BankMigrationThreshold = 0.005;  // Lower = migrates more frequently
    
    [Header("Output")]
    [Tooltip("Automatically export river mesh CSV when simulation stops.")]
    public bool AutoExportMeshOnStop = true;
    
    [Tooltip("Filename for exported river mesh CSV (written to persistentDataPath).")]
    public string MeshExportFileName = "RiverMeshOutput.csv";

    private RiverMeshPhysicsSolver riverMeshSolver;

    // --- Unity Lifecycle Methods ---

    void OnEnable()
    {
        // Subscribe to mesh loaded event from CSVGeometryLoader
        CSVGeometryLoader.OnMeshLoaded += HandleMeshLoaded;
    }

    void OnDisable()
    {
        // Unsubscribe from events
        CSVGeometryLoader.OnMeshLoaded -= HandleMeshLoaded;
    }

    void Start()
    {
        Debug.Log("[SimulationController] Start() called - Getting component references...");
        
        // Get references to components on RiverGeometry
        EnsureComponentReferences();
        
        // Try to initialize if mesh is already loaded
        if (geometryLoader != null)
        {
            Mesh riverMesh = geometryLoader.GetMesh();
            if (riverMesh != null)
            {
                HandleMeshLoaded(riverMesh);
            }
            else
            {
                // Even if mesh isn't loaded yet, ensure shader is ready
                EnsureVelocityShader();
            }
        }
        
        Debug.Log("[SimulationController] ✓ Component references obtained.");
    }

    /// <summary>
    /// Ensures component references are set up. Can be called multiple times safely.
    /// </summary>
    private void EnsureComponentReferences()
    {
        if (RiverGeometry == null)
        {
            Debug.LogError("SimulationController: RiverGeometry GameObject reference is missing!");
            return;
        }
        
        if (geometryLoader == null)
        {
            geometryLoader = RiverGeometry.GetComponent<CSVGeometryLoader>();
            if (geometryLoader == null)
            {
                Debug.LogError("SimulationController: CSVGeometryLoader component not found on RiverGeometry GameObject!");
            }
        }
        
        if (terrainMesh == null && EnableTerrain)
        {
            // Try to find TerrainMesh on RiverGeometry
            terrainMesh = RiverGeometry.GetComponent<TerrainMesh>();
            if (terrainMesh == null)
            {
                Debug.LogWarning("[SimulationController] TerrainMesh not found. Terrain generation is disabled.");
            }
        }
    }

    /// <summary>
    /// Handles mesh loaded event from CSVGeometryLoader.
    /// </summary>
    private void HandleMeshLoaded(Mesh mesh)
    {
        if (mesh == null)
        {
            Debug.LogWarning("[SimulationController] Received null mesh from CSVGeometryLoader.");
            return;
        }

        if (riverMeshSolver != null)
        {
            Debug.Log("[SimulationController] River mesh solver already initialized, skipping...");
            return;
        }

        // Ensure component references are set
        EnsureComponentReferences();

        Debug.Log("[SimulationController] Starting river mesh physics initialization...");
        InitializeRiverMeshSimulation();
        
        // Initialize vertex colors for shader visualization
        InitializeVertexColors();
    }

    private float lastLogTime = 0f;
    private const float LOG_INTERVAL = 2f; // Log every 2 seconds

    void Update()
    {
        // Validate TimeStep
        if (TimeStep > 0.1f)
        {
            Debug.LogError($"[SimulationController] CRITICAL: TimeStep ({TimeStep}) is too large! This will cause numerical instability. Recommended: 0.01 or less. Clamping to 0.01.");
            TimeStep = 0.01f;
        }
        else if (TimeStep > 0.05f)
        {
            Debug.LogWarning($"[SimulationController] WARNING: TimeStep ({TimeStep}) is large and may cause instability. Recommended: 0.01 or less.");
        }
        
        // Check if RunSimulation state changed
        bool wasRunning = lastRunSimulationState;
        if (RunSimulation != lastRunSimulationState)
        {
            Debug.Log($"[SimulationController] RunSimulation toggled to: {RunSimulation}. Solver initialized: {riverMeshSolver != null}");
            lastRunSimulationState = RunSimulation;
            
            // If we just stopped, export mesh if configured
            if (wasRunning && !RunSimulation && AutoExportMeshOnStop)
            {
                ExportCurrentRiverMesh();
            }
        }
        
        // Only run simulation if solver is initialized and RunSimulation toggle is on
        if (riverMeshSolver == null)
        {
            return; // Solver not initialized yet
        }
        
        if (!RunSimulation)
        {
            // Simulation is paused - still update visualization if mesh is ready
            if (UpdateRiverGeometry && geometryLoader != null)
            {
                if (visualizationMode == VisualizationMode.Erosion)
                {
                    UpdateRiverGeometryMeshWithErosion();
                }
                else
                {
                    UpdateRiverGeometryMesh();
                }
            }
            else
            {
                // Even when paused, ensure vertex colors are initialized for shader visualization
                InitializeVertexColors();
            }
            return;
        }

        // Debug logging (throttled to avoid spam)
        if (Time.time - lastLogTime > LOG_INTERVAL)
        {
            Debug.Log($"[SimulationController] Simulation RUNNING - Running {IterationsPerFrame} iteration(s) per frame with dt={TimeStep}");
            lastLogTime = Time.time;
        }

        // Run the simulation a fixed number of times per frame for stability
        for (int i = 0; i < IterationsPerFrame; i++)
        {
            RunRiverMeshTimeStep(TimeStep);
        }

        // Update the visual mesh with the new data
        if (UpdateRiverGeometry)
        {
            if (visualizationMode == VisualizationMode.Erosion)
            {
                UpdateRiverGeometryMeshWithErosion();
            }
            else
            {
                UpdateRiverGeometryMesh();
            }
        }
    }

    // --- Initialization and Setup ---

    /// <summary>
    /// Initializes the river mesh physics simulation.
    /// </summary>
    private void InitializeRiverMeshSimulation()
    {
        Debug.Log("[SimulationController] InitializeRiverMeshSimulation() called");
        
        // Ensure component references are set
        EnsureComponentReferences();

        if (geometryLoader == null)
        {
            Debug.LogError("SimulationController: CSVGeometryLoader reference is missing! Make sure RiverGeometry GameObject has a CSVGeometryLoader component.");
            return;
        }

        // Initialize river mesh-based physics solver
        Mesh riverMesh = geometryLoader.GetMesh();
        if (riverMesh == null || riverMesh.vertices == null)
        {
            Debug.LogError("[SimulationController] River mesh is null or has no vertices. Cannot initialize simulation.");
            return;
        }

        if (!geometryLoader.useGridMesh)
        {
            Debug.LogError("[SimulationController] River mesh must be in grid format (useGridMesh must be enabled in CSVGeometryLoader). Cannot initialize simulation.");
            return;
        }

        Vector3[] vertices = riverMesh.vertices;
        int totalVertices = vertices.Length;
        int widthRes = geometryLoader.widthResolution;
        
        // Calculate number of cross-sections
        // For grid mesh: totalVertices = numCrossSections * widthResolution
        if (totalVertices % widthRes != 0)
        {
            Debug.LogError($"[SimulationController] River mesh vertex count ({totalVertices}) is not divisible by widthResolution ({widthRes}). Cannot initialize river mesh physics.");
            return;
        }

        int numCrossSections = totalVertices / widthRes;
        
        // Get scale factor from geometry loader (converts meters to Unity units)
        // Calculate unity to meters conversion: unityToMeters = 1.0 / scaleFactor
        float scaleFactor = geometryLoader.scaleFactor;
        double unityToMetersScale = (scaleFactor > 0.0f) ? (1.0 / (double)scaleFactor) : 2000.0;
        
        Debug.Log($"[SimulationController] Step 1: Creating RiverMeshPhysicsSolver for {numCrossSections} cross-sections x {widthRes} width points...");
        Debug.Log($"[SimulationController] Unit conversion: scaleFactor={scaleFactor}, unityToMetersScale={unityToMetersScale}");
        
        riverMeshSolver = new RiverMeshPhysicsSolver(
            vertices, numCrossSections, widthRes,
            nu: Nu, rho: Rho, g: G,
            sedimentDensity: SedimentDensity,
            porosity: Porosity,
            criticalShear: CriticalShear,
            transportCoefficient: TransportCoefficient,
            bankCriticalShear: BankCriticalShear,
            bankErosionRate: BankErosionRate,
            unityToMetersScale: unityToMetersScale
        );

        // Update cell types based on geometry
        riverMeshSolver.UpdateCellTypes(vertices);
        
        // Initialize bed elevation from mesh Y coordinates
        riverMeshSolver.InitializeBedElevationFromMesh(vertices);
        
        // Set bank migration threshold (controls how fast banks move)
        riverMeshSolver.bankMigrationThreshold = BankMigrationThreshold;
        
        // Set initial water depth
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthRes; w++)
            {
                if (riverMeshSolver.cellType[i, w] == RiverCellType.FLUID)
                {
                    riverMeshSolver.waterDepth[i, w] = InitialWaterDepth;
                }
            }
        }
        
        Debug.Log("[SimulationController] ✓ RiverMeshPhysicsSolver created");
        Debug.Log("[SimulationController] Step 2: River mesh solver initialized with river-local coordinates (u along river, v across river)");
        
        // Ensure velocity heatmap shader is applied to the material
        EnsureVelocityShader();
        
        Debug.Log($"[SimulationController] ✓ Initialization complete. Simulation is {(RunSimulation ? "RUNNING" : "PAUSED")}.");
    }
    
    /// <summary>
    /// Ensures the river mesh material uses the velocity heatmap shader.
    /// </summary>
    private void EnsureVelocityShader()
    {
        if (geometryLoader == null || RiverGeometry == null)
            return;
            
        GameObject riverMeshObject = RiverGeometry.transform.Find("RiverMesh")?.gameObject;
        if (riverMeshObject == null)
            return;
            
        MeshRenderer mr = riverMeshObject.GetComponent<MeshRenderer>();
        if (mr == null)
            return;
            
        Material mat = mr.material;
        if (mat == null)
            return;
            
        // Check if material already uses velocity heatmap shader
        Shader velocityHeatmapShader = Shader.Find("Custom/RiverVelocityHeatmap");
        if (velocityHeatmapShader != null && mat.shader != velocityHeatmapShader)
        {
            mat.shader = velocityHeatmapShader;
            mat.SetFloat("_MaxVelocity", 1.0f);
            mat.SetFloat("_MinVelocity", 0.0f);
            mat.SetColor("_Color", Color.white);
            mat.SetFloat("_Metallic", 0.0f);
            mat.SetFloat("_Glossiness", 0.5f);
            Debug.Log("[SimulationController] Applied RiverVelocityHeatmap shader to river material.");
        }
    }

    // --- Core Simulation Logic ---

    private int stepCount = 0;

    /// <summary>
    /// Runs one time step using the river mesh physics solver (river-local coordinates).
    /// </summary>
    private void RunRiverMeshTimeStep(float dt)
    {
        if (riverMeshSolver == null)
        {
            Debug.LogWarning("RunRiverMeshTimeStep called but riverMeshSolver is not initialized.");
            return;
        }

        // Run Navier-Stokes step in river-local coordinates
        riverMeshSolver.NavierStokesStep(dt);
        
        // Compute bed shear stress
        double[,] tau = riverMeshSolver.ComputeShearStress();
        
        // Compute sediment flux vector
        (double[,] qs_s, double[,] qs_w) = riverMeshSolver.ComputeSedimentFluxVector(tau);
        
        // Solve Exner equation (updates bed elevation h)
        (_, riverMeshSolver.h) = riverMeshSolver.ExnerEquation(qs_s, qs_w, dt);
        
        stepCount++;
        
        // Log periodically
        if (stepCount % 100 == 0)
        {
            double maxVel = 0.0;
            for (int i = 0; i < riverMeshSolver.numCrossSections; i++)
            {
                for (int w = 0; w < riverMeshSolver.widthResolution; w++)
                {
                    double vel = riverMeshSolver.GetVelocityMagnitude(i, w);
                    if (vel > maxVel) maxVel = vel;
                }
            }
            Debug.Log($"[SimulationController] River mesh step {stepCount}: Max velocity = {maxVel:F6} m/s (u along river, v across river)");
        }
    }
    

    // --- Visualization ---

    /// <summary>
    /// Initializes vertex colors for the velocity shader, even when simulation hasn't run.
    /// This ensures the shader can display something even with zero velocity.
    /// </summary>
    private void InitializeVertexColors()
    {
        if (geometryLoader == null)
            return;
            
        Mesh riverMesh = geometryLoader.GetMesh();
        if (riverMesh == null || riverMesh.vertices == null)
            return;
            
        // Initialize colors if not already set
        Color[] colors = riverMesh.colors;
        if (colors == null || colors.Length != riverMesh.vertices.Length)
        {
            colors = new Color[riverMesh.vertices.Length];
            // Initialize with zero velocity (blue)
            for (int i = 0; i < colors.Length; i++)
            {
                colors[i] = new Color(0, 0, 0, 1); // Zero velocity = blue in heatmap
            }
            riverMesh.colors = colors;
            
            // Update mesh filter
            GameObject riverMeshObject = RiverGeometry.transform.Find("RiverMesh")?.gameObject;
            if (riverMeshObject != null)
            {
                MeshFilter mf = riverMeshObject.GetComponent<MeshFilter>();
                if (mf != null)
                {
                    mf.sharedMesh = riverMesh;
                }
            }
        }
    }
    
    /// <summary>
    /// Updates vertex colors with erosion data for erosion heatmap visualization.
    /// Red channel: normalized erosion rate (0-1)
    /// Green channel: deposition flag (1.0 = deposition, 0.0 = erosion)
    /// Blue channel: bank migration indicator (1.0 = active migration, 0.0 = no migration)
    /// </summary>
    private void UpdateRiverGeometryMeshWithErosion()
    {
        Mesh riverMesh = geometryLoader.GetMesh();
        if (riverMesh == null || riverMesh.vertices == null)
        {
            return;
        }
        
        Vector3[] riverVertices = riverMesh.vertices;
        
        // Initialize or resize color array
        Color[] colors = riverMesh.colors;
        if (colors == null || colors.Length != riverVertices.Length)
        {
            colors = new Color[riverVertices.Length];
        }
        
        if (riverMeshSolver == null || !geometryLoader.useGridMesh)
        {
            return; // Solver not initialized or mesh not in grid format
        }

        // Use river mesh solver (direct mapping - vertices correspond to solver cells)
        int numCrossSections = riverMeshSolver.numCrossSections;
        int widthResolution = riverMeshSolver.widthResolution;
        
        // Collect all erosion rates for percentile-based normalization
        List<float> erosionRates = new List<float>();
        
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                double erosionRate = Math.Abs(riverMeshSolver.GetErosionRate(i, w));
                erosionRates.Add((float)erosionRate);
            }
        }
        
        // Use percentile-based normalization to avoid outliers dominating
        // Sort and use 95th percentile as max (or actual max if smaller)
        erosionRates.Sort();
        float percentile95 = erosionRates.Count > 0 ? erosionRates[Mathf.Min((int)(erosionRates.Count * 0.95f), erosionRates.Count - 1)] : 0.001f;
        float maxErosionRate = Mathf.Max(percentile95, 0.001f);
        
        // Threshold for significant erosion - use configurable percentile
        // Only erosion rates above this percentile will show as red
        float thresholdPercentile = Mathf.Clamp01(ErosionVisualizationThreshold);
        int thresholdIndex = erosionRates.Count > 0 ? Mathf.Min((int)(erosionRates.Count * thresholdPercentile), erosionRates.Count - 1) : 0;
        float significantErosionThreshold = erosionRates.Count > 0 ? erosionRates[thresholdIndex] : 0.001f;
        
        // Ensure threshold is at least a reasonable fraction of max to avoid showing everything
        // This ensures only truly significant erosion shows as red
        float minThresholdFraction = 0.3f; // At least 30% of max must be threshold
        significantErosionThreshold = Mathf.Max(significantErosionThreshold, maxErosionRate * minThresholdFraction);
        
        // Store original bank positions for each cross-section (for horizontal migration)
        // Use current mesh boundary vertices as reference positions (these represent the original bank positions)
        Vector3[] originalLeftBank = new Vector3[numCrossSections];
        Vector3[] originalRightBank = new Vector3[numCrossSections];
        bool hasValidBankPositions = false;
        
        // Extract bank positions from current mesh (first and last vertices of each cross-section)
        for (int i = 0; i < numCrossSections; i++)
        {
            int firstVertexIdx = i * widthResolution;
            int lastVertexIdx = i * widthResolution + (widthResolution - 1);
            if (firstVertexIdx < riverVertices.Length && lastVertexIdx < riverVertices.Length)
            {
                originalLeftBank[i] = riverVertices[firstVertexIdx];
                originalRightBank[i] = riverVertices[lastVertexIdx];
                
                // Validate positions (check for NaN or invalid values)
                if (float.IsNaN(originalLeftBank[i].x) || float.IsNaN(originalRightBank[i].x) ||
                    float.IsInfinity(originalLeftBank[i].x) || float.IsInfinity(originalRightBank[i].x))
                {
                    continue;
                }
                hasValidBankPositions = true;
            }
        }
        
        // If we don't have valid bank positions, skip horizontal migration
        if (!hasValidBankPositions)
        {
            Debug.LogWarning("[SimulationController] Cannot update horizontal positions - invalid bank positions detected");
        }
        
        // Update vertices and colors with erosion data
        for (int i = 0; i < numCrossSections; i++)
        {
            // Get current bank edges from solver
            (int leftBankEdge, int rightBankEdge) = riverMeshSolver.GetBankEdges(i);
            
            // Get original bank positions for this cross-section
            Vector3 leftBankPos = originalLeftBank[i];
            Vector3 rightBankPos = originalRightBank[i];
            
            // Calculate center point and direction for this cross-section
            Vector3 centerPos = (leftBankPos + rightBankPos) * 0.5f;
            Vector3 bankDirection = rightBankPos - leftBankPos;
            float originalFullWidth = bankDirection.magnitude;
            
            // Determine current fluid region
            int originalFluidStart = 1; // Assuming original banks were at edges
            int originalFluidEnd = widthResolution - 2;
            int originalFluidWidth = Mathf.Max(1, originalFluidEnd - originalFluidStart + 1);
            
            for (int w = 0; w < widthResolution; w++)
            {
                int vertexIdx = i * widthResolution + w;
                if (vertexIdx >= riverVertices.Length) continue;
                
                Vector3 vertex = riverVertices[vertexIdx];
                Vector3 newPosition = vertex;
                
                // Only apply horizontal migration if enabled and we have valid bank positions and edges
                if (EnableHorizontalBankMigration && hasValidBankPositions && leftBankEdge >= 0 && rightBankEdge >= 0 && leftBankEdge < rightBankEdge)
                {
                    // Calculate current fluid region
                    int currentFluidStart = leftBankEdge + 1;
                    int currentFluidEnd = rightBankEdge - 1;
                    int currentFluidWidth = Mathf.Max(1, currentFluidEnd - currentFluidStart + 1);
                    
                    // Calculate where the current bank edges are in world space
                    // Map bank edge indices to normalized positions (0 = leftmost, 1 = rightmost)
                    float leftEdgeNormalized = (float)leftBankEdge / (float)(widthResolution - 1);
                    float rightEdgeNormalized = (float)rightBankEdge / (float)(widthResolution - 1);
                    
                    // Calculate world-space positions of current bank edges
                    Vector3 currentLeftEdgePos = Vector3.Lerp(leftBankPos, rightBankPos, leftEdgeNormalized);
                    Vector3 currentRightEdgePos = Vector3.Lerp(leftBankPos, rightBankPos, rightEdgeNormalized);
                    
                    // Validate calculated positions
                    if (float.IsNaN(currentLeftEdgePos.x) || float.IsNaN(currentRightEdgePos.x) ||
                        float.IsInfinity(currentLeftEdgePos.x) || float.IsInfinity(currentRightEdgePos.x))
                    {
                        // Fallback to original interpolation if calculated positions are invalid
                        float normalizedWidth = (float)w / (float)(widthResolution - 1);
                        newPosition = Vector3.Lerp(leftBankPos, rightBankPos, normalizedWidth);
                    }
                    else
                    {
                        // Map vertex width index to normalized position
                        float wNormalized = (float)w / (float)(widthResolution - 1);
                        
                        if (w <= leftBankEdge)
                        {
                            // Vertex is in left bank region - interpolate between original left and current left edge
                            float t = (leftBankEdge > 0) ? (float)w / (float)leftBankEdge : 0f;
                            newPosition = Vector3.Lerp(leftBankPos, currentLeftEdgePos, t);
                        }
                        else if (w >= rightBankEdge)
                        {
                            // Vertex is in right bank region - interpolate between current right edge and original right
                            float t = (widthResolution - 1 - rightBankEdge > 0) ? 
                                (float)(w - rightBankEdge) / (float)(widthResolution - 1 - rightBankEdge) : 0f;
                            newPosition = Vector3.Lerp(currentRightEdgePos, rightBankPos, t);
                        }
                        else
                        {
                            // Vertex is in fluid region - interpolate between current bank edges
                            float t = (currentFluidWidth > 1) ? 
                                (float)(w - currentFluidStart) / (float)(currentFluidWidth - 1) : 0.5f;
                            newPosition = Vector3.Lerp(currentLeftEdgePos, currentRightEdgePos, t);
                        }
                        
                        // Final validation
                        if (float.IsNaN(newPosition.x) || float.IsInfinity(newPosition.x))
                        {
                            // Fallback to original position if new position is invalid
                            newPosition = vertex;
                        }
                    }
                }
                else
                {
                    // No valid bank edges found or invalid bank positions - use original interpolation across full width
                    // This preserves the original mesh appearance
                    float normalizedWidth = (float)w / (float)(widthResolution - 1);
                    newPosition = Vector3.Lerp(leftBankPos, rightBankPos, normalizedWidth);
                    
                    // Validate the interpolated position
                    if (float.IsNaN(newPosition.x) || float.IsInfinity(newPosition.x))
                    {
                        // Keep original vertex position if interpolation fails
                        newPosition = vertex;
                    }
                }
                
                // Update elevation from bed elevation (h represents changes from initial state, in meters)
                // Convert meters to Unity units for visualization
                // Apply: finalElevation = (initialElevationMeters + hMeters) * metersToUnityScale * visualizationScale
                double initialElevationMeters = riverMeshSolver.GetInitialBedElevation(i, w);
                double bedElevationChangeMeters = riverMeshSolver.h[i, w];
                double totalElevationMeters = initialElevationMeters + bedElevationChangeMeters;
                
                // Convert meters to Unity units
                double metersToUnityScale = riverMeshSolver.GetMetersToUnityScale();
                float finalElevation = (float)(totalElevationMeters * metersToUnityScale * RiverGeometryElevationScale);
                riverVertices[vertexIdx] = new Vector3(newPosition.x, finalElevation, newPosition.z);
                
                // Get erosion rate (negative = erosion, positive = deposition)
                double erosionRate = riverMeshSolver.GetErosionRate(i, w);
                float absErosionRate = Mathf.Abs((float)erosionRate);
                
                // Only normalize if above threshold, otherwise show as zero (blue)
                float normalizedErosion = 0.0f;
                if (absErosionRate > significantErosionThreshold)
                {
                    // Normalize based on threshold range (threshold to max)
                    float range = maxErosionRate - significantErosionThreshold;
                    if (range > 0.0001f)
                    {
                        normalizedErosion = Mathf.Clamp01((absErosionRate - significantErosionThreshold) / range);
                    }
                }
                
                // Determine if this is deposition (positive dh_dt) or erosion (negative dh_dt)
                float isDeposition = (erosionRate > 0) ? 1.0f : 0.0f;
                
                // Check for bank migration
                float bankMigration = riverMeshSolver.IsBankMigrating(i, w) ? 1.0f : 0.0f;
                
                // Store in vertex color: R = normalized erosion, G = deposition flag, B = bank migration
                colors[vertexIdx] = new Color(normalizedErosion, isDeposition, bankMigration, 1);
            }
        }
        
        // Regenerate triangles to respect cell type boundaries
        RegenerateTrianglesWithCellTypeBoundaries(riverMesh, numCrossSections, widthResolution);
        
        // Apply updated vertices and colors to mesh
        riverMesh.vertices = riverVertices;
        riverMesh.colors = colors;
        riverMesh.RecalculateNormals();
        riverMesh.RecalculateBounds();
        
        // Update the MeshFilter on the child GameObject to reflect changes
        GameObject riverMeshObject = RiverGeometry.transform.Find("RiverMesh")?.gameObject;
        if (riverMeshObject != null)
        {
            MeshFilter mf = riverMeshObject.GetComponent<MeshFilter>();
            if (mf != null)
            {
                mf.sharedMesh = riverMesh;
            }
        }
    }
    
    /// <summary>
    /// Updates the original red river geometry mesh based on simulation results.
    /// Maps simulation grid elevations to river geometry vertices.
    /// Works with both unstructured meshes (original strip mesh) and grid meshes (remeshed structure).
    /// Also updates vertex colors with velocity data for heatmap visualization.
    /// Supports both grid-based and river mesh-based physics solvers.
    /// </summary>
    private void UpdateRiverGeometryMesh()
    {
        Mesh riverMesh = geometryLoader.GetMesh();
        if (riverMesh == null || riverMesh.vertices == null)
        {
            return;
        }
        
        Vector3[] riverVertices = riverMesh.vertices;
        
        // Initialize or resize color array
        Color[] colors = riverMesh.colors;
        if (colors == null || colors.Length != riverVertices.Length)
        {
            colors = new Color[riverVertices.Length];
        }
        
        float maxVelocity = 0.0f;
        
        if (riverMeshSolver == null || !geometryLoader.useGridMesh)
        {
            return; // Solver not initialized or mesh not in grid format
        }

        // Use river mesh solver (direct mapping - vertices correspond to solver cells)
        int numCrossSections = riverMeshSolver.numCrossSections;
        int widthResolution = riverMeshSolver.widthResolution;
        
        // Calculate maximum velocity magnitude (normal) across all points
        maxVelocity = 0.0f;
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                double vel = riverMeshSolver.GetVelocityMagnitude(i, w);
                if (vel > maxVelocity) maxVelocity = (float)vel;
            }
        }
        
        // Set minimum threshold to avoid division by zero
        if (maxVelocity < 0.01f) maxVelocity = 1.0f;
        
        // Update vertices directly from river mesh solver
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                int vertexIdx = i * widthResolution + w;
                if (vertexIdx >= riverVertices.Length) continue;
                
                Vector3 vertex = riverVertices[vertexIdx];
                
                // Update elevation from bed elevation (h represents changes from initial state, in meters)
                // Convert meters to Unity units for visualization
                // Apply: finalElevation = (initialElevationMeters + hMeters) * metersToUnityScale * visualizationScale
                double initialElevationMeters = riverMeshSolver.GetInitialBedElevation(i, w);
                double bedElevationChangeMeters = riverMeshSolver.h[i, w];
                double totalElevationMeters = initialElevationMeters + bedElevationChangeMeters;
                
                // Convert meters to Unity units
                double metersToUnityScale = riverMeshSolver.GetMetersToUnityScale();
                float finalElevation = (float)(totalElevationMeters * metersToUnityScale * RiverGeometryElevationScale);
                riverVertices[vertexIdx] = new Vector3(vertex.x, finalElevation, vertex.z);
                
                // Get velocity magnitude (u along river, v across river)
                double vel = riverMeshSolver.GetVelocityMagnitude(i, w);
                float normalizedVelocity = Mathf.Clamp01((float)vel / maxVelocity);
                colors[vertexIdx] = new Color(normalizedVelocity, 0, 0, 1);
            }
        }
        
        // Regenerate triangles to respect cell type boundaries (prevent triangles across banks)
        // This must be done before assigning vertices so triangles are correct when recalculating normals
        RegenerateTrianglesWithCellTypeBoundaries(riverMesh, numCrossSections, widthResolution);
        
        // Apply updated vertices and colors to mesh
        riverMesh.vertices = riverVertices;
        riverMesh.colors = colors;
        riverMesh.RecalculateNormals();
        riverMesh.RecalculateBounds();
        
        // Update the MeshFilter on the child GameObject to reflect changes
        GameObject riverMeshObject = RiverGeometry.transform.Find("RiverMesh")?.gameObject;
        if (riverMeshObject != null)
        {
            MeshFilter mf = riverMeshObject.GetComponent<MeshFilter>();
            if (mf != null)
            {
                mf.sharedMesh = riverMesh;
            }
            
            // Ensure material uses velocity heatmap shader and update properties
            MeshRenderer mr = riverMeshObject.GetComponent<MeshRenderer>();
            if (mr != null)
            {
                Material mat = mr.material;
                if (mat != null)
                {
                    // Ensure shader is correct
                    Shader velocityHeatmapShader = Shader.Find("Custom/RiverVelocityHeatmap");
                    if (velocityHeatmapShader != null && mat.shader != velocityHeatmapShader)
                    {
                        mat.shader = velocityHeatmapShader;
                    }
                    
                    // Update material's max velocity property for proper heatmap scaling
                    mat.SetFloat("_MaxVelocity", maxVelocity);
                    mat.SetFloat("_MinVelocity", 0.0f);
                }
            }
        }
    }

    /// <summary>
    /// Exports the current river mesh geometry to CSV matching the Jurua schema.
    /// Columns: ,order,group,centerline_x,centerline_y,centerline_x_corrected,centerline_y_corrected,right_bank_x,right_bank_y,left_bank_x,left_bank_y,width (m),curvature
    /// </summary>
    /// <param name="overridePath">Optional full path override. Defaults to Application.persistentDataPath/MeshExportFileName.</param>
    public void ExportCurrentRiverMesh(string overridePath = null)
    {
        if (geometryLoader == null)
        {
            Debug.LogError("[SimulationController] Cannot export mesh - geometryLoader is null.");
            return;
        }
        if (riverMeshSolver == null)
        {
            Debug.LogError("[SimulationController] Cannot export mesh - solver not initialized.");
            return;
        }
        
        Mesh mesh = geometryLoader.GetMesh();
        if (mesh == null || mesh.vertices == null)
        {
            Debug.LogError("[SimulationController] Cannot export mesh - mesh is null.");
            return;
        }
        
        Vector3[] verts = mesh.vertices;
        int numCrossSections = riverMeshSolver.numCrossSections;
        int widthRes = riverMeshSolver.widthResolution;
        if (verts.Length < numCrossSections * widthRes)
        {
            Debug.LogError($"[SimulationController] Cannot export mesh - vertex count {verts.Length} insufficient for {numCrossSections}x{widthRes} grid.");
            return;
        }
        
        string path = string.IsNullOrEmpty(overridePath)
            ? Path.Combine(Application.persistentDataPath, MeshExportFileName)
            : overridePath;
        
        var sb = new StringBuilder();
        sb.AppendLine(",order,group,centerline_x,centerline_y,centerline_x_corrected,centerline_y_corrected,right_bank_x,right_bank_y,left_bank_x,left_bank_y,width (m),curvature");
        
        var culture = CultureInfo.InvariantCulture;
        for (int i = 0; i < numCrossSections; i++)
        {
            int leftIdx = i * widthRes;
            int rightIdx = i * widthRes + (widthRes - 1);
            
            Vector3 left = verts[leftIdx];
            Vector3 right = verts[rightIdx];
            Vector3 center = (left + right) * 0.5f;
            
            // Use x/z as planform coordinates; y is elevation
            float width = Vector2.Distance(new Vector2(left.x, left.z), new Vector2(right.x, right.z));
            
            int indexCol = i;          // leading unnamed column
            float order = i + 1;       // mimic sequential order
            float group = 1.0f;        // single group
            float curvature = 0.0f;    // not computed here
            
            sb.Append(indexCol.ToString(culture)).Append(',')
              .Append(order.ToString(culture)).Append(',')
              .Append(group.ToString(culture)).Append(',')
              .Append(center.x.ToString(culture)).Append(',')
              .Append(center.z.ToString(culture)).Append(',')
              .Append(center.x.ToString(culture)).Append(',')
              .Append(center.z.ToString(culture)).Append(',')
              .Append(right.x.ToString(culture)).Append(',')
              .Append(right.z.ToString(culture)).Append(',')
              .Append(left.x.ToString(culture)).Append(',')
              .Append(left.z.ToString(culture)).Append(',')
              .Append(width.ToString(culture)).Append(',')
              .Append(curvature.ToString(culture)).AppendLine();
        }
        
        try
        {
            File.WriteAllText(path, sb.ToString());
            Debug.Log($"[SimulationController] Exported river mesh CSV to: {path}");
        }
        catch (Exception ex)
        {
            Debug.LogError($"[SimulationController] Failed to export mesh CSV: {ex.Message}");
        }
    }

    /// <summary>
    /// Regenerates mesh triangles to respect cell type boundaries.
    /// Prevents triangles from spanning across BANK-FLUID boundaries which causes visual artifacts.
    /// </summary>
    private void RegenerateTrianglesWithCellTypeBoundaries(Mesh riverMesh, int numCrossSections, int widthResolution)
    {
        if (riverMeshSolver == null || riverMesh == null)
        {
            return;
        }
        
        List<int> triangles = new List<int>();
        
        // Generate triangles only within the same cell type regions
        for (int i = 0; i < numCrossSections - 1; i++)
        {
            for (int w = 0; w < widthResolution - 1; w++)
            {
                // Get cell types for the four vertices of this quad
                int cellType00 = riverMeshSolver.cellType[i, w];
                int cellType01 = riverMeshSolver.cellType[i, w + 1];
                int cellType10 = riverMeshSolver.cellType[i + 1, w];
                int cellType11 = riverMeshSolver.cellType[i + 1, w + 1];
                
                // Check if this quad spans across different cell types (BANK-FLUID boundary)
                bool spansBankBoundary = false;
                
                // Check if any edge crosses from BANK to FLUID or vice versa
                if (cellType00 != cellType01) spansBankBoundary = true; // Left edge
                if (cellType00 != cellType10) spansBankBoundary = true; // Top edge
                if (cellType01 != cellType11) spansBankBoundary = true; // Right edge
                if (cellType10 != cellType11) spansBankBoundary = true; // Bottom edge
                
                // Skip triangles that span across cell type boundaries
                if (spansBankBoundary)
                {
                    continue;
                }
                
                // All vertices have the same cell type - generate triangles normally
                int current = i * widthResolution + w;
                int next = (i + 1) * widthResolution + w;
                int currentRight = i * widthResolution + (w + 1);
                int nextRight = (i + 1) * widthResolution + (w + 1);
                
                // Triangle 1: current, currentRight, next
                triangles.Add(current);
                triangles.Add(currentRight);
                triangles.Add(next);
                
                // Triangle 2: next, currentRight, nextRight
                triangles.Add(next);
                triangles.Add(currentRight);
                triangles.Add(nextRight);
            }
        }
        
        // Update mesh triangles
        riverMesh.triangles = triangles.ToArray();
    }
    
    // --- Public Getters (Optional, for external visualization/debugging) ---

    public RiverMeshPhysicsSolver GetSolver()
    {
        return riverMeshSolver;
    }
    
    /// <summary>
    /// Forces an update of the river geometry mesh visualization.
    /// Useful when running physics steps externally (e.g., in LongTermSimulationController).
    /// </summary>
    public void ForceMeshUpdate()
    {
        if (UpdateRiverGeometry && riverMeshSolver != null)
        {
            if (visualizationMode == VisualizationMode.Erosion)
            {
                UpdateRiverGeometryMeshWithErosion();
            }
            else
            {
                UpdateRiverGeometryMesh();
            }
        }
    }
}
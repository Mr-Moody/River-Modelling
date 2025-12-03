using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;

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
    
    [Header("Visualization")]
    public enum VisualizationMode { Velocity, Erosion }
    [Tooltip("Visualization mode: Velocity shows flow speed, Erosion shows bed elevation changes and bank migration.")]
    public VisualizationMode visualizationMode = VisualizationMode.Velocity;
    
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
        if (RunSimulation != lastRunSimulationState)
        {
            Debug.Log($"[SimulationController] RunSimulation toggled to: {RunSimulation}. Solver initialized: {riverMeshSolver != null}");
            lastRunSimulationState = RunSimulation;
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
        
        Debug.Log($"[SimulationController] Step 1: Creating RiverMeshPhysicsSolver for {numCrossSections} cross-sections x {widthRes} width points...");
        
        riverMeshSolver = new RiverMeshPhysicsSolver(
            vertices, numCrossSections, widthRes,
            nu: Nu, rho: Rho, g: G,
            sedimentDensity: SedimentDensity,
            porosity: Porosity,
            criticalShear: CriticalShear,
            transportCoefficient: TransportCoefficient,
            bankCriticalShear: BankCriticalShear,
            bankErosionRate: BankErosionRate
        );

        // Update cell types based on geometry
        riverMeshSolver.UpdateCellTypes(vertices);
        
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
        
        // Find max/min erosion rates for normalization
        float maxErosionRate = 0.0f;
        float minErosionRate = 0.0f;
        
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                double erosionRate = Math.Abs(riverMeshSolver.GetErosionRate(i, w));
                if (erosionRate > maxErosionRate) maxErosionRate = (float)erosionRate;
            }
        }
        
        // Set minimum threshold to avoid division by zero
        if (maxErosionRate < 0.0001f) maxErosionRate = 0.001f;
        
        // Update vertices and colors with erosion data
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                int vertexIdx = i * widthResolution + w;
                if (vertexIdx >= riverVertices.Length) continue;
                
                Vector3 vertex = riverVertices[vertexIdx];
                
                // Update elevation from bed elevation
                float bedElevation = (float)riverMeshSolver.h[i, w] * RiverGeometryElevationScale;
                riverVertices[vertexIdx] = new Vector3(vertex.x, bedElevation, vertex.z);
                
                // Get erosion rate (negative = erosion, positive = deposition)
                double erosionRate = riverMeshSolver.GetErosionRate(i, w);
                float absErosionRate = Mathf.Abs((float)erosionRate);
                float normalizedErosion = Mathf.Clamp01(absErosionRate / maxErosionRate);
                
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
                
                // Update elevation from bed elevation
                float bedElevation = (float)riverMeshSolver.h[i, w] * RiverGeometryElevationScale;
                riverVertices[vertexIdx] = new Vector3(vertex.x, bedElevation, vertex.z);
                
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
}
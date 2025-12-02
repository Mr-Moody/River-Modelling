using UnityEngine;
using System;
using System.Collections;

/// <summary>
/// The main simulation controller class.
/// Listens to mesh updates from RiverGeometry and runs the physics simulation,
/// then sends results back to MeshGrid for visualization.
/// </summary>
public class SimulationController : MonoBehaviour
{
    // --- Integration Component ---
    [Header("Geometry Integration")]
    [Tooltip("Reference to the RiverGeometry GameObject containing MeshGrid component.")]
    public GameObject RiverGeometry;
    
    [Header("Visualization")]
    [Tooltip("If enabled, shows the simulation grid mesh (white grid). Disable to hide it.")]
    public bool ShowSimulationGrid = false;
    
    [Header("River Geometry Update")]
    [Tooltip("If enabled, the original red river geometry will be updated based on simulation results (erosion/deposition).")]
    public bool UpdateRiverGeometry = true;
    
    [Tooltip("Scale factor for applying bed elevation changes to river geometry. Higher values make changes more visible.")]
    [Range(0.1f, 10f)]
    public float RiverGeometryElevationScale = 1.0f;
    
    [Header("Terrain")]
    [Tooltip("If enabled, generates and updates 3D terrain mesh around the river.")]
    public bool EnableTerrain = true;
    
    private MeshGrid meshGrid;
    private RiverToGrid riverToGrid;
    private CSVGeometryLoader geometryLoader;
    private TerrainMesh terrainMesh;

    // --- Configuration Parameters ---
    [Header("Simulation Control")]
    [Tooltip("Toggle to start/stop the simulation. When off, mesh will display but simulation steps won't run.")]
    public bool RunSimulation = false;
    
    private bool lastRunSimulationState = false;

    [Header("Grid & Time")]
    // NOTE: GridWidth and GridHeight will be OVERRIDDEN by RiverToGrid calculations.
    [HideInInspector] public int GridWidth;
    [HideInInspector] public int GridHeight;

    public float CellSize = 0.5f;
    [Range(0.001f, 0.1f)]
    [Tooltip("Time step for simulation. Should be small (0.01 or less) for stability. Larger values cause numerical instability.")]
    public float TimeStep = 0.01f;
    public int IterationsPerFrame = 1;
    public float InitialWaterDepth = 0.5f; // New initial condition parameter

    // --- Physics Parameters (Passed to Solver) ---
    [Header("Physics Parameters")]
    public double Nu = 1e-6;
    public double Rho = 1000.0;
    public double G = 9.81;
    public double SedimentDensity = 2650.0;
    public double Porosity = 0.4;
    public double CriticalShear = 0.05;
    public double TransportCoefficient = 0.1;

    private PhysicsSolver solver;

    // --- Data Arrays (Pointers to Solver Data) ---
    private double[,] h; 		 // Bed Elevation
    private double[,] waterDepth; 	 // Water Depth
    private double[,] u; 		 // X-Velocity
    private double[,] v; 		 // Y-Velocity

    // --- Unity Lifecycle Methods ---

    void OnEnable()
    {
        // Subscribe to mesh update events from MeshGrid
        MeshGrid.OnMeshUpdated += HandleMeshUpdated;
        RiverToGrid.OnGridInitialized += HandleGridInitialized;
    }

    void OnDisable()
    {
        // Unsubscribe from events
        MeshGrid.OnMeshUpdated -= HandleMeshUpdated;
        RiverToGrid.OnGridInitialized -= HandleGridInitialized;
    }

    void Start()
    {
        Debug.Log("[SimulationController] Start() called - Getting component references...");
        
        // Get references to components on RiverGeometry
        EnsureComponentReferences();
        
        Debug.Log("[SimulationController] ✓ Component references obtained. Waiting for grid initialization event...");
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

        if (meshGrid == null)
        {
            meshGrid = RiverGeometry.GetComponent<MeshGrid>();
            if (meshGrid == null)
            {
                Debug.LogError("SimulationController: MeshGrid component not found on RiverGeometry GameObject!");
            }
        }

        if (riverToGrid == null)
        {
            riverToGrid = RiverGeometry.GetComponent<RiverToGrid>();
            if (riverToGrid == null)
            {
                Debug.LogError("SimulationController: RiverToGrid component not found on RiverGeometry GameObject!");
            }
        }
        
        if (geometryLoader == null)
        {
            geometryLoader = RiverGeometry.GetComponent<CSVGeometryLoader>();
            if (geometryLoader == null)
            {
                Debug.LogWarning("SimulationController: CSVGeometryLoader component not found. River geometry updates will be disabled.");
            }
        }
        
        if (terrainMesh == null && EnableTerrain)
        {
            // Try to find TerrainMesh on RiverGeometry or create it
            terrainMesh = RiverGeometry.GetComponent<TerrainMesh>();
            if (terrainMesh == null)
            {
                // Create a child GameObject for terrain
                // Position it at the same location as RiverGeometry so terrain aligns with river
                GameObject terrainObject = new GameObject("Terrain");
                terrainObject.transform.SetParent(RiverGeometry.transform);
                terrainObject.transform.localPosition = Vector3.zero; // Will be repositioned in InitializeTerrain
                terrainObject.transform.localRotation = Quaternion.identity;
                terrainObject.transform.localScale = Vector3.one;
                terrainMesh = terrainObject.AddComponent<TerrainMesh>();
                terrainMesh.riverToGrid = riverToGrid;
                Debug.Log("[SimulationController] Created TerrainMesh component");
            }
        }
    }

    /// <summary>
    /// Handles grid initialization event from RiverToGrid.
    /// </summary>
    private void HandleGridInitialized(int gridWidth, int gridHeight, float minX, float minZ, float maxX, float maxZ)
    {
        Debug.Log($"[SimulationController] HandleGridInitialized called - Grid: {gridWidth}x{gridHeight}");
        
        if (solver != null)
        {
            Debug.Log("[SimulationController] Solver already initialized, skipping...");
            // Already initialized, skip
            return;
        }

        // Ensure component references are set (in case event fires before Start())
        EnsureComponentReferences();

        Debug.Log("[SimulationController] Starting simulation initialization...");
        InitializeSimulation(gridWidth, gridHeight, minX, minZ);
        
        // Initialize terrain after grid is ready
        if (EnableTerrain && terrainMesh != null && riverToGrid != null && riverToGrid.isInitialized)
        {
            terrainMesh.InitializeTerrain();
            Debug.Log("[SimulationController] ✓ Terrain initialized after grid");
        }
    }

    /// <summary>
    /// Handles mesh update event from MeshGrid (called when mesh is first created or updated).
    /// </summary>
    private void HandleMeshUpdated(Mesh mesh)
    {
        // Mesh has been updated, simulation will use it in Update()
        // This is mainly for initialization tracking
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
            Debug.Log($"[SimulationController] RunSimulation toggled to: {RunSimulation}. Solver initialized: {solver != null}");
            lastRunSimulationState = RunSimulation;
        }
        
        // Only run simulation if solver is initialized and RunSimulation toggle is on
        if (solver == null)
        {
            return; // Solver not initialized yet
        }
        
        if (!RunSimulation)
        {
            // Simulation is paused - still update visualization if mesh is ready
            if (meshGrid != null && meshGrid.GetMesh() != null)
            {
                UpdateVisualization();
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
            RunTimeStep(TimeStep);
        }

        // Update the visual mesh with the new data
        UpdateVisualization();
    }

    // --- Initialization and Setup ---

    private void InitializeSimulation(int gridWidth, int gridHeight, float minX, float minZ)
    {
        Debug.Log("[SimulationController] InitializeSimulation() called");
        
        // Ensure component references are set (in case event fires before Start())
        EnsureComponentReferences();

        if (riverToGrid == null)
        {
            Debug.LogError("SimulationController: RiverToGrid reference is missing! Make sure RiverGeometry GameObject has a RiverToGrid component.");
            return;
        }

        // Validate grid dimensions
        if (gridWidth <= 0 || gridHeight <= 0)
        {
            Debug.LogError($"SimulationController: Invalid grid dimensions received: {gridWidth}x{gridHeight}. Cannot initialize simulation.");
            return;
        }

        // Set Grid dimensions
        GridWidth = gridWidth;
        GridHeight = gridHeight;
        float cellSize = riverToGrid.GetCellSize();

        Debug.Log($"[SimulationController] Step 1: Creating PhysicsSolver for {GridWidth}x{GridHeight} grid...");
        
        // Initialize the Physics Solver
        solver = new PhysicsSolver(
            GridWidth, GridHeight, cellSize,
            nu: Nu, rho: Rho, g: G,
            sedimentDensity: SedimentDensity,
            porosity: Porosity,
            criticalShear: CriticalShear,
            transportCoefficient: TransportCoefficient
        );

        Debug.Log("[SimulationController] ✓ PhysicsSolver created");
        Debug.Log("[SimulationController] Step 2: Generating cell types and depths (this may take a moment for large grids)...");

        // Set initial conditions using the loaded geometry
        riverToGrid.GenerateCellTypesAndDepths(solver, InitialWaterDepth);

        // Count fluid cells to verify initialization
        int fluidCellCount = 0;
        for (int i = 0; i < GridWidth; i++)
        {
            for (int j = 0; j < GridHeight; j++)
            {
                if (solver.cellType[i, j] == RiverCellType.FLUID)
                {
                    fluidCellCount++;
                }
            }
        }
        
        Debug.Log($"[SimulationController] ✓ Cell types and depths generated. Found {fluidCellCount} fluid cells out of {GridWidth * GridHeight} total cells.");
        
        if (fluidCellCount == 0)
        {
            Debug.LogError("[SimulationController] ERROR: No fluid cells found! The simulation cannot run. " +
                          "Check that the grid cellSize is small enough to capture the river's sinuous shape. " +
                          "The river geometry may be too small relative to the grid cell size.");
        }
        Debug.Log("[SimulationController] Step 3: Caching array references...");

        // Cache array references from the solver
        h = solver.h;
        waterDepth = solver.waterDepth;
        u = solver.u;
        v = solver.v;

        Debug.Log("[SimulationController] Step 4: Waiting for mesh to be ready...");
        
        // Hide simulation grid if disabled
        if (!ShowSimulationGrid)
        {
            // Hide grid mesh - will be done after mesh is created
            StartCoroutine(HideSimulationGridWhenReady());
        }

        // Don't update visualization immediately - wait for MeshGrid to be initialized via event
        // The MeshGrid will be initialized by the HandleGridInitialized event handler
        // We'll update visualization in the next frame or when mesh is ready
        StartCoroutine(UpdateVisualizationWhenReady());

        Debug.Log($"[SimulationController] ✓ Initialization complete: Grid Size {GridWidth}x{GridHeight}. Simulation is {(RunSimulation ? "RUNNING" : "PAUSED")}.");
    }

    // --- Core Simulation Logic ---

    private int stepCount = 0;
    private double lastMaxVelocity = 0.0;

    private void RunTimeStep(float dt)
    {
        // Safety check: ensure solver is initialized
        if (solver == null)
        {
            Debug.LogWarning("RunTimeStep called but solver is not initialized. Check initialization errors.");
            return;
        }

        // Track max velocity for debugging (before step)
        double maxVelBefore = 0.0;
        for (int i = 0; i < solver.nx; i++)
        {
            for (int j = 0; j < solver.ny; j++)
            {
                double vel = Math.Sqrt(solver.u[i, j] * solver.u[i, j] + solver.v[i, j] * solver.v[i, j]);
                if (vel > maxVelBefore) maxVelBefore = vel;
            }
        }

        // 1. Solve the Shallow Water Equations (updates u, v, waterDepth)
        solver.NavierStokesStep(dt);

        // 2. Compute the Bed Shear Stress based on new flow fields
        double[,] tau = solver.ComputeShearStress();

        // 3. Compute the Sediment Flux Vector
        (double[,] qs_x, double[,] qs_y) = solver.ComputeSedimentFluxVector(tau);

        // 4. Solve the Exner Equation (updates h)
        // The solver updates h internally, returning the new elevation field
        // Note: The structure requires the PhysicsSolver class to handle the Exner equation update logic.
        // Assuming ExnerEquation updates solver.h internally and returns the updated array reference.
        (_, solver.h) = solver.ExnerEquation(qs_x, qs_y, dt);

        // 5. Validate and clamp all values to prevent instability
        ValidateAndClampValues();

        // Update the cached references (h is updated since it points to solver.h)
        
        stepCount++;
        
        // Log every 100 steps to verify simulation is running
        if (stepCount % 100 == 0)
        {
            double newMaxVel = 0.0;
            for (int i = 0; i < solver.nx; i++)
            {
                for (int j = 0; j < solver.ny; j++)
                {
                    double vel = Math.Sqrt(solver.u[i, j] * solver.u[i, j] + solver.v[i, j] * solver.v[i, j]);
                    if (vel > newMaxVel) newMaxVel = vel;
                }
            }
            
            Debug.Log($"[SimulationController] Step {stepCount}: Max velocity changed from {lastMaxVelocity:F6} to {newMaxVel:F6}");
            lastMaxVelocity = newMaxVel;
        }
    }
    
    private void ValidateAndClampValues()
    {
        if (solver == null) return;
        
        bool hadInvalidValues = false;
        double maxVelocity = 10.0;
        double maxDepth = 10.0;
        double maxElevation = 10.0;
        
        for (int i = 0; i < solver.nx; i++)
        {
            for (int j = 0; j < solver.ny; j++)
            {
                // Check for NaN or Infinity
                if (double.IsNaN(solver.u[i, j]) || double.IsInfinity(solver.u[i, j]))
                {
                    solver.u[i, j] = 0.0;
                    hadInvalidValues = true;
                }
                if (double.IsNaN(solver.v[i, j]) || double.IsInfinity(solver.v[i, j]))
                {
                    solver.v[i, j] = 0.0;
                    hadInvalidValues = true;
                }
                if (double.IsNaN(solver.waterDepth[i, j]) || double.IsInfinity(solver.waterDepth[i, j]))
                {
                    solver.waterDepth[i, j] = (solver.cellType[i, j] == RiverCellType.FLUID) ? InitialWaterDepth : 0.0;
                    hadInvalidValues = true;
                }
                if (double.IsNaN(solver.h[i, j]) || double.IsInfinity(solver.h[i, j]))
                {
                    solver.h[i, j] = 0.0;
                    hadInvalidValues = true;
                }
                
                // Clamp to reasonable ranges
                solver.u[i, j] = Math.Max(-maxVelocity, Math.Min(maxVelocity, solver.u[i, j]));
                solver.v[i, j] = Math.Max(-maxVelocity, Math.Min(maxVelocity, solver.v[i, j]));
                solver.waterDepth[i, j] = Math.Max(0.0, Math.Min(maxDepth, solver.waterDepth[i, j]));
                solver.h[i, j] = Math.Max(-maxElevation, Math.Min(maxElevation, solver.h[i, j]));
            }
        }
        
        if (hadInvalidValues && stepCount % 100 == 0)
        {
            Debug.LogWarning($"[SimulationController] Detected invalid values (NaN/Inf) at step {stepCount}. Values have been reset. Check TimeStep - it may be too large.");
        }
    }

    // --- Visualization ---

    private System.Collections.IEnumerator UpdateVisualizationWhenReady()
    {
        // Wait a frame to ensure MeshGrid has been initialized
        yield return null;
        
        // Wait until mesh dimensions match (check every frame for up to 1 second)
        float timeout = 1f;
        float elapsed = 0f;
        
        while (elapsed < timeout)
        {
            if (meshGrid != null && solver != null)
            {
                // Check if mesh is ready by verifying dimensions
                // This is a simple check - in a real scenario you might want a more robust check
                yield return null;
                elapsed += Time.deltaTime;
                
                // Try to update - UpdateMesh will return early if dimensions don't match
                UpdateVisualization();
                
                // If we got here without error, mesh is likely ready
                Debug.Log("[SimulationController] ✓ Mesh visualization updated");
                yield break;
            }
            yield return null;
            elapsed += Time.deltaTime;
        }
        
        Debug.LogWarning("[SimulationController] Timeout waiting for mesh to be ready");
    }
    
    private System.Collections.IEnumerator HideSimulationGridWhenReady()
    {
        // Wait a few frames for mesh to be created
        yield return new WaitForSeconds(0.5f);
        
        if (RiverGeometry != null)
        {
            GameObject gridMeshObject = RiverGeometry.transform.Find("SimulationGridMesh")?.gameObject;
            if (gridMeshObject != null)
            {
                gridMeshObject.SetActive(false);
                Debug.Log("[SimulationController] Simulation grid mesh hidden");
            }
            else
            {
                // Try finding it by component
                MeshRenderer[] renderers = RiverGeometry.GetComponentsInChildren<MeshRenderer>();
                foreach (var renderer in renderers)
                {
                    if (renderer.gameObject.name.Contains("SimulationGrid") || renderer.gameObject.name.Contains("Grid"))
                    {
                        renderer.gameObject.SetActive(false);
                        Debug.Log("[SimulationController] Simulation grid mesh hidden (found by component)");
                        break;
                    }
                }
            }
        }
    }

    private int visualizationUpdateCount = 0;
    private float lastVizLogTime = 0f;

    private void UpdateVisualization()
    {
        // Update simulation grid mesh only if enabled
        if (ShowSimulationGrid && meshGrid != null && solver != null)
        {
            // Pass all relevant data arrays to the MeshGrid for visualization
            meshGrid.UpdateMesh(h, waterDepth, u, v, solver.cellType);
            
            visualizationUpdateCount++;
            
            // Log visualization updates periodically
            if (Time.time - lastVizLogTime > 5f)
            {
                Debug.Log($"[SimulationController] Visualization updated {visualizationUpdateCount} times. MeshGrid: {(meshGrid != null ? "OK" : "NULL")}, Solver: {(solver != null ? "OK" : "NULL")}");
                lastVizLogTime = Time.time;
            }
        }
        else if (meshGrid != null && solver != null)
        {
            // Grid is hidden but we still track updates
            visualizationUpdateCount++;
        }
        
        // Update terrain mesh if enabled
        if (EnableTerrain && terrainMesh != null && solver != null && riverToGrid != null)
        {
            float cellSize = riverToGrid.GetCellSize();
            terrainMesh.UpdateTerrain(h, solver.cellType, cellSize, riverToGrid.MinX, riverToGrid.MinZ, GridWidth, GridHeight);
        }
        
        // Update the original red river geometry if enabled
        if (UpdateRiverGeometry && geometryLoader != null && solver != null)
        {
            UpdateRiverGeometryMesh();
        }
    }
    
    /// <summary>
    /// Updates the original red river geometry mesh based on simulation results.
    /// Maps simulation grid elevations to river geometry vertices.
    /// Works with both unstructured meshes (original strip mesh) and grid meshes (remeshed structure).
    /// </summary>
    private void UpdateRiverGeometryMesh()
    {
        Mesh riverMesh = geometryLoader.GetMesh();
        if (riverMesh == null || riverMesh.vertices == null)
        {
            return;
        }
        
        Vector3[] riverVertices = riverMesh.vertices;
        float cellSize = riverToGrid.GetCellSize();
        float minX = riverToGrid.MinX;
        float minZ = riverToGrid.MinZ;
        
        // Update each vertex in the river geometry based on nearest grid cell elevation
        for (int i = 0; i < riverVertices.Length; i++)
        {
            Vector3 vertex = riverVertices[i];
            
            // Convert vertex world position to grid coordinates
            float gridX = (vertex.x - minX) / cellSize;
            float gridZ = (vertex.z - minZ) / cellSize;
            
            // Clamp to valid grid bounds
            int xIdx = Mathf.Clamp(Mathf.RoundToInt(gridX), 0, solver.nx - 1);
            int zIdx = Mathf.Clamp(Mathf.RoundToInt(gridZ), 0, solver.ny - 1);
            
            // Get bed elevation from simulation (h) and apply scale factor
            float bedElevation = (float)solver.h[xIdx, zIdx] * RiverGeometryElevationScale;
            
            // Update vertex Y coordinate (elevation)
            // Preserve original elevation offset if needed, or use bed elevation directly
            riverVertices[i] = new Vector3(vertex.x, bedElevation, vertex.z);
        }
        
        // Apply updated vertices to mesh
        riverMesh.vertices = riverVertices;
        riverMesh.RecalculateNormals();
        riverMesh.RecalculateBounds();
        
        // Update the MeshFilter to reflect changes
        MeshFilter mf = RiverGeometry.GetComponent<MeshFilter>();
        if (mf != null && mf.mesh == riverMesh)
        {
            // Mesh is already assigned, just mark as dirty
            mf.sharedMesh = riverMesh;
        }
    }

    // --- Public Getters (Optional, for external visualization/debugging) ---

    public PhysicsSolver GetSolver()
    {
        return solver;
    }

    public double[,] GetBedElevation() => h;
    public double[,] GetWaterDepth() => waterDepth;
    public double[,] GetVelocityU() => u;
    public double[,] GetVelocityV() => v;
}
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
    
    private MeshGrid meshGrid;
    private RiverToGrid riverToGrid;

    // --- Configuration Parameters ---
    [Header("Simulation Control")]
    [Tooltip("Toggle to start/stop the simulation. When off, mesh will display but simulation steps won't run.")]
    public bool RunSimulation = false;

    [Header("Grid & Time")]
    // NOTE: GridWidth and GridHeight will be OVERRIDDEN by RiverToGrid calculations.
    [HideInInspector] public int GridWidth;
    [HideInInspector] public int GridHeight;

    public float CellSize = 0.5f;
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
    }

    /// <summary>
    /// Handles mesh update event from MeshGrid (called when mesh is first created or updated).
    /// </summary>
    private void HandleMeshUpdated(Mesh mesh)
    {
        // Mesh has been updated, simulation will use it in Update()
        // This is mainly for initialization tracking
    }

    void Update()
    {
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

        Debug.Log("[SimulationController] Step 4: Waiting for mesh to be ready before updating visualization...");

        // Don't update visualization immediately - wait for MeshGrid to be initialized via event
        // The MeshGrid will be initialized by the HandleGridInitialized event handler
        // We'll update visualization in the next frame or when mesh is ready
        StartCoroutine(UpdateVisualizationWhenReady());

        Debug.Log($"[SimulationController] ✓ Initialization complete: Grid Size {GridWidth}x{GridHeight}. Simulation is {(RunSimulation ? "RUNNING" : "PAUSED")}.");
    }

    // --- Core Simulation Logic ---

    private void RunTimeStep(float dt)
    {
        // Safety check: ensure solver is initialized
        if (solver == null)
        {
            Debug.LogWarning("RunTimeStep called but solver is not initialized. Check initialization errors.");
            return;
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

        // Update the cached references (h is updated since it points to solver.h)
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

    private void UpdateVisualization()
    {
        if (meshGrid != null && solver != null)
        {
            // Pass all relevant data arrays to the MeshGrid for visualization
            meshGrid.UpdateMesh(h, waterDepth, u, v, solver.cellType);
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
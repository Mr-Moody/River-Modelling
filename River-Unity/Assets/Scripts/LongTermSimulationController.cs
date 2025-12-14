using UnityEngine;
using System;
using System.IO;

/// <summary>
/// Controller for running long-term deterministic river simulations (e.g., 30 years from 1987-2017).
/// Uses day-based time steps (1-7 days) and automatically exports final geometry for comparison.
/// </summary>
public class LongTermSimulationController : MonoBehaviour
{
    [Header("Simulation Configuration")]
    [Tooltip("Reference to the SimulationController component")]
    public SimulationController simulationController;
    
    [Tooltip("Time step in days (1-7 days)")]
    [Range(1, 7)]
    public int timeStepDays = 1;
    
    [Tooltip("Total simulation duration in years")]
    [Range(1, 100)]
    public int simulationYears = 30;
    
    [Header("Export Configuration")]
    [Tooltip("Automatically export final geometry when simulation completes")]
    public bool autoExportOnComplete = true;
    
    [Tooltip("Base filename for exported geometry (timestamp will be appended)")]
    public string exportFileName = "RiverSimulation_2017";
    
    [Tooltip("Base path for export (empty = use Application.persistentDataPath)")]
    public string exportBasePath = "";
    
    [Header("Performance")]
    [Tooltip("Number of physics steps to run per Unity frame (higher = faster but may cause instability)")]
    [Range(1, 100)]
    public int stepsPerFrame = 10;
    
    [Tooltip("Update mesh visualization every N steps (higher = better performance, lower = smoother visualization)")]
    [Range(1, 1000)]
    public int visualizationUpdateInterval = 10;
    
    [Header("Monte Carlo Control")]
    [Tooltip("GameObject containing Monte Carlo components (will be disabled when long-term simulation runs)")]
    public GameObject monteCarloGameObject;
    
    // Runtime state
    private bool isRunning = false;
    private float elapsedSimulationTime = 0f; // In seconds
    private float totalSimulationTime = 0f; // In seconds
    private int currentStep = 0;
    private int totalSteps = 0;
    private float physicsTimeStep = 0f; // In seconds (converted from days)
    private float simulationStartTime = 0f; // Real time when simulation started (Time.time)
    
    // Events
    public event Action OnSimulationStarted;
    public event Action OnSimulationCompleted;
    public event Action<int, float, float> OnProgressUpdated; // (currentStep, elapsedYears, progressPercent)
    
    void Start()
    {
        // Auto-find SimulationController if not assigned
        if (simulationController == null)
        {
            simulationController = FindFirstObjectByType<SimulationController>();
            if (simulationController == null)
            {
                Debug.LogError("[LongTermSimulationController] SimulationController not found! Please assign it in the Inspector.");
            }
        }
        
        // Auto-find Monte Carlo GameObject if not assigned
        if (monteCarloGameObject == null)
        {
            MonteCarloSimulationManager mcManager = FindFirstObjectByType<MonteCarloSimulationManager>();
            if (mcManager != null)
            {
                monteCarloGameObject = mcManager.gameObject;
                Debug.Log($"[LongTermSimulationController] Auto-found Monte Carlo GameObject: {monteCarloGameObject.name}");
            }
        }
    }
    
    /// <summary>
    /// Starts the long-term simulation.
    /// </summary>
    public void StartLongTermSimulation()
    {
        if (isRunning)
        {
            Debug.LogWarning("[LongTermSimulationController] Simulation is already running!");
            return;
        }
        
        if (simulationController == null)
        {
            Debug.LogError("[LongTermSimulationController] Cannot start simulation - SimulationController is null!");
            return;
        }
        
        // Get solver to verify it's initialized
        RiverMeshPhysicsSolver solver = simulationController.GetSolver();
        if (solver == null)
        {
            Debug.LogError("[LongTermSimulationController] Cannot start simulation - solver is not initialized! Make sure the river mesh is loaded.");
            return;
        }
        
        // Convert days to seconds for physics solver
        physicsTimeStep = timeStepDays * 86400f; // 86400 seconds per day
        
        // Calculate total steps and time
        totalSteps = (simulationYears * 365) / timeStepDays;
        totalSimulationTime = simulationYears * 365f * 86400f; // Total in seconds
        
        // Reset state
        currentStep = 0;
        elapsedSimulationTime = 0f;
        simulationStartTime = Time.time;
        isRunning = true;
        
        Debug.Log($"[LongTermSimulationController] ═══════════════════════════════════════════════════════════");
        Debug.Log($"[LongTermSimulationController] STARTING LONG-TERM SIMULATION");
        Debug.Log($"[LongTermSimulationController] ═══════════════════════════════════════════════════════════");
        Debug.Log($"[LongTermSimulationController] Duration: {simulationYears} years");
        Debug.Log($"[LongTermSimulationController] Time Step: {timeStepDays} day(s) ({physicsTimeStep:F0} seconds)");
        Debug.Log($"[LongTermSimulationController] Total Steps: {totalSteps:N0}");
        Debug.Log($"[LongTermSimulationController] Steps per Frame: {stepsPerFrame}");
        Debug.Log($"[LongTermSimulationController] ═══════════════════════════════════════════════════════════");
        
        // Ensure simulation controller's RunSimulation is off (we'll control it manually)
        simulationController.RunSimulation = false;
        
        // Disable Monte Carlo GameObject if assigned
        if (monteCarloGameObject != null)
        {
            monteCarloGameObject.SetActive(false);
            Debug.Log($"[LongTermSimulationController] Disabled Monte Carlo GameObject: {monteCarloGameObject.name}");
        }
        
        OnSimulationStarted?.Invoke();
    }
    
    /// <summary>
    /// Stops the long-term simulation early.
    /// </summary>
    public void StopLongTermSimulation()
    {
        if (!isRunning)
        {
            Debug.LogWarning("[LongTermSimulationController] Cannot stop - simulation is not running.");
            return;
        }
        
        Debug.Log($"[LongTermSimulationController] Simulation stopped at step {currentStep}/{totalSteps}");
        isRunning = false;
        
        // Re-enable Monte Carlo GameObject if assigned
        if (monteCarloGameObject != null)
        {
            monteCarloGameObject.SetActive(true);
            Debug.Log($"[LongTermSimulationController] Re-enabled Monte Carlo GameObject: {monteCarloGameObject.name}");
        }
        
        // Export current state if enabled
        if (autoExportOnComplete)
        {
            ExportCurrentGeometry("_stopped");
        }
    }
    
    void Update()
    {
        if (!isRunning || simulationController == null)
        {
            return;
        }
        
        RiverMeshPhysicsSolver solver = simulationController.GetSolver();
        if (solver == null)
        {
            Debug.LogError("[LongTermSimulationController] Solver became null during simulation!");
            StopLongTermSimulation();
            return;
        }
        
        // Run multiple physics steps per frame for performance
        for (int i = 0; i < stepsPerFrame && currentStep < totalSteps; i++)
        {
            RunSimulationStep(solver);
            currentStep++;
            elapsedSimulationTime += physicsTimeStep;
            
            // Check if simulation is complete
            if (currentStep >= totalSteps)
            {
                CompleteSimulation();
                return;
            }
        }
        
        // Update visualization periodically
        if (currentStep % visualizationUpdateInterval == 0)
        {
            UpdateVisualization();
        }
        
        // Report progress periodically (every 100 steps)
        if (currentStep % 100 == 0)
        {
            float elapsedYears = elapsedSimulationTime / (365f * 86400f);
            float progressPercent = (currentStep / (float)totalSteps) * 100f;
            OnProgressUpdated?.Invoke(currentStep, elapsedYears, progressPercent);
            
            // Calculate estimated time to completion
            float elapsedRealTime = Time.time - simulationStartTime;
            string estimatedTimeRemaining = "Calculating...";
            
            if (currentStep > 0 && elapsedRealTime > 0f)
            {
                float stepsPerSecond = currentStep / elapsedRealTime;
                int remainingSteps = totalSteps - currentStep;
                float estimatedSecondsRemaining = remainingSteps / stepsPerSecond;
                
                // Format time estimate
                if (estimatedSecondsRemaining < 60f)
                {
                    estimatedTimeRemaining = $"{estimatedSecondsRemaining:F0} seconds";
                }
                else if (estimatedSecondsRemaining < 3600f)
                {
                    float minutes = estimatedSecondsRemaining / 60f;
                    estimatedTimeRemaining = $"{minutes:F1} minutes";
                }
                else
                {
                    float hours = estimatedSecondsRemaining / 3600f;
                    estimatedTimeRemaining = $"{hours:F2} hours";
                }
            }
            
            Debug.Log($"[LongTermSimulationController] Progress: {elapsedYears:F2} years / {simulationYears} years " +
                     $"({progressPercent:F1}%) - Step {currentStep}/{totalSteps} - " +
                     $"Estimated time remaining: {estimatedTimeRemaining}");
        }
    }
    
    /// <summary>
    /// Runs a single physics simulation step.
    /// </summary>
    private void RunSimulationStep(RiverMeshPhysicsSolver solver)
    {
        // Run Navier-Stokes step in river-local coordinates
        solver.NavierStokesStep(physicsTimeStep);
        
        // Compute bed shear stress
        double[,] tau = solver.ComputeShearStress();
        
        // Compute sediment flux vector
        (double[,] qs_s, double[,] qs_w) = solver.ComputeSedimentFluxVector(tau);
        
        // Solve Exner equation (updates bed elevation h)
        (_, solver.h) = solver.ExnerEquation(qs_s, qs_w, physicsTimeStep);
    }
    
    /// <summary>
    /// Updates the mesh visualization.
    /// </summary>
    private void UpdateVisualization()
    {
        if (simulationController != null)
        {
            simulationController.ForceMeshUpdate();
        }
    }
    
    /// <summary>
    /// Completes the simulation and exports final geometry.
    /// </summary>
    private void CompleteSimulation()
    {
        isRunning = false;
        
        float elapsedYears = elapsedSimulationTime / (365f * 86400f);
        
        Debug.Log($"[LongTermSimulationController] ═══════════════════════════════════════════════════════════");
        Debug.Log($"[LongTermSimulationController] SIMULATION COMPLETE");
        Debug.Log($"[LongTermSimulationController] ═══════════════════════════════════════════════════════════");
        Debug.Log($"[LongTermSimulationController] Total Steps: {currentStep}");
        Debug.Log($"[LongTermSimulationController] Elapsed Time: {elapsedYears:F2} years");
        Debug.Log($"[LongTermSimulationController] ═══════════════════════════════════════════════════════════");
        
        // Update final visualization
        UpdateVisualization();
        
        // Re-enable Monte Carlo GameObject if assigned
        if (monteCarloGameObject != null)
        {
            monteCarloGameObject.SetActive(true);
            Debug.Log($"[LongTermSimulationController] Re-enabled Monte Carlo GameObject: {monteCarloGameObject.name}");
        }
        
        // Export final geometry if enabled
        if (autoExportOnComplete)
        {
            ExportCurrentGeometry();
        }
        
        OnSimulationCompleted?.Invoke();
    }
    
    /// <summary>
    /// Exports the current river geometry to CSV.
    /// </summary>
    private void ExportCurrentGeometry(string suffix = "")
    {
        if (simulationController == null)
        {
            Debug.LogError("[LongTermSimulationController] Cannot export - SimulationController is null!");
            return;
        }
        
        // Generate filename with timestamp
        DateTime now = DateTime.Now;
        string timestamp = now.ToString("yyyyMMdd_HHmmss");
        string fileName = $"{exportFileName}{suffix}_{timestamp}.csv";
        
        // Determine export path
        string basePath = string.IsNullOrEmpty(exportBasePath) 
            ? Application.persistentDataPath 
            : exportBasePath;
        
        string fullPath = Path.Combine(basePath, fileName);
        
        // Use SimulationController's export method
        simulationController.ExportCurrentRiverMesh(fullPath);
        
        Debug.Log($"[LongTermSimulationController] Geometry exported to: {fullPath}");
    }
    
    /// <summary>
    /// Gets the current simulation progress.
    /// </summary>
    public (int currentStep, int totalSteps, float elapsedYears, float progressPercent) GetProgress()
    {
        float elapsedYears = elapsedSimulationTime / (365f * 86400f);
        float progressPercent = totalSteps > 0 ? (currentStep / (float)totalSteps) * 100f : 0f;
        return (currentStep, totalSteps, elapsedYears, progressPercent);
    }
    
    /// <summary>
    /// Checks if the simulation is currently running.
    /// </summary>
    public bool IsRunning => isRunning;
}

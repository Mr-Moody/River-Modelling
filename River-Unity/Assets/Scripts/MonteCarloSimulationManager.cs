using UnityEngine;
using System;
using System.Collections.Generic;

/// <summary>
/// Monte Carlo Simulation Manager for Uncertainty Quantification in River Morphodynamics.
/// 
/// This framework treats the sediment transport coefficient from the governing Exner equation
/// as a random variable with a log-normal distribution rather than a deterministic constant.
/// Additionally, 3D Perlin noise generates randomized initial bed elevation fields.
/// 
/// Governing Equation: Exner Equation
///   dh/dt = -1/(1-λ) * ∇·qs
/// 
/// Where sediment flux qs uses Meyer-Peter Müller formulation:
///   qs = K * (τ - τc)^1.5
/// 
/// The transport coefficient K is treated as:
///   K ~ LogNormal(μ, σ)
/// 
/// Over each Monte Carlo realization, K and the initial bed elevation field are varied
/// to produce an ensemble of predicted river states rather than a single deterministic outcome.
/// </summary>
public class MonteCarloSimulationManager : MonoBehaviour
{
    // ═══════════════════════════════════════════════════════════════════════════
    // CONFIGURATION
    // ═══════════════════════════════════════════════════════════════════════════
    
    [Header("Monte Carlo Configuration")]
    [Tooltip("Number of Monte Carlo realizations to run")]
    [Range(10, 1000)]
    public int numRealizations = 100;
    
    [Tooltip("Number of simulation time steps per realization")]
    [Range(100, 10000)]
    public int stepsPerRealization = 1000;
    
    [Tooltip("Time step for each simulation step")]
    [Range(0.001f, 0.1f)]
    public float timeStep = 0.01f;
    
    [Tooltip("Random seed for reproducibility (0 = use system time)")]
    public int masterSeed = 0;
    
    [Header("Transport Coefficient Distribution (Log-Normal)")]
    [Tooltip("Mean of the underlying normal distribution for ln(K)")]
    public double transportCoefficientMu = -2.3;  // ln(0.1) ≈ -2.3, so mean K ≈ 0.1
    
    [Tooltip("Standard deviation of the underlying normal distribution for ln(K)")]
    [Range(0.01f, 1.0f)]
    public float transportCoefficientSigma = 0.3f;
    
    [Header("Bed Elevation Perlin Noise")]
    [Tooltip("Enable randomized initial bed elevation using 3D Perlin noise")]
    public bool enablePerlinBedElevation = true;
    
    [Tooltip("Amplitude of Perlin noise bed perturbations (meters)")]
    [Range(0.01f, 1.0f)]
    public float perlinAmplitude = 0.1f;
    
    [Tooltip("Frequency/scale of Perlin noise (higher = more detail)")]
    [Range(0.1f, 10f)]
    public float perlinFrequency = 1.0f;
    
    [Tooltip("Number of octaves for fractal Perlin noise")]
    [Range(1, 8)]
    public int perlinOctaves = 4;
    
    [Tooltip("Persistence for fractal noise (amplitude decay per octave)")]
    [Range(0.1f, 0.9f)]
    public float perlinPersistence = 0.5f;

    [Tooltip("Exponent applied to noise to shape relief (>=1)")]
    [Range(1f, 5f)]
    public float perlinMountainExponent = 1.0f;
    
    [Header("References")]
    [Tooltip("Reference to the main SimulationController")]
    public SimulationController simulationController;
    
    [Header("Output Configuration")]
    [Tooltip("Store full ensemble results (memory intensive for large ensembles)")]
    public bool storeFullEnsemble = false;
    
    [Tooltip("Compute and store statistics every N steps")]
    [Range(1, 100)]
    public int statisticsInterval = 10;
    
    [Header("Export Configuration")]
    [Tooltip("Automatically export results to CSV files when ensemble completes")]
    public bool autoExportResults = true;
    
    [Tooltip("Base path for export folder (empty = use Application.persistentDataPath)")]
    public string exportBasePath = "";
    
    [Tooltip("Export full field data to CSV files (requires storeFullEnsemble=true or will enable it automatically)")]
    public bool exportFullFields = true;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // RUNTIME STATE
    // ═══════════════════════════════════════════════════════════════════════════
    
    private System.Random masterRandom;
    private bool isRunning = false;
    private int currentRealization = 0;
    private int currentStep = 0;
    
    // Time tracking
    private float totalSimulationTime = 0f;
    private float elapsedSimulationTime = 0f;
    
    // Ensemble storage
    private MonteCarloEnsemble ensemble;
    
    // Current realization parameters
    private double currentTransportCoefficient;
    private int currentPerlinSeed;
    
    // Export path
    private string resultsFolderPath = "";
    
    // Events for external monitoring
    public event Action<int, int> OnRealizationStarted;      // (realization index, total)
    public event Action<int, MonteCarloRealizationResult> OnRealizationCompleted;
    public event Action<MonteCarloEnsemble> OnEnsembleCompleted;
    public event Action<int, int, float> OnProgressUpdated;  // (realization, step, progress%)
    
    // ═══════════════════════════════════════════════════════════════════════════
    // UNITY LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════════
    
    void Awake()
    {
        // Initialize master random generator
        if (masterSeed == 0)
        {
            masterSeed = Environment.TickCount;
        }
        masterRandom = new System.Random(masterSeed);
        
        Debug.Log($"[MonteCarloManager] Initialized with master seed: {masterSeed}");
    }
    
    void Start()
    {
        // Find SimulationController if not assigned
        if (simulationController == null)
        {
            simulationController = FindFirstObjectByType<SimulationController>();
            if (simulationController == null)
            {
                Debug.LogError("[MonteCarloManager] SimulationController not found! Assign it in the Inspector.");
            }
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // PUBLIC API
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// <summary>
    /// Starts the Monte Carlo ensemble simulation.
    /// </summary>
    public void StartEnsembleSimulation()
    {
        if (isRunning)
        {
            Debug.LogWarning("[MonteCarloManager] Ensemble simulation already running!");
            return;
        }
        
        if (simulationController == null)
        {
            Debug.LogError("[MonteCarloManager] Cannot start - SimulationController not assigned!");
            return;
        }
        
        Debug.Log($"[MonteCarloManager] Starting Monte Carlo ensemble with {numRealizations} realizations...");
        Debug.Log($"[MonteCarloManager] Transport Coefficient Distribution: LogNormal(μ={transportCoefficientMu:F3}, σ={transportCoefficientSigma:F3})");
        Debug.Log($"[MonteCarloManager] Perlin Noise: Enabled={enablePerlinBedElevation}, Amp={perlinAmplitude:F3}, Freq={perlinFrequency:F2}");
        
        // Calculate total simulation time
        totalSimulationTime = numRealizations * stepsPerRealization * timeStep;
        elapsedSimulationTime = 0f;
        Debug.Log($"[MonteCarloManager] Total simulation time: {totalSimulationTime:F2}s ({numRealizations} realizations × {stepsPerRealization} steps × {timeStep:F4}s/step)");
        
        // Create results folder if export is enabled
        if (autoExportResults)
        {
            resultsFolderPath = MonteCarloCSVExporter.CreateResultsFolder(exportBasePath);
            if (!string.IsNullOrEmpty(resultsFolderPath))
            {
                Debug.Log($"[MonteCarloManager] Results will be exported to: {resultsFolderPath}");
            }
            else
            {
                Debug.LogWarning("[MonteCarloManager] Failed to create results folder. Export may fail.");
            }
        }
        
        // Ensure full fields are stored if export is enabled
        bool shouldStoreFullFields = storeFullEnsemble || (exportFullFields && autoExportResults);
        if (shouldStoreFullFields && !storeFullEnsemble)
        {
            Debug.Log("[MonteCarloManager] Enabling full field storage for CSV export (storeFullEnsemble will be enabled automatically)");
        }
        
        // Initialize ensemble storage
        ensemble = new MonteCarloEnsemble(numRealizations, shouldStoreFullFields);
        
        // Start first realization
        currentRealization = 0;
        isRunning = true;
        
        StartNextRealization();
    }
    
    /// <summary>
    /// Stops the Monte Carlo simulation.
    /// </summary>
    public void StopEnsembleSimulation()
    {
        Debug.Log("[MonteCarloManager] ===== STOP REQUESTED =====");
        
        if (!isRunning)
        {
            Debug.LogWarning("[MonteCarloManager] Stop requested but simulation is not running.");
            return;
        }
        
        Debug.Log($"[MonteCarloManager] Stopping simulation at realization {currentRealization + 1}/{numRealizations}");
        isRunning = false;
        
        if (simulationController != null)
        {
            simulationController.RunSimulation = false;
            Debug.Log("[MonteCarloManager] Set RunSimulation = false");
        }
        else
        {
            Debug.LogError("[MonteCarloManager] Cannot stop - SimulationController is null!");
        }
        
        ResetMonteCarloTransportCoefficient();
        
        Debug.Log($"[MonteCarloManager] Ensemble simulation stopped at realization {currentRealization + 1}/{numRealizations}");
    }
    
    /// <summary>
    /// Gets the current ensemble results.
    /// </summary>
    public MonteCarloEnsemble GetEnsemble()
    {
        return ensemble;
    }

    /// <summary>
    /// Gets the latest realization result (useful for UI hooks).
    /// </summary>
    public MonteCarloRealizationResult GetLatestResult()
    {
        return ensemble?.LatestRealization;
    }

    /// <summary>
    /// Gets the latest snapshot if full data is being stored.
    /// </summary>
    public MonteCarloStateSnapshot GetLatestSnapshot()
    {
        return ensemble?.LatestRealization?.Snapshot;
    }
    
    /// <summary>
    /// Samples a transport coefficient from the log-normal distribution.
    /// </summary>
    public double SampleTransportCoefficient()
    {
        return SampleLogNormal(transportCoefficientMu, transportCoefficientSigma);
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // INTERNAL MONTE CARLO LOGIC
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// <summary>
    /// Starts the next realization in the ensemble.
    /// </summary>
    private void StartNextRealization()
    {
        if (currentRealization >= numRealizations)
        {
            Debug.Log($"[MonteCarloManager] All realizations complete ({currentRealization}/{numRealizations}), completing ensemble...");
            CompleteEnsemble();
            return;
        }
        
        float timePercentage = totalSimulationTime > 0 ? (elapsedSimulationTime / totalSimulationTime * 100f) : 0f;
        Debug.Log($"[MonteCarloManager] ===== Starting Stage {currentRealization + 1} / {numRealizations} =====");
        Debug.Log($"[MonteCarloManager] Time: {elapsedSimulationTime:F2}s / {totalSimulationTime:F2}s ({timePercentage:F1}%)");
        
        // Sample random parameters for this realization
        currentTransportCoefficient = SampleTransportCoefficient();
        currentPerlinSeed = masterRandom.Next();
        
        Debug.Log($"[MonteCarloManager] Realization {currentRealization + 1}/{numRealizations}: " +
                  $"K={currentTransportCoefficient:F6}, PerlinSeed={currentPerlinSeed}");
        
        // Fire event
        Debug.Log($"[MonteCarloManager] Firing OnRealizationStarted event for stage {currentRealization + 1}");
        OnRealizationStarted?.Invoke(currentRealization, numRealizations);
        
        // Reset and configure the solver
        Debug.Log($"[MonteCarloManager] Resetting solver with new parameters...");
        ResetSolverWithParameters(currentTransportCoefficient, currentPerlinSeed);
        
        // Start simulation
        currentStep = 0;
        Debug.Log($"[MonteCarloManager] Setting RunSimulation = true for stage {currentRealization + 1}");
        simulationController.RunSimulation = true;
    }
    
    /// <summary>
    /// Resets the solver with new Monte Carlo parameters.
    /// </summary>
    private void ResetSolverWithParameters(double transportCoeff, int perlinSeed)
    {
        RiverMeshPhysicsSolver solver = simulationController.GetSolver();
        if (solver == null)
        {
            Debug.LogError("[MonteCarloManager] Solver not available!");
            return;
        }
        
        // Apply the sampled transport coefficient
        // Note: Since transportCoefficient is readonly in the solver, we need to use reflection
        // or create a new solver. For this implementation, we'll use a wrapper approach.
        ApplyTransportCoefficient(solver, transportCoeff);
        
        // Apply Perlin noise to initial bed elevation
        if (enablePerlinBedElevation)
        {
            ApplyPerlinBedElevation(solver, perlinSeed);
        }
        
        // Reset flow fields to initial conditions
        ResetFlowFields(solver);
    }
    
    /// <summary>
    /// Applies the transport coefficient to the solver via MonteCarloParameters hook.
    /// </summary>
    private void ApplyTransportCoefficient(RiverMeshPhysicsSolver solver, double transportCoeff)
    {
        // Store in a static/accessible location for the solver to use via its property
        MonteCarloParameters.CurrentTransportCoefficient = transportCoeff;
        MonteCarloParameters.UseMonteCarloTransportCoefficient = true;
        Debug.Log($"[MonteCarloManager] Applied transport coefficient: {transportCoeff:F6}");
    }

    /// <summary>
    /// Resets Monte Carlo transport coefficient override.
    /// </summary>
    private void ResetMonteCarloTransportCoefficient()
    {
        MonteCarloParameters.UseMonteCarloTransportCoefficient = false;
    }
    
    /// <summary>
    /// Applies Perlin noise to the initial bed elevation field using provided formula.
    /// </summary>
    private void ApplyPerlinBedElevation(RiverMeshPhysicsSolver solver, int seed)
    {
        int numCrossSections = solver.numCrossSections;
        int widthResolution = solver.widthResolution;
        
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                if (solver.cellType[i, w] != RiverCellType.FLUID)
                {
                    continue;
                }

                // Map indices to normalized coords
                float nx = (i / (float)numCrossSections);
                float ny = (w / (float)widthResolution);
                float nz = seed * 0.001f;

                float noise = GeneratePerlinNoise(nx, ny, nz, seed);
                solver.h[i, w] += noise * perlinAmplitude;
            }
        }
        
        Debug.Log($"[MonteCarloManager] Applied Perlin noise bed elevation (seed={seed}, amp={perlinAmplitude:F3})");
    }

    /// <summary>
    /// Generates Perlin noise using octave blend and exponent shaping.
    /// Mirrors the provided GenerateNoise formulation.
    /// </summary>
    private float GeneratePerlinNoise(float x, float y, float z, int seed)
    {
        float noise = 0f, freq = 1f, amp = 1f;
        for (int o = 0; o < perlinOctaves; o++)
        {
            float n = (
                Mathf.PerlinNoise((x + 1f) * perlinFrequency * freq + seed,
                                  (y + 1f) * perlinFrequency * freq + seed) +
                Mathf.PerlinNoise((y + 1f) * perlinFrequency * freq + seed,
                                  (z + 1f) * perlinFrequency * freq + seed) +
                Mathf.PerlinNoise((x + 1f) * perlinFrequency * freq + seed,
                                  (z + 1f) * perlinFrequency * freq + seed)
            ) / 3f;

            noise += n * amp;
            freq *= 2f;
            amp *= perlinPersistence;
        }

        return Mathf.Pow(noise, perlinMountainExponent);
    }
    
    /// <summary>
    /// Resets flow fields to initial conditions.
    /// </summary>
    private void ResetFlowFields(RiverMeshPhysicsSolver solver)
    {
        int numCrossSections = solver.numCrossSections;
        int widthResolution = solver.widthResolution;
        
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                if (solver.cellType[i, w] == RiverCellType.FLUID)
                {
                    solver.waterDepth[i, w] = simulationController.InitialWaterDepth;
                    solver.u[i, w] = 0.1;  // Initial downstream velocity
                    solver.v[i, w] = 0.0;  // No cross-flow
                }
                else
                {
                    solver.waterDepth[i, w] = 0.0;
                    solver.u[i, w] = 0.0;
                    solver.v[i, w] = 0.0;
                }
            }
        }
    }
    
    void Update()
    {
        if (!isRunning || simulationController == null)
        {
            return;
        }
        
        // Check if we should advance the simulation
        if (simulationController.RunSimulation)
        {
            currentStep++;
            
            // Update elapsed time
            elapsedSimulationTime += timeStep;
            
            // Report progress
            float progress = (currentRealization * stepsPerRealization + currentStep) / 
                            (float)(numRealizations * stepsPerRealization) * 100f;
            OnProgressUpdated?.Invoke(currentRealization, currentStep, progress);
            
            // Log time progress periodically (every 100 steps)
            if (currentStep % 100 == 0)
            {
                float timePercentage = totalSimulationTime > 0 ? (elapsedSimulationTime / totalSimulationTime * 100f) : 0f;
                Debug.Log($"[MonteCarloManager] Progress: Stage {currentRealization + 1}/{numRealizations}, Step {currentStep}/{stepsPerRealization}, " +
                         $"Time: {elapsedSimulationTime:F2}s / {totalSimulationTime:F2}s ({timePercentage:F1}%)");
            }
            
            // Collect statistics at intervals
            if (currentStep % statisticsInterval == 0)
            {
                CollectIntermediateStatistics();
            }
            
            // Check if realization is complete
            if (currentStep >= stepsPerRealization)
            {
                CompleteCurrentRealization();
            }
        }
    }
    
    /// <summary>
    /// Collects intermediate statistics during a realization.
    /// </summary>
    private void CollectIntermediateStatistics()
    {
        RiverMeshPhysicsSolver solver = simulationController.GetSolver();
        if (solver == null) return;
        
        // Compute current bed elevation statistics
        double meanH = 0, varH = 0;
        int count = 0;
        
        for (int i = 0; i < solver.numCrossSections; i++)
        {
            for (int w = 0; w < solver.widthResolution; w++)
            {
                if (solver.cellType[i, w] == RiverCellType.FLUID)
                {
                    meanH += solver.h[i, w];
                    count++;
                }
            }
        }
        
        if (count > 0)
        {
            meanH /= count;
            
            // Compute variance
            for (int i = 0; i < solver.numCrossSections; i++)
            {
                for (int w = 0; w < solver.widthResolution; w++)
                {
                    if (solver.cellType[i, w] == RiverCellType.FLUID)
                    {
                        double diff = solver.h[i, w] - meanH;
                        varH += diff * diff;
                    }
                }
            }
            varH /= count;
        }
    }
    
    /// <summary>
    /// Completes the current realization and stores results.
    /// </summary>
    private void CompleteCurrentRealization()
    {
        float timePercentage = totalSimulationTime > 0 ? (elapsedSimulationTime / totalSimulationTime * 100f) : 0f;
        Debug.Log($"[MonteCarloManager] ===== Completing Stage {currentRealization + 1} / {numRealizations} =====");
        Debug.Log($"[MonteCarloManager] Time: {elapsedSimulationTime:F2}s / {totalSimulationTime:F2}s ({timePercentage:F1}%)");
        
        RiverMeshPhysicsSolver solver = simulationController.GetSolver();
        if (solver == null)
        {
            Debug.LogError("[MonteCarloManager] Cannot complete realization - solver is null!");
            return;
        }
        
        // Stop this realization
        Debug.Log($"[MonteCarloManager] Stopping simulation for stage {currentRealization + 1}");
        simulationController.RunSimulation = false;
        
        // Determine if we need to store full fields (for export or ensemble storage)
        bool shouldStoreFullFields = storeFullEnsemble || (exportFullFields && autoExportResults);
        
        // Create result object
        Debug.Log($"[MonteCarloManager] Creating result object for stage {currentRealization + 1}...");
        MonteCarloRealizationResult result = new MonteCarloRealizationResult(
            currentRealization,
            currentTransportCoefficient,
            currentPerlinSeed,
            solver,
            shouldStoreFullFields
        );
        
        // Store in ensemble
        ensemble.AddRealization(result);
        
        // Export full field data for this stage if enabled
        if (autoExportResults && exportFullFields && !string.IsNullOrEmpty(resultsFolderPath))
        {
            bool exportSuccess = MonteCarloCSVExporter.ExportFullFieldData(result, resultsFolderPath, currentRealization + 1);
            if (!exportSuccess)
            {
                Debug.LogWarning($"[MonteCarloManager] Failed to export full field data for stage {currentRealization + 1}");
            }
        }
        
        // Fire event
        Debug.Log($"[MonteCarloManager] Firing OnRealizationCompleted event for stage {currentRealization + 1}");
        OnRealizationCompleted?.Invoke(currentRealization, result);
        
        Debug.Log($"[MonteCarloManager] Completed realization {currentRealization + 1}/{numRealizations}: " +
                  $"MeanH={result.MeanBedElevation:F4}, StdH={result.StdBedElevation:F4}");
        
        // Move to next realization
        currentRealization++;
        
        if (currentRealization < numRealizations)
        {
            Debug.Log($"[MonteCarloManager] Moving to next stage. Will start stage {currentRealization + 1} in 0.1 seconds...");
            // Small delay before next realization
            Invoke(nameof(StartNextRealization), 0.1f);
        }
        else
        {
            Debug.Log($"[MonteCarloManager] All stages complete! Completing ensemble...");
            CompleteEnsemble();
        }
    }
    
    /// <summary>
    /// Completes the entire ensemble simulation.
    /// </summary>
    private void CompleteEnsemble()
    {
        isRunning = false;
        
        // Compute ensemble statistics
        ensemble.ComputeEnsembleStatistics();
        
        Debug.Log($"[MonteCarloManager] ═══════════════════════════════════════════════════════════");
        Debug.Log($"[MonteCarloManager] MONTE CARLO ENSEMBLE COMPLETE");
        Debug.Log($"[MonteCarloManager] ═══════════════════════════════════════════════════════════");
        Debug.Log($"[MonteCarloManager] Realizations: {ensemble.CompletedRealizations}");
        Debug.Log($"[MonteCarloManager] Ensemble Mean Bed Elevation: {ensemble.EnsembleMeanBedElevation:F4}");
        Debug.Log($"[MonteCarloManager] Ensemble Std Bed Elevation: {ensemble.EnsembleStdBedElevation:F4}");
        Debug.Log($"[MonteCarloManager] 95% CI: [{ensemble.BedElevation95CI_Lower:F4}, {ensemble.BedElevation95CI_Upper:F4}]");
        Debug.Log($"[MonteCarloManager] ═══════════════════════════════════════════════════════════");
        
        // Export summary statistics if enabled
        if (autoExportResults && !string.IsNullOrEmpty(resultsFolderPath))
        {
            bool exportSuccess = MonteCarloCSVExporter.ExportSummaryStatistics(ensemble, resultsFolderPath);
            if (exportSuccess)
            {
                Debug.Log($"[MonteCarloManager] Summary statistics exported to: {resultsFolderPath}");
            }
            else
            {
                Debug.LogWarning("[MonteCarloManager] Failed to export summary statistics");
            }
        }
        
        // Fire event
        OnEnsembleCompleted?.Invoke(ensemble);
        
        // Release Monte Carlo override so solver reverts to base coefficient
        ResetMonteCarloTransportCoefficient();
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // SAMPLING METHODS
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// <summary>
    /// Samples from a log-normal distribution.
    /// If X ~ LogNormal(μ, σ), then ln(X) ~ Normal(μ, σ).
    /// </summary>
    private double SampleLogNormal(double mu, double sigma)
    {
        // Sample from standard normal using Box-Muller transform
        double u1 = masterRandom.NextDouble();
        double u2 = masterRandom.NextDouble();
        
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        
        // Transform to desired normal
        double normalSample = mu + sigma * z;
        
        // Transform to log-normal
        return Math.Exp(normalSample);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SUPPORTING CLASSES
// ═══════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Static class to pass Monte Carlo parameters to the solver.
/// This is a workaround for the readonly field limitation.
/// </summary>
public static class MonteCarloParameters
{
    public static bool UseMonteCarloTransportCoefficient = false;
    public static double CurrentTransportCoefficient = 0.1;
}

/// <summary>
/// Stores results from a single Monte Carlo realization.
/// </summary>
[System.Serializable]
    public class MonteCarloRealizationResult
{
    public int RealizationIndex { get; private set; }
    public double TransportCoefficient { get; private set; }
    public int PerlinSeed { get; private set; }
        public MonteCarloStateSnapshot Snapshot { get; private set; }
    
    // Summary statistics
    public double MeanBedElevation { get; private set; }
    public double StdBedElevation { get; private set; }
    public double MinBedElevation { get; private set; }
    public double MaxBedElevation { get; private set; }
    public double MeanVelocity { get; private set; }
    public double MaxErosionRate { get; private set; }
    
    // Optional: Full field storage (memory intensive)
    public double[,] FinalBedElevation { get; private set; }
    public double[,] FinalVelocityMagnitude { get; private set; }
    
    public MonteCarloRealizationResult(int index, double transportCoeff, int perlinSeed, 
                                       RiverMeshPhysicsSolver solver, bool storeFullFields = false)
    {
        RealizationIndex = index;
        TransportCoefficient = transportCoeff;
        PerlinSeed = perlinSeed;
        
        ComputeStatistics(solver);
        
        if (storeFullFields)
        {
            Snapshot = new MonteCarloStateSnapshot(solver);
            StoreFinalFields(solver);
        }
    }
    
    private void ComputeStatistics(RiverMeshPhysicsSolver solver)
    {
        int numCrossSections = solver.numCrossSections;
        int widthResolution = solver.widthResolution;
        
        double sumH = 0, sumH2 = 0;
        double sumV = 0;
        double maxErosion = 0;
        double minH = double.MaxValue, maxH = double.MinValue;
        int fluidCount = 0;
        
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                if (solver.cellType[i, w] == RiverCellType.FLUID)
                {
                    double h = solver.h[i, w];
                    sumH += h;
                    sumH2 += h * h;
                    minH = Math.Min(minH, h);
                    maxH = Math.Max(maxH, h);
                    
                    double vel = solver.GetVelocityMagnitude(i, w);
                    sumV += vel;
                    
                    double erosion = Math.Abs(solver.GetErosionRate(i, w));
                    maxErosion = Math.Max(maxErosion, erosion);
                    
                    fluidCount++;
                }
            }
        }
        
        if (fluidCount > 0)
        {
            MeanBedElevation = sumH / fluidCount;
            StdBedElevation = Math.Sqrt(sumH2 / fluidCount - MeanBedElevation * MeanBedElevation);
            MinBedElevation = minH;
            MaxBedElevation = maxH;
            MeanVelocity = sumV / fluidCount;
            MaxErosionRate = maxErosion;
        }
    }
    
    private void StoreFinalFields(RiverMeshPhysicsSolver solver)
    {
        int numCrossSections = solver.numCrossSections;
        int widthResolution = solver.widthResolution;
        
        FinalBedElevation = new double[numCrossSections, widthResolution];
        FinalVelocityMagnitude = new double[numCrossSections, widthResolution];
        
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                FinalBedElevation[i, w] = solver.h[i, w];
                FinalVelocityMagnitude[i, w] = solver.GetVelocityMagnitude(i, w);
            }
        }
    }
}

/// <summary>
/// Stores and analyzes the full Monte Carlo ensemble.
/// </summary>
[System.Serializable]
public class MonteCarloEnsemble
{
    private List<MonteCarloRealizationResult> realizations;
    private int maxRealizations;
    private bool storeFullData;
    
    // Ensemble statistics
    public int CompletedRealizations => realizations.Count;
    public double EnsembleMeanBedElevation { get; private set; }
    public double EnsembleStdBedElevation { get; private set; }
    public double BedElevation95CI_Lower { get; private set; }
    public double BedElevation95CI_Upper { get; private set; }
    
    // Transport coefficient statistics
    public double MeanTransportCoefficient { get; private set; }
    public double StdTransportCoefficient { get; private set; }
    public MonteCarloRealizationResult LatestRealization => realizations.Count > 0 ? realizations[realizations.Count - 1] : null;
    
    public MonteCarloEnsemble(int maxRealizations, bool storeFullData = false)
    {
        this.maxRealizations = maxRealizations;
        this.storeFullData = storeFullData;
        realizations = new List<MonteCarloRealizationResult>(maxRealizations);
    }
    
    public void AddRealization(MonteCarloRealizationResult result)
    {
        realizations.Add(result);
    }
    
    public MonteCarloRealizationResult GetRealization(int index)
    {
        if (index >= 0 && index < realizations.Count)
        {
            return realizations[index];
        }
        return null;
    }
    
    /// <summary>
    /// Computes ensemble statistics from all realizations.
    /// </summary>
    public void ComputeEnsembleStatistics()
    {
        if (realizations.Count == 0)
        {
            return;
        }
        
        // Compute mean and std of bed elevation across ensemble
        double sumMeanH = 0, sumMeanH2 = 0;
        double sumK = 0, sumK2 = 0;
        List<double> allMeanH = new List<double>();
        
        foreach (var r in realizations)
        {
            sumMeanH += r.MeanBedElevation;
            sumMeanH2 += r.MeanBedElevation * r.MeanBedElevation;
            allMeanH.Add(r.MeanBedElevation);
            
            sumK += r.TransportCoefficient;
            sumK2 += r.TransportCoefficient * r.TransportCoefficient;
        }
        
        int n = realizations.Count;
        EnsembleMeanBedElevation = sumMeanH / n;
        EnsembleStdBedElevation = Math.Sqrt(sumMeanH2 / n - EnsembleMeanBedElevation * EnsembleMeanBedElevation);
        
        MeanTransportCoefficient = sumK / n;
        StdTransportCoefficient = Math.Sqrt(sumK2 / n - MeanTransportCoefficient * MeanTransportCoefficient);
        
        // Compute 95% confidence interval using percentiles
        allMeanH.Sort();
        int lowerIdx = (int)(0.025 * n);
        int upperIdx = (int)(0.975 * n);
        
        BedElevation95CI_Lower = allMeanH[Math.Max(0, lowerIdx)];
        BedElevation95CI_Upper = allMeanH[Math.Min(n - 1, upperIdx)];
    }
    
    /// <summary>
    /// Gets all realization results as a list.
    /// </summary>
    public List<MonteCarloRealizationResult> GetAllRealizations()
    {
        return new List<MonteCarloRealizationResult>(realizations);
    }
}

/// <summary>
/// 3D Perlin noise generator with seeded initialization.
/// </summary>
public class PerlinNoise3D
{
    private int[] permutation;
    
    public PerlinNoise3D(int seed)
    {
        System.Random random = new System.Random(seed);
        
        // Initialize permutation table
        permutation = new int[512];
        int[] p = new int[256];
        for (int i = 0; i < 256; i++)
        {
            p[i] = i;
        }
        
        // Shuffle
        for (int i = 255; i > 0; i--)
        {
            int j = random.Next(i + 1);
            int temp = p[i];
            p[i] = p[j];
            p[j] = temp;
        }
        
        // Duplicate for wraparound
        for (int i = 0; i < 512; i++)
        {
            permutation[i] = p[i & 255];
        }
    }
    
    /// <summary>
    /// Computes 3D Perlin noise at the given coordinates.
    /// </summary>
    public float Noise(float x, float y, float z)
    {
        // Find unit cube containing point
        int X = (int)Math.Floor(x) & 255;
        int Y = (int)Math.Floor(y) & 255;
        int Z = (int)Math.Floor(z) & 255;
        
        // Find relative position in cube
        x -= (float)Math.Floor(x);
        y -= (float)Math.Floor(y);
        z -= (float)Math.Floor(z);
        
        // Compute fade curves
        float u = Fade(x);
        float v = Fade(y);
        float w = Fade(z);
        
        // Hash coordinates of cube corners
        int A = permutation[X] + Y;
        int AA = permutation[A] + Z;
        int AB = permutation[A + 1] + Z;
        int B = permutation[X + 1] + Y;
        int BA = permutation[B] + Z;
        int BB = permutation[B + 1] + Z;
        
        // Blend results from corners
        return Lerp(w, Lerp(v, Lerp(u, Grad(permutation[AA], x, y, z),
                                       Grad(permutation[BA], x - 1, y, z)),
                              Lerp(u, Grad(permutation[AB], x, y - 1, z),
                                       Grad(permutation[BB], x - 1, y - 1, z))),
                       Lerp(v, Lerp(u, Grad(permutation[AA + 1], x, y, z - 1),
                                       Grad(permutation[BA + 1], x - 1, y, z - 1)),
                              Lerp(u, Grad(permutation[AB + 1], x, y - 1, z - 1),
                                       Grad(permutation[BB + 1], x - 1, y - 1, z - 1))));
    }
    
    /// <summary>
    /// Computes fractal (octave) Perlin noise.
    /// </summary>
    public float FractalNoise(float x, float y, float z, int octaves, float persistence)
    {
        float total = 0;
        float frequency = 1;
        float amplitude = 1;
        float maxValue = 0;
        
        for (int i = 0; i < octaves; i++)
        {
            total += Noise(x * frequency, y * frequency, z * frequency) * amplitude;
            maxValue += amplitude;
            amplitude *= persistence;
            frequency *= 2;
        }
        
        return total / maxValue;  // Normalize to [-1, 1]
    }
    
    private float Fade(float t)
    {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }
    
    private float Lerp(float t, float a, float b)
    {
        return a + t * (b - a);
    }
    
    private float Grad(int hash, float x, float y, float z)
    {
        int h = hash & 15;
        float u = h < 8 ? x : y;
        float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }
}

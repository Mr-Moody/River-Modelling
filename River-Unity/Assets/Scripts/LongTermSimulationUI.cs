using UnityEngine;
using UnityEngine.UI;
using TMPro;

/// <summary>
/// UI Interface for Long-Term Simulation Controller.
/// Provides controls for starting/stopping the simulation, selecting time step, and displaying progress.
/// </summary>
public class LongTermSimulationUI : MonoBehaviour
{
    [Header("References")]
    [Tooltip("Reference to the LongTermSimulationController component (drag the GameObject with LongTermSimulationController from Hierarchy)")]
    public LongTermSimulationController simulationController;
    
    [Header("Control Buttons")]
    public Button startButton;
    public Button stopButton;
    public Button renderToggleButton;
    
    [Header("Time Step Selection")]
    public Dropdown timeStepDropdown;
    
    [Header("Progress Display")]
    public Slider progressBar;
    public TextMeshProUGUI progressText;
    
    [Header("Cursor Control")]
    [Tooltip("Key to press to toggle cursor lock/unlock (default: C)")]
    public KeyCode cursorToggleKey = KeyCode.C;
    
    [Header("Monte Carlo Control")]
    [Tooltip("GameObject containing Monte Carlo components (will be disabled when long-term simulation runs)")]
    public GameObject monteCarloGameObject;
    
    private bool isCursorLocked = true;
    private bool isRenderingEnabled = true;
    
    private void Start()
    {
        // Find LongTermSimulationController if not assigned
        if (simulationController == null)
        {
            simulationController = FindFirstObjectByType<LongTermSimulationController>();
            if (simulationController == null)
            {
                Debug.LogError("[LongTermSimulationUI] LongTermSimulationController not found! Please assign it in the Inspector.");
                return;
            }
        }
        
        // Auto-find UI elements if not assigned
        AutoFindUIElements();
        
        // Initialize UI
        InitializeUI();
        
        // Subscribe to events
        if (simulationController != null)
        {
            simulationController.OnSimulationStarted += OnSimulationStarted;
            simulationController.OnSimulationCompleted += OnSimulationCompleted;
            simulationController.OnProgressUpdated += OnProgressUpdated;
        }
        
        // Initialize cursor state (unlocked by default for UI interaction)
        UnlockCursor();
        Debug.Log("[LongTermSimulationUI] Cursor unlocked by default for UI interaction. Press C to lock/unlock.");
        
        // Auto-find Monte Carlo GameObject if not assigned
        if (monteCarloGameObject == null)
        {
            MonteCarloSimulationManager mcManager = FindFirstObjectByType<MonteCarloSimulationManager>();
            if (mcManager != null)
            {
                monteCarloGameObject = mcManager.gameObject;
                Debug.Log($"[LongTermSimulationUI] Auto-found Monte Carlo GameObject: {monteCarloGameObject.name}");
            }
        }
        
        // Verify EventSystem exists
        if (UnityEngine.EventSystems.EventSystem.current == null)
        {
            Debug.LogWarning("[LongTermSimulationUI] No EventSystem found! Creating one...");
            GameObject eventSystem = new GameObject("EventSystem");
            eventSystem.AddComponent<UnityEngine.EventSystems.EventSystem>();
            eventSystem.AddComponent<UnityEngine.EventSystems.StandaloneInputModule>();
        }
    }
    
    private void OnDestroy()
    {
        // Unsubscribe from events
        if (simulationController != null)
        {
            simulationController.OnSimulationStarted -= OnSimulationStarted;
            simulationController.OnSimulationCompleted -= OnSimulationCompleted;
            simulationController.OnProgressUpdated -= OnProgressUpdated;
        }
    }
    
    private void Update()
    {
        // Toggle cursor lock/unlock with C key
        if (Input.GetKeyDown(cursorToggleKey))
        {
            ToggleCursor();
        }
        
        // Update progress display continuously
        if (simulationController != null && simulationController.IsRunning)
        {
            var (currentStep, totalSteps, elapsedYears, progressPercent) = simulationController.GetProgress();
            UpdateProgressDisplay(currentStep, totalSteps, elapsedYears, progressPercent);
        }
    }
    
    /// <summary>
    /// Auto-finds UI elements if not assigned.
    /// </summary>
    private void AutoFindUIElements()
    {
        Transform root = transform;
        
        // Recursively find buttons (they might be nested)
        if (startButton == null)
        {
            startButton = FindChildByName(root, "StartButton")?.GetComponent<Button>();
        }
        
        if (stopButton == null)
        {
            stopButton = FindChildByName(root, "StopButton")?.GetComponent<Button>();
        }
        
        if (renderToggleButton == null)
        {
            renderToggleButton = FindChildByName(root, "RenderToggleButton")?.GetComponent<Button>();
        }
        
        if (timeStepDropdown == null)
        {
            timeStepDropdown = FindChildByName(root, "TimeStepDropdown")?.GetComponent<Dropdown>();
        }
        
        if (progressBar == null)
        {
            progressBar = FindChildByName(root, "ProgressBar")?.GetComponent<Slider>();
        }
        
        if (progressText == null)
        {
            progressText = FindChildByName(root, "ProgressText")?.GetComponent<TextMeshProUGUI>();
        }
    }
    
    /// <summary>
    /// Recursively finds a child transform by name.
    /// </summary>
    private Transform FindChildByName(Transform parent, string name)
    {
        // Check direct children first
        for (int i = 0; i < parent.childCount; i++)
        {
            Transform child = parent.GetChild(i);
            if (child.name == name)
            {
                return child;
            }
            
            // Recursively search grandchildren
            Transform found = FindChildByName(child, name);
            if (found != null)
            {
                return found;
            }
        }
        return null;
    }
    
    /// <summary>
    /// Initializes the UI elements.
    /// </summary>
    private void InitializeUI()
    {
        Debug.Log("[LongTermSimulationUI] Initializing UI...");
        
        // Setup start button
        if (startButton != null)
        {
            startButton.onClick.RemoveAllListeners();
            startButton.onClick.AddListener(OnStartClicked);
            Debug.Log("[LongTermSimulationUI] Start button initialized and listener added");
        }
        else
        {
            Debug.LogError("[LongTermSimulationUI] Start button not found! Cannot initialize.");
        }
        
        // Setup stop button
        if (stopButton != null)
        {
            stopButton.onClick.RemoveAllListeners();
            stopButton.onClick.AddListener(OnStopClicked);
            stopButton.interactable = false;
        }
        else
        {
            Debug.LogWarning("[LongTermSimulationUI] Stop button not found!");
        }
        
        // Setup render toggle button
        if (renderToggleButton != null)
        {
            renderToggleButton.onClick.RemoveAllListeners();
            renderToggleButton.onClick.AddListener(OnRenderToggleClicked);
            UpdateRenderButtonText();
        }
        else
        {
            Debug.LogWarning("[LongTermSimulationUI] Render toggle button not found!");
        }
        
        // Setup time step dropdown
        if (timeStepDropdown != null)
        {
            timeStepDropdown.ClearOptions();
            for (int i = 1; i <= 7; i++)
            {
                timeStepDropdown.options.Add(new Dropdown.OptionData($"{i} day{(i > 1 ? "s" : "")}"));
            }
            // Initialize dropdown to match controller's current value (from editor)
            if (simulationController != null)
            {
                // Clamp value to valid range (1-7) and convert to 0-indexed
                int editorValue = Mathf.Clamp(simulationController.timeStepDays, 1, 7);
                timeStepDropdown.value = editorValue - 1; // Convert to 0-indexed
                Debug.Log($"[LongTermSimulationUI] Initialized time step dropdown to match editor value: {editorValue} day(s)");
            }
            else
            {
                timeStepDropdown.value = 0; // Default to 1 day if no controller
            }
            timeStepDropdown.onValueChanged.RemoveAllListeners();
            timeStepDropdown.onValueChanged.AddListener(OnTimeStepChanged);
        }
        else
        {
            Debug.LogWarning("[LongTermSimulationUI] Time step dropdown not found!");
        }
        
        // Setup progress bar
        if (progressBar != null)
        {
            progressBar.minValue = 0f;
            progressBar.maxValue = 100f;
            progressBar.value = 0f;
        }
        
        // Initialize progress text
        UpdateProgressDisplay(0, 0, 0f, 0f);
    }
    
    /// <summary>
    /// Handles start button click.
    /// </summary>
    private void OnStartClicked()
    {
        Debug.Log("========================================");
        Debug.Log("[LongTermSimulationUI] *** START BUTTON CLICKED ***");
        Debug.Log("========================================");
        
        if (simulationController == null)
        {
            Debug.LogError("[LongTermSimulationUI] Cannot start - SimulationController is null!");
            Debug.LogError("[LongTermSimulationUI] Please assign the LongTermSimulationController reference in the Inspector.");
            return;
        }
        
        Debug.Log($"[LongTermSimulationUI] SimulationController found: {simulationController.name}");
        
        // Use the current timeStepDays value from the controller (set in editor or via dropdown)
        // Only update from dropdown if it was explicitly changed by the user
        if (timeStepDropdown != null)
        {
            int dropdownValue = timeStepDropdown.value + 1; // 0-indexed to 1-7
            // Only update if dropdown value differs from controller value (user changed it)
            if (dropdownValue != simulationController.timeStepDays)
            {
                simulationController.timeStepDays = dropdownValue;
                Debug.Log($"[LongTermSimulationUI] Time step updated from dropdown to: {simulationController.timeStepDays} day(s)");
            }
            else
            {
                Debug.Log($"[LongTermSimulationUI] Using time step from editor: {simulationController.timeStepDays} day(s)");
            }
        }
        else
        {
            Debug.Log($"[LongTermSimulationUI] Time step dropdown is null, using editor value: {simulationController.timeStepDays} day(s)");
        }
        
        // Start simulation
        Debug.Log("[LongTermSimulationUI] Calling StartLongTermSimulation()...");
        simulationController.StartLongTermSimulation();
        Debug.Log("[LongTermSimulationUI] StartLongTermSimulation() called");
    }
    
    /// <summary>
    /// Handles stop button click.
    /// </summary>
    private void OnStopClicked()
    {
        Debug.Log("[LongTermSimulationUI] *** STOP BUTTON CLICKED ***");
        
        if (simulationController == null)
        {
            Debug.LogError("[LongTermSimulationUI] Cannot stop - SimulationController is null!");
            return;
        }
        
        Debug.Log("[LongTermSimulationUI] Calling StopLongTermSimulation()...");
        simulationController.StopLongTermSimulation();
        Debug.Log("[LongTermSimulationUI] StopLongTermSimulation() called");
    }
    
    // Test method to verify button is working - can be called from Inspector
    [ContextMenu("Test Start Button")]
    public void TestStartButton()
    {
        Debug.Log("[LongTermSimulationUI] TestStartButton() called directly");
        OnStartClicked();
    }
    
    /// <summary>
    /// Handles time step dropdown change.
    /// </summary>
    private void OnTimeStepChanged(int value)
    {
        if (simulationController != null)
        {
            simulationController.timeStepDays = value + 1; // 0-indexed to 1-7
            Debug.Log($"[LongTermSimulationUI] Time step changed to {simulationController.timeStepDays} day(s)");
        }
    }
    
    /// <summary>
    /// Handles render toggle button click.
    /// </summary>
    private void OnRenderToggleClicked()
    {
        isRenderingEnabled = !isRenderingEnabled;
        
        // Find all cameras in the scene
        Camera[] cameras = FindObjectsByType<Camera>(FindObjectsSortMode.None);
        
        int enabledCount = 0;
        int disabledCount = 0;
        
        foreach (Camera cam in cameras)
        {
            cam.enabled = isRenderingEnabled;
            if (isRenderingEnabled)
                enabledCount++;
            else
                disabledCount++;
        }
        
        UpdateRenderButtonText();
        Debug.Log($"[LongTermSimulationUI] Rendering {(isRenderingEnabled ? "ENABLED" : "DISABLED")} - {enabledCount} cameras {(isRenderingEnabled ? "enabled" : "disabled")}");
    }
    
    /// <summary>
    /// Updates the render toggle button text.
    /// </summary>
    private void UpdateRenderButtonText()
    {
        if (renderToggleButton != null)
        {
            TextMeshProUGUI buttonText = renderToggleButton.GetComponentInChildren<TextMeshProUGUI>();
            if (buttonText != null)
            {
                buttonText.text = isRenderingEnabled ? "Disable Rendering" : "Enable Rendering";
            }
        }
    }
    
    /// <summary>
    /// Handles simulation started event.
    /// </summary>
    private void OnSimulationStarted()
    {
        Debug.Log("[LongTermSimulationUI] Simulation started");
        UpdateUIState(true);
        
        // Disable Monte Carlo GameObject if assigned
        if (monteCarloGameObject != null)
        {
            monteCarloGameObject.SetActive(false);
            Debug.Log($"[LongTermSimulationUI] Disabled Monte Carlo GameObject: {monteCarloGameObject.name}");
        }
    }
    
    /// <summary>
    /// Handles simulation completed event.
    /// </summary>
    private void OnSimulationCompleted()
    {
        Debug.Log("[LongTermSimulationUI] Simulation completed");
        UpdateUIState(false);
        
        // Re-enable Monte Carlo GameObject if assigned
        if (monteCarloGameObject != null)
        {
            monteCarloGameObject.SetActive(true);
            Debug.Log($"[LongTermSimulationUI] Re-enabled Monte Carlo GameObject: {monteCarloGameObject.name}");
        }
    }
    
    /// <summary>
    /// Handles progress updated event.
    /// </summary>
    private void OnProgressUpdated(int currentStep, float elapsedYears, float progressPercent)
    {
        if (simulationController != null)
        {
            var (_, totalSteps, _, _) = simulationController.GetProgress();
            UpdateProgressDisplay(currentStep, totalSteps, elapsedYears, progressPercent);
        }
    }
    
    /// <summary>
    /// Updates the progress display.
    /// </summary>
    private void UpdateProgressDisplay(int currentStep, int totalSteps, float elapsedYears, float progressPercent)
    {
        // Update progress bar
        if (progressBar != null)
        {
            progressBar.value = progressPercent;
        }
        
        // Update progress text
        if (progressText != null)
        {
            if (totalSteps > 0)
            {
                progressText.text = $"{elapsedYears:F2} years / {simulationController.simulationYears} years ({progressPercent:F1}%)";
            }
            else
            {
                progressText.text = "Ready to start";
            }
        }
    }
    
    /// <summary>
    /// Updates UI state based on simulation status.
    /// </summary>
    private void UpdateUIState(bool isRunning)
    {
        if (startButton != null)
        {
            startButton.interactable = !isRunning;
        }
        
        if (stopButton != null)
        {
            stopButton.interactable = isRunning;
        }
        
        if (timeStepDropdown != null)
        {
            timeStepDropdown.interactable = !isRunning;
        }
    }
    
    /// <summary>
    /// Toggles cursor lock/unlock state.
    /// </summary>
    private void ToggleCursor()
    {
        if (isCursorLocked)
        {
            UnlockCursor();
        }
        else
        {
            LockCursor();
        }
    }
    
    /// <summary>
    /// Locks and hides the cursor.
    /// </summary>
    private void LockCursor()
    {
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;
        isCursorLocked = true;
        Debug.Log("[LongTermSimulationUI] Cursor locked and hidden");
    }
    
    /// <summary>
    /// Unlocks and shows the cursor.
    /// </summary>
    private void UnlockCursor()
    {
        Cursor.lockState = CursorLockMode.None;
        Cursor.visible = true;
        isCursorLocked = false;
        Debug.Log("[LongTermSimulationUI] Cursor unlocked and visible");
    }
}

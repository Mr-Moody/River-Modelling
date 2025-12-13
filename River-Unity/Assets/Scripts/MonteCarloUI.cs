using UnityEngine;
using UnityEngine.UI;
using TMPro;

/// <summary>
/// Simplified UI Interface for Monte Carlo Simulation Manager.
/// Provides only essential controls: Start, Stop, Progress Bar, and Stage display.
/// All configuration parameters are edited in the Unity Inspector on the MonteCarloSimulationManager component.
/// </summary>
public class MonteCarloUI : MonoBehaviour
{
    [Header("References")]
    [Tooltip("Reference to the MonteCarloSimulationManager component")]
    public MonteCarloSimulationManager monteCarloManager;
    
    [Header("Control Buttons")]
    public Button startButton;
    public Button stopButton;
    public Button renderToggleButton;
    
    [Header("Progress Display")]
    public Slider progressBar;
    public TextMeshProUGUI stageText;
    
    [Header("Cursor Control")]
    [Tooltip("Key to press to toggle cursor lock/unlock (default: C)")]
    public KeyCode cursorToggleKey = KeyCode.C;
    
    private bool isCursorLocked = true;
    private bool isRenderingEnabled = true;
    
    void Start()
    {
        // Find MonteCarloSimulationManager if not assigned
        if (monteCarloManager == null)
        {
            monteCarloManager = FindFirstObjectByType<MonteCarloSimulationManager>();
            if (monteCarloManager == null)
            {
                Debug.LogError("[MonteCarloUI] MonteCarloSimulationManager not found! Please assign it in the Inspector.");
                return;
            }
        }
        
        // Auto-find UI elements if not assigned
        AutoFindUIElements();
        
        // Subscribe to events
        monteCarloManager.OnRealizationStarted += OnRealizationStarted;
        monteCarloManager.OnEnsembleCompleted += OnEnsembleCompleted;
        monteCarloManager.OnProgressUpdated += OnProgressUpdated;
        
        // Initialize UI
        InitializeUI();
        
        // Initialize cursor state (unlocked by default for UI interaction)
        UnlockCursor();
        Debug.Log("[MonteCarloUI] Cursor unlocked by default for UI interaction. Press C to lock/unlock.");
        
        // Initial state
        UpdateUIState(false);
        
        // Verify EventSystem exists (required for UI interaction)
        if (UnityEngine.EventSystems.EventSystem.current == null)
        {
            Debug.LogWarning("[MonteCarloUI] No EventSystem found! UI buttons may not work. Creating one...");
            GameObject eventSystem = new GameObject("EventSystem");
            eventSystem.AddComponent<UnityEngine.EventSystems.EventSystem>();
            eventSystem.AddComponent<UnityEngine.EventSystems.StandaloneInputModule>();
        }
    }
    
    void Update()
    {
        // Toggle cursor lock/unlock with C key
        if (Input.GetKeyDown(cursorToggleKey))
        {
            ToggleCursor();
        }
        
        // Debug: Check if button is being clicked (for troubleshooting)
        if (startButton != null && Input.GetMouseButtonDown(0))
        {
            // Check if mouse is over the button
            UnityEngine.EventSystems.PointerEventData pointerData = new UnityEngine.EventSystems.PointerEventData(UnityEngine.EventSystems.EventSystem.current);
            pointerData.position = Input.mousePosition;
            
            var results = new System.Collections.Generic.List<UnityEngine.EventSystems.RaycastResult>();
            UnityEngine.EventSystems.EventSystem.current.RaycastAll(pointerData, results);
            
            foreach (var result in results)
            {
                if (result.gameObject == startButton.gameObject || result.gameObject.transform.IsChildOf(startButton.transform))
                {
                    Debug.Log("[MonteCarloUI] Mouse click detected over Start button!");
                }
            }
        }
    }
    
    /// <summary>
    /// Automatically finds UI elements by name if they're not assigned.
    /// </summary>
    private void AutoFindUIElements()
    {
        Transform root = transform;
        
        // Find buttons
        if (startButton == null)
            startButton = FindChildByName(root, "StartButton")?.GetComponent<Button>();
        if (stopButton == null)
            stopButton = FindChildByName(root, "StopButton")?.GetComponent<Button>();
        if (renderToggleButton == null)
            renderToggleButton = FindChildByName(root, "RenderToggleButton")?.GetComponent<Button>();
        
        // Find progress elements
        if (progressBar == null)
            progressBar = FindChildByName(root, "ProgressBar")?.GetComponent<Slider>();
        if (stageText == null)
            stageText = FindChildByName(root, "StageText")?.GetComponent<TextMeshProUGUI>();
    }
    
    /// <summary>
    /// Recursively finds a child transform by name.
    /// </summary>
    private Transform FindChildByName(Transform parent, string name)
    {
        if (parent == null) return null;
        
        // Check direct children first
        foreach (Transform child in parent)
        {
            if (child.name == name || child.name.Contains(name))
                return child;
        }
        
        // Recursively search grandchildren
        foreach (Transform child in parent)
        {
            Transform found = FindChildByName(child, name);
            if (found != null) return found;
        }
        
        return null;
    }
    
    void OnDestroy()
    {
        // Unsubscribe from events
        if (monteCarloManager != null)
        {
            monteCarloManager.OnRealizationStarted -= OnRealizationStarted;
            monteCarloManager.OnEnsembleCompleted -= OnEnsembleCompleted;
            monteCarloManager.OnProgressUpdated -= OnProgressUpdated;
        }
    }
    
    private void InitializeUI()
    {
        Debug.Log("[MonteCarloUI] Initializing UI...");
        
        // Set initial button states
        if (startButton != null)
        {
            // Remove any existing listeners first
            startButton.onClick.RemoveAllListeners();
            startButton.onClick.AddListener(OnStartClicked);
            
            Debug.Log($"[MonteCarloUI] Start button found: {startButton.name}");
            Debug.Log($"[MonteCarloUI] Start button interactable: {startButton.interactable}");
            Debug.Log($"[MonteCarloUI] Start button enabled: {startButton.enabled}");
            Debug.Log($"[MonteCarloUI] Start button gameObject active: {startButton.gameObject.activeInHierarchy}");
            Debug.Log("[MonteCarloUI] Start button listener added successfully");
        }
        else
        {
            Debug.LogError("[MonteCarloUI] Start button is null! Button click will not work.");
        }
        
        if (stopButton != null)
        {
            stopButton.onClick.AddListener(OnStopClicked);
            stopButton.interactable = false;
            Debug.Log("[MonteCarloUI] Stop button found and listener added");
        }
        else
        {
            Debug.LogWarning("[MonteCarloUI] Stop button is null!");
        }
        
        if (renderToggleButton != null)
        {
            renderToggleButton.onClick.AddListener(OnRenderToggleClicked);
            UpdateRenderButtonText();
            Debug.Log("[MonteCarloUI] Render toggle button found and listener added");
        }
        else
        {
            Debug.LogWarning("[MonteCarloUI] Render toggle button is null!");
        }
        
        // Initialize progress bar
        if (progressBar != null)
        {
            progressBar.minValue = 0f;
            progressBar.maxValue = 100f;
            progressBar.value = 0f;
            Debug.Log("[MonteCarloUI] Progress bar initialized");
        }
        else
        {
            Debug.LogWarning("[MonteCarloUI] Progress bar is null!");
        }
        
        // Initialize stage text
        if (stageText != null)
        {
            stageText.text = "Stage: 0 / 0";
            Debug.Log("[MonteCarloUI] Stage text initialized");
        }
        else
        {
            Debug.LogWarning("[MonteCarloUI] Stage text is null!");
        }
        
        // Check manager reference
        if (monteCarloManager == null)
        {
            Debug.LogError("[MonteCarloUI] MonteCarloSimulationManager is null! Cannot start simulation.");
        }
        else
        {
            Debug.Log($"[MonteCarloUI] MonteCarloSimulationManager found: {monteCarloManager.name}");
        }
    }
    
    private void OnStartClicked()
    {
        Debug.Log("========================================");
        Debug.Log("[MonteCarloUI] *** START BUTTON CLICKED ***");
        Debug.Log("========================================");
        
        // Start simulation
        if (monteCarloManager != null)
        {
            Debug.Log($"[MonteCarloUI] Manager found: {monteCarloManager.name}");
            Debug.Log("[MonteCarloUI] Calling StartEnsembleSimulation()...");
            monteCarloManager.StartEnsembleSimulation();
            UpdateUIState(true);
            Debug.Log("[MonteCarloUI] StartEnsembleSimulation() called, UI state updated");
        }
        else
        {
            Debug.LogError("[MonteCarloUI] Cannot start - MonteCarloSimulationManager is null!");
        }
    }
    
    // Test method to verify button is working - can be called from Inspector
    [ContextMenu("Test Start Button")]
    public void TestStartButton()
    {
        Debug.Log("[MonteCarloUI] TestStartButton() called directly");
        OnStartClicked();
    }
    
    private void OnStopClicked()
    {
        Debug.Log("[MonteCarloUI] Stop button clicked!");
        
        if (monteCarloManager != null)
        {
            Debug.Log("[MonteCarloUI] Calling StopEnsembleSimulation()...");
            monteCarloManager.StopEnsembleSimulation();
            UpdateUIState(false);
            Debug.Log("[MonteCarloUI] StopEnsembleSimulation() called, UI state updated");
        }
        else
        {
            Debug.LogError("[MonteCarloUI] Cannot stop - MonteCarloSimulationManager is null!");
        }
    }
    
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
        
        Debug.Log($"[MonteCarloUI] Rendering {(isRenderingEnabled ? "ENABLED" : "DISABLED")} - {enabledCount} cameras {(isRenderingEnabled ? "enabled" : "disabled")}");
    }
    
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
    
    private void UpdateUIState(bool isRunning)
    {
        if (startButton != null)
            startButton.interactable = !isRunning;
        
        if (stopButton != null)
            stopButton.interactable = isRunning;
    }
    
    // Event handlers
    private void OnRealizationStarted(int realizationIndex, int total)
    {
        Debug.Log($"[MonteCarloUI] New stage started: {realizationIndex + 1} / {total}");
        
        if (stageText != null)
            stageText.text = $"Stage: {realizationIndex + 1} / {total}";
        else
            Debug.LogWarning("[MonteCarloUI] Stage text is null, cannot update display");
    }
    
    private void OnProgressUpdated(int realization, int step, float progress)
    {
        if (progressBar != null)
            progressBar.value = progress;
        
        // Update stage text with current realization
        if (stageText != null && monteCarloManager != null)
        {
            int total = monteCarloManager.numRealizations;
            stageText.text = $"Stage: {realization + 1} / {total}";
        }
    }
    
    private void OnEnsembleCompleted(MonteCarloEnsemble ensemble)
    {
        UpdateUIState(false);
        
        if (progressBar != null)
            progressBar.value = 100f;
        
        if (stageText != null)
            stageText.text = $"Stage: {ensemble.CompletedRealizations} / {ensemble.CompletedRealizations} (Complete)";
    }
    
    /// <summary>
    /// Toggles cursor lock state and visibility.
    /// When locked: cursor is hidden and locked to center (for camera control).
    /// When unlocked: cursor is visible and free (for UI interaction).
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
    /// Locks and hides the cursor for camera control.
    /// </summary>
    private void LockCursor()
    {
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;
        isCursorLocked = true;
    }
    
    /// <summary>
    /// Unlocks and shows the cursor for UI interaction.
    /// </summary>
    private void UnlockCursor()
    {
        Cursor.lockState = CursorLockMode.None;
        Cursor.visible = true;
        isCursorLocked = false;
    }
}

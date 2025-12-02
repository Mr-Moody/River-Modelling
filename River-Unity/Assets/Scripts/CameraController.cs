using UnityEngine;

public class CameraController : MonoBehaviour
{
    [Header("Movement Settings")]
    [Tooltip("Movement speed in units per second")]
    public float moveSpeed = 5.0f;
    
    [Tooltip("Vertical movement speed (for Space/Shift)")]
    public float verticalMoveSpeed = 5.0f;
    
    [Header("Mouse Look Settings")]
    [Tooltip("Mouse sensitivity for looking around")]
    public float mouseSensitivity = 2.0f;
    
    [Tooltip("Lock cursor to center of screen when looking around")]
    public bool lockCursor = true;
    
    private float rotationX = 0f;
    private float rotationY = 0f;
    
    private SimulationController simulationController;
    private bool simulationToggleKeyPressed = false;
    
    void Start()
    {
        // Lock cursor to center for mouse look
        if (lockCursor)
        {
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }
        
        // Find SimulationController in the scene
        simulationController = FindObjectOfType<SimulationController>();
        if (simulationController == null)
        {
            Debug.LogWarning("CameraController: SimulationController not found in scene. F key toggle will not work.");
        }
    }

    void Update()
    {
        HandleMouseLook();
        HandleMovement();
        HandleSimulationToggle();
    }
    
    void HandleMouseLook()
    {
        // Get mouse input
        float mouseX = Input.GetAxis("Mouse X") * mouseSensitivity;
        float mouseY = Input.GetAxis("Mouse Y") * mouseSensitivity;
        
        // Rotate camera based on mouse movement
        rotationY += mouseX;
        rotationX -= mouseY;
        
        // Clamp vertical rotation to prevent flipping
        rotationX = Mathf.Clamp(rotationX, -90f, 90f);
        
        // Apply rotation
        transform.localRotation = Quaternion.Euler(rotationX, rotationY, 0f);
        
        // Unlock cursor with Escape key
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            if (Cursor.lockState == CursorLockMode.Locked)
            {
                Cursor.lockState = CursorLockMode.None;
                Cursor.visible = true;
            }
            else
            {
                Cursor.lockState = CursorLockMode.Locked;
                Cursor.visible = false;
            }
        }
    }
    
    void HandleMovement()
    {
        // Get movement input
        float horizontal = Input.GetAxis("Horizontal"); // A/D keys
        float vertical = Input.GetAxis("Vertical");     // W/S keys
        
        // Calculate movement direction relative to camera's local rotation
        Vector3 forward = transform.forward;
        Vector3 right = transform.right;
        Vector3 up = transform.up;
        
        // Calculate movement in camera's local space (all relative to facing direction)
        Vector3 movement = (forward * vertical + right * horizontal) * moveSpeed;
        
        // Vertical movement relative to camera's up direction (Space = camera up, Shift = camera down)
        if (Input.GetKey(KeyCode.Space))
        {
            movement += up * verticalMoveSpeed;
        }
        else if (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift))
        {
            movement += -up * verticalMoveSpeed;
        }
        
        // Apply movement relative to camera position
        transform.position += movement * Time.deltaTime;
    }
    
    void HandleSimulationToggle()
    {
        // Toggle simulation with F key
        if (Input.GetKeyDown(KeyCode.F))
        {
            if (!simulationToggleKeyPressed)
            {
                simulationToggleKeyPressed = true;
                
                if (simulationController != null)
                {
                    simulationController.RunSimulation = !simulationController.RunSimulation;
                    Debug.Log($"Simulation toggled: {(simulationController.RunSimulation ? "STARTED" : "STOPPED")}");
                }
                else
                {
                    Debug.LogWarning("CameraController: Cannot toggle simulation - SimulationController not found!");
                }
            }
        }
        else if (Input.GetKeyUp(KeyCode.F))
        {
            simulationToggleKeyPressed = false;
        }
    }
}

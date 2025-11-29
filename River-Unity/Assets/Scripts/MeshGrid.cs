using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System; // Required for Math.Min/Max and Array handling

// [ExecuteAlways] - Removed to prevent editor mode issues
public class MeshGrid : MonoBehaviour
{
    [Header("References")]
    [Tooltip("Reference to RiverToGrid component on the same GameObject.")]
    public RiverToGrid riverToGrid;

    private Mesh mesh;
    private Vector3[] vertices;
    private Color[] colors;
    private int[] triangles;
    private int[] triIndices;

    // --- NEW Internal Offsets to align mesh with RiverToGrid bounds ---
    private float minXOffset = 0f;
    private float minZOffset = 0f;

    // --- Events ---
    /// <summary>
    /// Event fired when mesh is updated with new simulation data.
    /// Parameters: Updated Mesh object
    /// </summary>
    public static event Action<Mesh> OnMeshUpdated;

    // These are the physical dimensions of the grid in Unity units
    [Range(1f, 100f)]
    public float physicalHeight = 50f;
    [Range(1f, 100f)]
    public float physicalWidth = 100f;

    // These are the resolution parameters (number of cells)
    [Range(1, 200)]
    public int numCellsX = 20;
    [Range(1, 200)]
    public int numCellsY = 20;

    // Internal resolution (vertex count)
    private int vertsX => numCellsX + 1;
    private int vertsZ => numCellsY + 1;

    void OnEnable()
    {
        // Early return if not in play mode to avoid editor issues
        if (!Application.isPlaying)
        {
            return;
        }

        Debug.Log("[MeshGrid] OnEnable() called - Subscribing to events");
        
        // Subscribe to grid initialization event
        RiverToGrid.OnGridInitialized += HandleGridInitialized;
        
        Debug.Log("[MeshGrid] ✓ Event subscription complete");
    }

    void OnDisable()
    {
        // Unsubscribe from events
        RiverToGrid.OnGridInitialized -= HandleGridInitialized;
    }

    void Awake()
    {
        // Early return if not in play mode
        if (!Application.isPlaying)
        {
            return;
        }

        Debug.Log("[MeshGrid] Awake() called");
        
        // Try to get RiverToGrid reference if not assigned
        if (riverToGrid == null)
        {
            riverToGrid = GetComponent<RiverToGrid>();
            Debug.Log($"[MeshGrid] RiverToGrid reference: {(riverToGrid != null ? "found" : "not found")}");
        }
        
        // Only generate default mesh if not initialized via events
        // Events will handle initialization when grid is ready
        if (riverToGrid == null || !riverToGrid.isInitialized)
        {
            Debug.Log("[MeshGrid] Generating default mesh in Awake()...");
            generateMesh();
            Debug.Log("[MeshGrid] ✓ Default mesh generation complete");
        }
        else
        {
            Debug.Log("[MeshGrid] Skipping default mesh - will be initialized via events");
        }
    }

    /// <summary>
    /// Handles grid initialization event from RiverToGrid.
    /// </summary>
    private void HandleGridInitialized(int gridWidth, int gridHeight, float minX, float minZ, float maxX, float maxZ)
    {
        Debug.Log($"[MeshGrid] HandleGridInitialized called - Grid: {gridWidth}x{gridHeight}, Bounds: X[{minX:F2}, {maxX:F2}], Z[{minZ:F2}, {maxZ:F2}]");
        
        if (riverToGrid == null)
        {
            Debug.LogError("MeshGrid: RiverToGrid reference is missing!");
            return;
        }

        float cellSize = riverToGrid.GetCellSize();
        Debug.Log($"[MeshGrid] Starting mesh initialization with cellSize: {cellSize}");
        Initialize(gridWidth, gridHeight, cellSize, minX, minZ);
        Debug.Log("[MeshGrid] ✓ Mesh initialization complete");
    }

    private bool isGeneratingMesh = false; // Guard to prevent re-entry

    void OnValidate()
    {
        // Skip entirely if already generating or not in play mode
        if (isGeneratingMesh || !Application.isPlaying)
        {
            return;
        }

        // Ensure values are at least 1 (defensive for script changes)
        if (physicalHeight < 1f) physicalHeight = 1f;
        if (physicalWidth < 1f) physicalWidth = 1f;
        if (numCellsX < 1) numCellsX = 1;
        if (numCellsY < 1) numCellsY = 1;

        // Recreate mesh when values change in the inspector
        // We do NOT use the min/max offsets here, as OnValidate should reflect
        // the Inspector settings primarily.
        minXOffset = 0f;
        minZOffset = 0f;
        
        // Defer mesh generation to avoid SendMessage errors during OnValidate
        // Setting sharedMesh during OnValidate causes SendMessage errors
        // Use a coroutine to defer mesh generation until after OnValidate completes
        StartCoroutine(DeferredMeshGeneration());
    }

    private System.Collections.IEnumerator DeferredMeshGeneration()
    {
        yield return null; // Wait one frame to exit OnValidate
        if (!isGeneratingMesh)
        {
            generateMesh();
        }
    }

    // --- Public Initialization Method (FIXED SIGNATURE) ---

    /// <summary>
    /// Initializes the MeshGrid based on the Simulation's parameters, using 
    /// minX and minZ to correctly offset the mesh to match the physical river geometry bounds.
    /// </summary>
    /// <param name="gridWidth">Number of cells in the X direction.</param>
    /// <param name="gridHeight">Number of cells in the Z direction.</param>
    /// <param name="cellSize">Physical size of each cell.</param>
    /// <param name="minX">The minimum X extent (offset) of the river geometry.</param>
    /// <param name="minZ">The minimum Z extent (offset) of the river geometry.</param>
    public void Initialize(int gridWidth, int gridHeight, float cellSize, float minX, float minZ)
    {
        Debug.Log($"[MeshGrid] Initialize() called - Grid: {gridWidth}x{gridHeight}, CellSize: {cellSize}");
        
        // The simulation's grid dimensions (cells)
        numCellsX = gridWidth;
        numCellsY = gridHeight;

        // The simulation's physical size
        physicalWidth = gridWidth * cellSize;
        physicalHeight = gridHeight * cellSize;

        // --- NEW: Store the offsets ---
        this.minXOffset = minX;
        this.minZOffset = minZ;

        Debug.Log($"[MeshGrid] Generating mesh: {numCellsX}x{numCellsY} cells, Physical size: {physicalWidth:F2}x{physicalHeight:F2}");
        generateMesh();
        Debug.Log("[MeshGrid] ✓ Mesh generation complete");
    }

    // --- Public Update Method ---

    /// <summary>
    /// Returns the current mesh, or null if not yet initialized.
    /// </summary>
    public Mesh GetMesh()
    {
        return mesh;
    }

    /// <summary>
    /// Updates the mesh vertices' Y-position (elevation) and colors based on simulation data.
    /// </summary>
    public void UpdateMesh(double[,] h, double[,] waterDepth, double[,] u, double[,] v, int[,] cellType)
    {
        if (mesh == null || vertices == null || h == null)
        {
            Debug.LogError("Mesh or data not initialized.");
            return;
        }

        // Check if mesh dimensions match simulation data dimensions
        // The simulation data can be either cell-centered (numCellsX x numCellsY) or vertex-centered (vertsX x vertsZ)
        // PhysicsSolver uses nx = gridWidth + 1, so data arrays are (gridWidth+1) x (gridHeight+1) = vertsX x vertsZ
        
        int dataWidth = h.GetLength(0);
        int dataHeight = h.GetLength(1);
        
        // Check if dimensions match vertex count (most likely case)
        bool matchesVertices = (dataWidth == vertsX && dataHeight == vertsZ);
        // Check if dimensions match cell count (less likely but possible)
        bool matchesCells = (dataWidth == numCellsX && dataHeight == numCellsY);
        
        if (!matchesVertices && !matchesCells)
        {
            // Mesh might not be initialized yet - this can happen during initialization
            // Only log error if mesh is already properly sized (not the default 2x2)
            if (numCellsX > 10 || numCellsY > 10)
            {
                Debug.LogWarning($"Simulation data dimensions ({dataWidth}x{dataHeight}) do not match mesh dimensions (cells: {numCellsX}x{numCellsY}, vertices: {vertsX}x{vertsZ}). Mesh may not be initialized yet.");
            }
            return;
        }

        // Get the latest vertices (copying is slow, but safe if mesh is not re-generated)
        // If mesh.vertices is not read-only, mesh.vertices.CopyTo(vertices, 0) is necessary.
        mesh.vertices.CopyTo(vertices, 0);

        if (colors == null || colors.Length != vertices.Length)
        {
            colors = new Color[vertices.Length];
        }

        // Determine if data is vertex-centered or cell-centered
        bool isVertexCentered = (dataWidth == vertsX && dataHeight == vertsZ);
        
        // The physics grid (h[x, z]) corresponds directly to the vertex array (x, z)
        for (int z = 0; z < vertsZ; z++)
        {
            for (int x = 0; x < vertsX; x++)
            {
                int i = z * vertsX + x;

                // 1. Update Bed Elevation (Y-coordinate)
                if (isVertexCentered)
                {
                    // Data is vertex-centered, use directly
                    vertices[i].y = (float)h[x, z];
                }
                else
                {
                    // Data is cell-centered, clamp to valid range
                    int hX = Mathf.Min(x, numCellsX - 1);
                    int hZ = Mathf.Min(z, numCellsY - 1);
                    vertices[i].y = (float)h[hX, hZ];
                }

                // 2. Color based on Flow Data (Using indices x and z for simplicity)

                // Clamp indices for boundary cells if the data arrays (u, v, waterDepth, cellType)
                // are sized numCellsX x numCellsY (cell-centered data) and not vertsX x vertsZ (vertex-centered data).
                // Assuming data arrays are cell-centered, we must clamp or handle boundary vertices.
                int dataX = Mathf.Min(x, numCellsX - 1);
                int dataZ = Mathf.Min(z, numCellsY - 1);

                float u_val = (float)u[dataX, dataZ];
                float v_val = (float)v[dataX, dataZ];

                float u_mag = (float)Math.Sqrt(u_val * u_val + v_val * v_val);
                float max_flow_speed = 1.0f;

                Color newColor = Color.green;

                if (cellType[dataX, dataZ] == RiverCellType.FLUID)
                {
                    float depth = (float)waterDepth[dataX, dataZ];
                    float flow_intensity = Mathf.Clamp(u_mag / max_flow_speed, 0f, 1f);
                    // Blue for water, brighter for deeper/faster water
                    newColor = Color.Lerp(Color.cyan * 0.5f, Color.blue, flow_intensity);
                    // Add depth visualization to color's alpha or brightness if needed
                }
                else if (cellType[dataX, dataZ] == RiverCellType.BANK)
                {
                    // Brown for banks
                    newColor = Color.Lerp(new Color(0.6f, 0.4f, 0.2f), Color.gray, 0.5f);
                }
                else // TERRAIN
                {
                    // Light green/brown for static terrain
                    newColor = Color.Lerp(Color.yellow, Color.green, 0.7f);
                }

                colors[i] = newColor;
            }
        }

        // Assign the updated data back to the mesh
        mesh.vertices = vertices;
        mesh.colors = colors;

        // Recalculate normals are critical for correct lighting on the new surface
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        // Fire event to notify that mesh has been updated
        OnMeshUpdated?.Invoke(mesh);
    }

    // --- Private Mesh Generation Logic (FIXED VERTEX CALCULATION) ---

    private void generateMesh()
    {
        // Guard against re-entry
        if (isGeneratingMesh)
        {
            Debug.LogWarning("[MeshGrid] generateMesh() already in progress, skipping");
            return;
        }

        isGeneratingMesh = true;

        try
        {
            Debug.Log($"[MeshGrid] generateMesh() called - Grid: {numCellsX}x{numCellsY}");
            
            // Clamp resolution (extra safety)
            if (numCellsX < 1) numCellsX = 1;
            if (numCellsY < 1) numCellsY = 1;

        Debug.Log("[MeshGrid] Step 1: Getting/creating MeshFilter on child GameObject...");
        
        // 1. Create or get a child GameObject for the simulation grid mesh
        // This prevents conflicts with the river geometry mesh on the parent
        GameObject gridMeshObject = null;
        Transform gridTransform = transform.Find("SimulationGridMesh");
        if (gridTransform != null)
        {
            gridMeshObject = gridTransform.gameObject;
        }
        else
        {
            gridMeshObject = new GameObject("SimulationGridMesh");
            gridMeshObject.transform.SetParent(transform);
            gridMeshObject.transform.localPosition = Vector3.zero;
            gridMeshObject.transform.localRotation = Quaternion.identity;
            gridMeshObject.transform.localScale = Vector3.one;
            Debug.Log("[MeshGrid] Created child GameObject for simulation grid mesh");
        }
        
        var mf = gridMeshObject.GetComponent<MeshFilter>();
        if (mf == null) 
        {
            mf = gridMeshObject.AddComponent<MeshFilter>();
            Debug.Log("[MeshGrid] Created MeshFilter component on child GameObject");
        }

        Debug.Log("[MeshGrid] Step 2: Getting/creating MeshRenderer on child GameObject...");
        
        // 2. Ensure MeshRenderer and a Material exist (required for visibility)
        var mr = gridMeshObject.GetComponent<MeshRenderer>();
        if (mr == null)
        {
            Debug.Log("[MeshGrid] Creating MeshRenderer and material on child GameObject...");
            mr = gridMeshObject.AddComponent<MeshRenderer>();
            
            // Assign a default material to make the mesh visible and support vertex colors
            // Use a try-catch in case Shader.Find blocks
            try
            {
                Shader standardShader = Shader.Find("Standard");
                if (standardShader != null)
                {
                    mr.sharedMaterial = new Material(standardShader);
                    Debug.Log("[MeshGrid] ✓ Material created successfully");
                }
                else
                {
                    Debug.LogWarning("[MeshGrid] Standard shader not found, using default material");
                    mr.sharedMaterial = new Material(Shader.Find("Diffuse")); // Fallback
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[MeshGrid] Error creating material: {e.Message}");
            }
        }

        Debug.Log($"[MeshGrid] Step 3: Initializing mesh - Vertices needed: {vertsX * vertsZ} ({vertsX}x{vertsZ})");
        
        // 3. Initialize or clear the mesh
        if (mesh == null)
        {
            mesh = new Mesh();
            mesh.name = "Procedural Grid";
            Debug.Log("[MeshGrid] Created new mesh");
        }
        else
        {
            mesh.Clear();
            Debug.Log("[MeshGrid] Cleared existing mesh");
        }

        Debug.Log("[MeshGrid] Step 4: Generating vertices and UVs...");
        
        // 4. Generate Vertices and UVs for the XZ (Horizontal) Plane
        vertices = new Vector3[vertsX * vertsZ];
        Vector2[] uvs = new Vector2[vertices.Length];

        for (int z = 0; z < vertsZ; z++) // z-loop corresponds to Z-axis positions
        {
            for (int x = 0; x < vertsX; x++) // x-loop corresponds to X-axis positions
            {
                int i = z * vertsX + x;
                float u_norm = (float)x / numCellsX;
                float v_norm = (float)z / numCellsY;

                // Calculate position relative to the local origin (0,0)
                float vx_relative = u_norm * physicalWidth;
                float vz_relative = v_norm * physicalHeight;

                // FIX: Apply the calculated offsets (minX/minZ) from RiverToGrid
                // This ensures the grid starts at the correct physical coordinate.
                // Position grid slightly below river geometry (Y=-0.01) to avoid z-fighting
                vertices[i] = new Vector3(minXOffset + vx_relative, -0.01f, minZOffset + vz_relative);

                uvs[i] = new Vector2(u_norm, v_norm);
            }
        }

        Debug.Log($"[MeshGrid] Step 5: Generating triangles - {numCellsX * numCellsY} quads = {numCellsX * numCellsY * 6} triangles");
        
        // 5. Generate Triangles (Indices) (Unchanged and correct)
        triangles = new int[numCellsX * numCellsY * 6];
        int t = 0;

        for (int z = 0; z < numCellsY; z++)
        {
            for (int x = 0; x < numCellsX; x++)
            {
                int i = z * vertsX + x; // Bottom-Left index of the quad

                // Triangle 1 (Bottom-Left, Top-Left, Bottom-Right) - Standard CW/CCW depending on view
                triangles[t++] = i;
                triangles[t++] = i + vertsX;
                triangles[t++] = i + 1;

                // Triangle 2 (Top-Right, Bottom-Right, Top-Left) - Completing the quad
                triangles[t++] = i + vertsX + 1;
                triangles[t++] = i + 1;
                triangles[t++] = i + vertsX;
            }
        }

        Debug.Log("[MeshGrid] Step 6: Assigning mesh data...");
        
        // 6. Assign Mesh Data and finalize
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.uv = uvs;

        Debug.Log("[MeshGrid] Recalculating normals and bounds...");
        
        // Optimize: Only recalculate if mesh has triangles
        if (mesh.triangles != null && mesh.triangles.Length > 0)
        {
            mesh.RecalculateNormals();
        }
        mesh.RecalculateBounds();

        Debug.Log("[MeshGrid] Step 7: Assigning mesh to MeshFilter...");
        
        // Assign the generated mesh to the MeshFilter
        // Only set sharedMesh if we're in play mode (to avoid SendMessage errors during OnValidate)
        // The deferred coroutine ensures we're not in OnValidate when this runs
        if (Application.isPlaying)
        {
            mf.sharedMesh = mesh;
            Debug.Log("[MeshGrid] ✓ Mesh assigned to MeshFilter");
            
            // Fire event when mesh is first generated
            OnMeshUpdated?.Invoke(mesh);
            Debug.Log("[MeshGrid] ✓ OnMeshUpdated event fired");
        }
        else
        {
            Debug.Log("[MeshGrid] Not in play mode, skipping mesh assignment (will be assigned later)");
        }
        
            Debug.Log("[MeshGrid] ✓ generateMesh() complete");
        }
        finally
        {
            isGeneratingMesh = false;
        }
    }
}
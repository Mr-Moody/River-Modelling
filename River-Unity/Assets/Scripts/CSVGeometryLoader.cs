using UnityEngine;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Globalization;

public class CSVGeometryLoader : MonoBehaviour
{
    [Header("Visualization")]
    [Tooltip("Material to apply to the river geometry mesh. If not assigned, will attempt to load Materials/RiverMaterial from Resources.")]
    public Material riverMaterial;
    // --- Events ---
    /// <summary>
    /// Event fired when mesh is successfully loaded from CSV.
    /// Parameters: Mesh object containing the loaded geometry
    /// </summary>
    public static event Action<Mesh> OnMeshLoaded;
    // --- Configuration ---
    [Tooltip("The CSV file asset (must be placed in a Resources folder for runtime loading).")]
    public TextAsset csvFile;

    [Tooltip("Scaling factor to apply to normalized coordinates.")]
    public float scaleFactor = 1.0f;

    [Tooltip("WARNING: Your CSV lacks an explicit elevation column. This value sets the vertical position (Unity Y) of the riverbed.")]
    public float defaultElevationY = 0.0f;

    [Tooltip("Simplify mesh by skipping every Nth cross-section. Higher values = fewer vertices but less detail. 1 = no simplification.")]
    [Range(1, 100)]
    public int simplificationFactor = 1;

    [Header("Grid Remeshing")]
    [Tooltip("Enable grid remeshing to convert unstructured river geometry into a structured grid mesh for simulation updates.")]
    public bool useGridMesh = false;
    
    [Tooltip("Number of points across the river width (width resolution). Higher values = more detail but more vertices.")]
    [Range(2, 50)]
    public int widthResolution = 10;

    // --- CSV Column Headers (MUST MATCH YOUR FILE) ---
    private const string ORDER_COL = "order";
    private const string LEFT_X_COL = "left_bank_x";
    private const string LEFT_Y_COL = "left_bank_y";
    private const string RIGHT_X_COL = "right_bank_x";
    private const string RIGHT_Y_COL = "right_bank_y";

    // --- Internal Structures ---
    private struct CrossSectionPoint
    {
        public int Order;
        public float LX, LY, RX, RY;
    }

    // The Mesh is now purely a container for the vertices, used to pass data to RiverToGrid.
    private Mesh mesh;

    /// <summary>
    /// Public getter for the generated mesh, allowing it to be assigned to RiverToGrid in the editor.
    /// </summary>
    public Mesh GetMesh()
    {
        return mesh;
    }
    
    /// <summary>
    /// Positions the river mesh child GameObject to align with the grid coordinate system.
    /// Should be called after RiverToGrid is initialized.
    /// </summary>
    public void AlignMeshWithGrid(float gridMinX, float gridMinZ)
    {
        GameObject meshObject = transform.Find("RiverMesh")?.gameObject;
        if (meshObject != null)
        {
            // The river mesh vertices are normalized to start at (0, 0)
            // The grid uses coordinates starting at (gridMinX, gridMinZ) which may be negative due to padding
            // Position the mesh GameObject to align with the grid
            meshObject.transform.localPosition = new Vector3(gridMinX, 0, gridMinZ);
            Debug.Log($"[CSVGeometryLoader] River mesh aligned with grid at position ({gridMinX:F2}, 0, {gridMinZ:F2})");
        }
    }

    void Start()
    {
        if (csvFile == null)
        {
            Debug.LogError("CSV File asset is not assigned.");
            return;
        }

        GenerateMeshFromCsv();

        if (mesh == null)
        {
            Debug.LogError("[CSVGeometryLoader] Mesh generation failed - mesh is null!");
        }
    }

    // --- Public Getter for Geometry Data ---

    /// <summary>
    /// Exposes the generated mesh vertices (normalized and scaled) for use by the RiverToGrid component.
    /// </summary>
    public Vector3[] GetNormalizedVertices()
    {
        // We use the internal Mesh object's vertices as the source of truth for the geometry.
        return mesh != null ? mesh.vertices : null;
    }

    // --- Core Data Generation ---

    /// <summary>
    /// Core function to read data, normalize, and create the raw mesh structure (without rendering).
    /// </summary>
    private void GenerateMeshFromCsv()
    {
        var culture = CultureInfo.InvariantCulture;

        List<CrossSectionPoint> rawPoints = new List<CrossSectionPoint>();

        // 1. Read and Parse Data
        using (StringReader reader = new StringReader(csvFile.text))
        {
            string headerLine = reader.ReadLine();
            if (string.IsNullOrEmpty(headerLine)) return;

            string[] headers = headerLine.Trim().Split(',');

            // Find column indices (same logic as before)
            int orderIndex = Array.IndexOf(headers, ORDER_COL);
            int lxIndex = Array.IndexOf(headers, LEFT_X_COL);
            int lyIndex = Array.IndexOf(headers, LEFT_Y_COL);
            int rxIndex = Array.IndexOf(headers, RIGHT_X_COL);
            int ryIndex = Array.IndexOf(headers, RIGHT_Y_COL);

            if (lxIndex == -1 || lyIndex == -1 || rxIndex == -1 || ryIndex == -1 || orderIndex == -1)
            {
                Debug.LogError("One or more required column headers not found.");
                return;
            }

            string line;
            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;

                string[] values = line.Split(',');

                if (values.Length > Math.Max(lxIndex, ryIndex))
                {
                    try
                    {
                        float orderFloat = float.Parse(values[orderIndex].Trim(), culture);
                        int order = (int)orderFloat;

                        float lx = float.Parse(values[lxIndex].Trim(), culture);
                        float ly = float.Parse(values[lyIndex].Trim(), culture);
                        float rx = float.Parse(values[rxIndex].Trim(), culture);
                        float ry = float.Parse(values[ryIndex].Trim(), culture);

                        rawPoints.Add(new CrossSectionPoint
                        {
                            Order = order,
                            LX = lx,
                            LY = ly,
                            RX = rx,
                            RY = ry
                        });
                    }
                    catch (FormatException e)
                    {
                        Debug.LogError($"Error parsing numeric value in line: {line}. Error: {e.Message}");
                    }
                }
            }
        }

        if (rawPoints.Count < 2)
        {
            Debug.LogError("Not enough cross-sections loaded to form a river strip (need at least 2). Loaded: " + rawPoints.Count);
            return;
        }

        rawPoints = rawPoints.OrderBy(p => p.Order).ToList();

        // Remove duplicate cross-sections at start/end that might create a closing face
        // Check if first and last cross-sections are duplicates (same position)
        if (rawPoints.Count >= 2)
        {
            var first = rawPoints[0];
            var last = rawPoints[rawPoints.Count - 1];
            
            // Check if first and last are at the same position (within small tolerance)
            float tolerance = 0.0001f;
            bool firstEqualsLast = Mathf.Abs(first.LX - last.LX) < tolerance &&
                                   Mathf.Abs(first.LY - last.LY) < tolerance &&
                                   Mathf.Abs(first.RX - last.RX) < tolerance &&
                                   Mathf.Abs(first.RY - last.RY) < tolerance;
            
            if (firstEqualsLast)
            {
                Debug.LogWarning("[CSVGeometryLoader] First and last cross-sections are duplicates. Removing last to prevent closing face.");
                rawPoints.RemoveAt(rawPoints.Count - 1);
            }
        }

        // Apply simplification if requested
        if (simplificationFactor > 1)
        {
            int originalCount = rawPoints.Count;
            rawPoints = rawPoints.Where((p, index) => index % simplificationFactor == 0).ToList();
        }

        // 2. Calculate bounds BEFORE normalization to find the minimum
        float minX = rawPoints.Min(p => Mathf.Min(p.LX, p.RX));
        float maxX = rawPoints.Max(p => Mathf.Max(p.LX, p.RX));
        float minZ = rawPoints.Min(p => Mathf.Min(p.LY, p.RY));
        float maxZ = rawPoints.Max(p => Mathf.Max(p.LY, p.RY));
        
        // Use the minimum as the offset to ensure mesh starts at origin
        float offsetX = minX;
        float offsetZ = minZ;

        // 3. Prepare Final Vertex List
        List<Vector3> vertices = new List<Vector3>();
        for (int i = 0; i < rawPoints.Count; i++)
        {
            CrossSectionPoint p = rawPoints[i];

            // Left Bank Vertex (Unity X, Y, Z) - Normalize to start at origin
            float localLX = (p.LX - offsetX) * scaleFactor;
            float localLZ = (p.LY - offsetZ) * scaleFactor;
            vertices.Add(new Vector3(localLX, defaultElevationY, localLZ));

            // Right Bank Vertex (Unity X, Y, Z) - Normalize to start at origin
            float localRX = (p.RX - offsetX) * scaleFactor;
            float localRZ = (p.RY - offsetZ) * scaleFactor;
            vertices.Add(new Vector3(localRX, defaultElevationY, localRZ));
        }

        // 4. Generate Mesh (Data container only)
        mesh = new Mesh();
        mesh.name = "RiverGeometryData";

        mesh.vertices = vertices.ToArray();

        // 5. Strip Triangulation
        int[] triangles = StripTriangulate(rawPoints.Count);
        mesh.triangles = triangles;

        // 6. Generate UVs for proper texturing
        Vector2[] uvs = new Vector2[vertices.Count];
        for (int i = 0; i < vertices.Count; i += 2)
        {
            float u = (float)(i / 2) / (rawPoints.Count - 1); // U coordinate along the river
            uvs[i] = new Vector2(u, 0f);     // Left bank
            uvs[i + 1] = new Vector2(u, 1f); // Right bank
        }
        mesh.uv = uvs;

        // Finalize Mesh Data (Normals/Bounds are still helpful for downstream code like RiverToGrid)
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        
        // Remesh into grid structure if enabled
        if (useGridMesh)
        {
            RemeshToGrid();
        }
        
        // Setup MeshFilter and MeshRenderer to display the river geometry
        SetupMeshRenderer();

        // Fire event to notify other components that mesh is ready
        OnMeshLoaded?.Invoke(mesh);
        
        Debug.Log($"[CSVGeometryLoader] Mesh loaded: {mesh.vertexCount} vertices, Bounds: {mesh.bounds.size.x:F2}x{mesh.bounds.size.z:F2}");
    }

    /// <summary>
    /// Sets up the MeshFilter and MeshRenderer components and applies the RiverMaterial.
    /// Creates a child GameObject for the mesh so it can be moved independently.
    /// </summary>
    private void SetupMeshRenderer()
    {
        if (mesh == null)
        {
            Debug.LogError("[CSVGeometryLoader] Cannot setup renderer - mesh is null!");
            return;
        }
        
        // Create or find a child GameObject for the mesh
        GameObject meshObject = transform.Find("RiverMesh")?.gameObject;
        if (meshObject == null)
        {
            meshObject = new GameObject("RiverMesh");
            meshObject.transform.SetParent(transform);
            meshObject.transform.localPosition = Vector3.zero;
            meshObject.transform.localRotation = Quaternion.identity;
            meshObject.transform.localScale = Vector3.one;
        }
        
        // Get or add MeshFilter component on the child GameObject
        MeshFilter meshFilter = meshObject.GetComponent<MeshFilter>();
        if (meshFilter == null)
        {
            meshFilter = meshObject.AddComponent<MeshFilter>();
        }
        
        meshFilter.mesh = mesh;

        // Get or add MeshRenderer component on the child GameObject
        MeshRenderer meshRenderer = meshObject.GetComponent<MeshRenderer>();
        if (meshRenderer == null)
        {
            meshRenderer = meshObject.AddComponent<MeshRenderer>();
        }

        // Apply RiverMaterial (use assigned material or try to load from Resources)
        Material materialToUse = riverMaterial;
        
        if (materialToUse == null)
        {
            // Try loading from Resources folder
            materialToUse = Resources.Load<Material>("Materials/RiverMaterial");
            if (materialToUse == null)
            {
                // Try alternative path
                materialToUse = Resources.Load<Material>("RiverMaterial");
            }
        }

        if (materialToUse != null)
        {
            // Check if the material uses the velocity heatmap shader, if not try to assign it
            Shader velocityHeatmapShader = Shader.Find("Custom/RiverVelocityHeatmap");
            if (velocityHeatmapShader != null && materialToUse.shader != velocityHeatmapShader)
            {
                // Try to switch to velocity heatmap shader if available
                // This allows the material to display velocity-based heatmap
                materialToUse.shader = velocityHeatmapShader;
                materialToUse.SetFloat("_MaxVelocity", 1.0f); // Default max velocity
                materialToUse.SetFloat("_MinVelocity", 0.0f);
                Debug.Log("[CSVGeometryLoader] RiverMaterial found and switched to RiverVelocityHeatmap shader for velocity visualization.");
            }
            meshRenderer.material = materialToUse;
        }
        else
        {
            // Try to create material with velocity heatmap shader first
            Shader velocityHeatmapShader = Shader.Find("Custom/RiverVelocityHeatmap");
            if (velocityHeatmapShader != null)
            {
                Material defaultMat = new Material(velocityHeatmapShader);
                defaultMat.SetColor("_Color", Color.white);
                defaultMat.SetFloat("_MaxVelocity", 1.0f);
                defaultMat.SetFloat("_MinVelocity", 0.0f);
                defaultMat.SetFloat("_Metallic", 0.0f);
                defaultMat.SetFloat("_Glossiness", 0.5f);
                meshRenderer.material = defaultMat;
                Debug.Log("[CSVGeometryLoader] RiverMaterial not found. Created material with RiverVelocityHeatmap shader for velocity visualization.");
            }
            else
            {
                // Fallback to Standard shader if velocity heatmap shader not found
                Shader standardShader = Shader.Find("Standard");
                if (standardShader != null)
                {
                    Material defaultMat = new Material(standardShader);
                    defaultMat.color = new Color(0.8f, 0.2f, 0.2f, 1f); // Red color to distinguish from grid
                    defaultMat.SetFloat("_Metallic", 0.3f);
                    defaultMat.SetFloat("_Glossiness", 0.5f);
                    meshRenderer.material = defaultMat;
                    Debug.LogWarning($"[CSVGeometryLoader] RiverMaterial not found and RiverVelocityHeatmap shader not available. Using Standard shader. Please assign Materials/RiverMaterial in the Inspector.");
                }
                else
                {
                    Debug.LogError("[CSVGeometryLoader] Could not find Standard shader or create default material.");
                }
            }
        }
        
        // Make sure the renderer is enabled
        meshRenderer.enabled = true;
        
        // Enable wireframe or add outline to make the river shape more visible
        // Set render queue to ensure it renders on top
        if (meshRenderer.material != null)
        {
            meshRenderer.material.renderQueue = 3000; // Render after opaque geometry
        }
    }
    

    /// <summary>
    /// Remeshes the unstructured river geometry into a structured grid mesh.
    /// The grid follows the river's path along its length and has configurable width resolution.
    /// This allows the simulation to update the physical geometry more easily.
    /// 
    /// Grid structure:
    /// - Length dimension: follows the river's cross-sections (one row per cross-section)
    /// - Width dimension: interpolated points across the river width (configurable via widthResolution)
    /// - Vertices are arranged in rows: each row is a cross-section with widthResolution points
    /// - Vertex index calculation: vertexIndex = rowIndex * widthResolution + columnIndex
    /// </summary>
    public void RemeshToGrid()
    {
        if (mesh == null || mesh.vertices == null || mesh.vertices.Length < 4)
        {
            Debug.LogError("[CSVGeometryLoader] Cannot remesh - mesh is null or invalid!");
            return;
        }

        Vector3[] originalVertices = mesh.vertices;
        int numCrossSections = originalVertices.Length / 2; // Each cross-section has 2 vertices (left and right bank)
        
        if (numCrossSections < 2)
        {
            Debug.LogError("[CSVGeometryLoader] Cannot remesh - need at least 2 cross-sections!");
            return;
        }

        // Extract cross-sections from original vertices
        List<Vector3> leftBank = new List<Vector3>();
        List<Vector3> rightBank = new List<Vector3>();
        
        for (int i = 0; i < originalVertices.Length; i += 2)
        {
            leftBank.Add(originalVertices[i]);      // Left bank vertex
            rightBank.Add(originalVertices[i + 1]); // Right bank vertex
        }

        // Create grid vertices: numCrossSections along length, widthResolution across width
        int lengthResolution = numCrossSections;
        List<Vector3> gridVertices = new List<Vector3>();
        List<Vector2> gridUVs = new List<Vector2>();

        for (int i = 0; i < lengthResolution; i++)
        {
            Vector3 left = leftBank[i];
            Vector3 right = rightBank[i];
            
            // Interpolate across the width
            for (int w = 0; w < widthResolution; w++)
            {
                Vector3 vertex;
                
                // Ensure first and last points are exactly at bank positions for perfect alignment
                if (w == 0)
                {
                    vertex = left; // Exact left bank position
                }
                else if (w == widthResolution - 1)
                {
                    vertex = right; // Exact right bank position
                }
                else
                {
                    float t = (float)w / (widthResolution - 1); // 0 to 1 across width
                    vertex = Vector3.Lerp(left, right, t);
                }
                
                gridVertices.Add(vertex);
                
                // UV coordinates: U along length, V across width
                float u = lengthResolution > 1 ? (float)i / (lengthResolution - 1) : 0f;
                float v = widthResolution > 1 ? (float)w / (widthResolution - 1) : 0f;
                gridUVs.Add(new Vector2(u, v));
            }
        }

        // Generate triangles for grid mesh
        // Note: Only connect adjacent cross-sections, don't connect last to first
        List<int> triangles = new List<int>();
        
        for (int i = 0; i < lengthResolution - 1; i++)
        {
            for (int w = 0; w < widthResolution - 1; w++)
            {
                int current = i * widthResolution + w;
                int next = (i + 1) * widthResolution + w;
                int currentRight = i * widthResolution + (w + 1);
                int nextRight = (i + 1) * widthResolution + (w + 1);

                // Triangle 1: Reversed winding order to face upwards
                triangles.Add(current);
                triangles.Add(currentRight);
                triangles.Add(next);

                // Triangle 2: Reversed winding order to face upwards
                triangles.Add(next);
                triangles.Add(currentRight);
                triangles.Add(nextRight);
            }
        }

        // Create new grid mesh
        mesh.Clear();
        mesh.vertices = gridVertices.ToArray();
        mesh.triangles = triangles.ToArray();
        mesh.uv = gridUVs.ToArray();
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        Debug.Log($"[CSVGeometryLoader] Remeshed to grid: {gridVertices.Count} vertices ({lengthResolution}x{widthResolution} grid), {triangles.Count / 3} triangles");
    }

    /// <summary>
    /// Creates a mesh strip by connecting sequential cross-sections (L_i, R_i) to (L_{i+1}, R_{i+1}).
    /// Corrected winding order to ensure normals face upwards.
    /// IMPORTANT: This does NOT create a closing face - it only connects consecutive cross-sections.
    /// The mesh is an open strip from start to end.
    /// </summary>
    private int[] StripTriangulate(int numCrossSections)
    {
        // Only create quads between consecutive cross-sections (not connecting last back to first)
        int numQuads = numCrossSections - 1;
        int[] triangles = new int[numQuads * 6];
        int t = 0;

        // Loop only connects i to i+1, stopping before the last cross-section
        // This ensures no closing face is created between start and end
        for (int i = 0; i < numQuads; i++)
        {
            int v0 = i * 2;       // L_i (Current Left Bank)
            int v1 = i * 2 + 1;   // R_i (Current Right Bank)
            int v2 = (i + 1) * 2;   // L_{i+1} (Next Left Bank)
            int v3 = (i + 1) * 2 + 1; // R_{i+1} (Next Right Bank)

            // Triangle 1: L_i, R_i, L_{i+1} (v0, v1, v2)
            triangles[t++] = v0;
            triangles[t++] = v1;
            triangles[t++] = v2;

            // Triangle 2: R_i, R_{i+1}, L_{i+1} (v1, v3, v2)
            triangles[t++] = v1;
            triangles[t++] = v3;
            triangles[t++] = v2;
        }

        return triangles;
    }
}
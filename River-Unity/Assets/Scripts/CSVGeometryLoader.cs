using UnityEngine;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Globalization;

[RequireComponent(typeof(MeshFilter))]
[RequireComponent(typeof(MeshRenderer))]
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

    void Start()
    {
        Debug.Log("[CSVGeometryLoader] Start() called - Beginning CSV loading...");
        
        if (csvFile == null)
        {
            Debug.LogError("CSV File asset is not assigned.");
            return;
        }

        Debug.Log($"[CSVGeometryLoader] CSV file found: {csvFile.name}, starting mesh generation...");
        GenerateMeshFromCsv();

        // Debug log confirming data load, NOT visualization success.
        if (mesh != null)
        {
            Debug.Log($"[CSVGeometryLoader] ✓ Mesh generation complete: {mesh.vertexCount} vertices generated. Firing OnMeshLoaded event...");
        }
        else
        {
            Debug.LogError("[CSVGeometryLoader] ✗ Mesh generation failed - mesh is null!");
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

        // Apply simplification if requested
        if (simplificationFactor > 1)
        {
            int originalCount = rawPoints.Count;
            rawPoints = rawPoints.Where((p, index) => index % simplificationFactor == 0).ToList();
            Debug.Log($"[CSVGeometryLoader] Mesh simplified: {originalCount} -> {rawPoints.Count} cross-sections (factor: {simplificationFactor})");
        }

        // 2. Calculate bounds BEFORE normalization to find the minimum
        float minX = rawPoints.Min(p => Mathf.Min(p.LX, p.RX));
        float maxX = rawPoints.Max(p => Mathf.Max(p.LX, p.RX));
        float minZ = rawPoints.Min(p => Mathf.Min(p.LY, p.RY));
        float maxZ = rawPoints.Max(p => Mathf.Max(p.LY, p.RY));
        
        // Use the minimum as the offset to ensure mesh starts at origin
        float offsetX = minX;
        float offsetZ = minZ;
        
        Debug.Log($"[CSVGeometryLoader] Raw data bounds: X[{minX:F2}, {maxX:F2}], Z[{minZ:F2}, {maxZ:F2}]");
        Debug.Log($"[CSVGeometryLoader] Using offset: ({offsetX:F2}, {offsetZ:F2}), scaleFactor: {scaleFactor}");

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
        
        // Verify normalization worked
        if (vertices.Count > 0)
        {
            float finalMinX = vertices.Min(v => v.x);
            float finalMaxX = vertices.Max(v => v.x);
            float finalMinZ = vertices.Min(v => v.z);
            float finalMaxZ = vertices.Max(v => v.z);
            Debug.Log($"[CSVGeometryLoader] Normalized mesh bounds: X[{finalMinX:F2}, {finalMaxX:F2}], Z[{finalMinZ:F2}, {finalMaxZ:F2}]");
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

        Debug.Log($"[CSVGeometryLoader] Recalculating normals and bounds for {vertices.Count} vertices...");
        Debug.Log($"[CSVGeometryLoader] Mesh bounds before: {mesh.bounds}");
        
        // Finalize Mesh Data (Normals/Bounds are still helpful for downstream code like RiverToGrid)
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        
        Debug.Log($"[CSVGeometryLoader] Mesh bounds after: {mesh.bounds}");
        Debug.Log($"[CSVGeometryLoader] Mesh center: {mesh.bounds.center}, size: {mesh.bounds.size}");

        Debug.Log("[CSVGeometryLoader] Mesh finalized. Setting up visualization...");
        
        // Setup MeshFilter and MeshRenderer to display the river geometry
        SetupMeshRenderer();

        Debug.Log("[CSVGeometryLoader] Mesh finalized. Invoking OnMeshLoaded event...");
        
        // Fire event to notify other components that mesh is ready
        OnMeshLoaded?.Invoke(mesh);
        
        Debug.Log("[CSVGeometryLoader] OnMeshLoaded event fired.");
    }

    /// <summary>
    /// Sets up the MeshFilter and MeshRenderer components and applies the RiverMaterial.
    /// </summary>
    private void SetupMeshRenderer()
    {
        Debug.Log("[CSVGeometryLoader] Setting up MeshFilter and MeshRenderer...");
        
        // Get or add MeshFilter component
        MeshFilter meshFilter = GetComponent<MeshFilter>();
        if (meshFilter == null)
        {
            meshFilter = gameObject.AddComponent<MeshFilter>();
            Debug.Log("[CSVGeometryLoader] Created MeshFilter component");
        }
        
        if (mesh == null)
        {
            Debug.LogError("[CSVGeometryLoader] Cannot setup renderer - mesh is null!");
            return;
        }
        
        meshFilter.mesh = mesh;
        Debug.Log($"[CSVGeometryLoader] ✓ Mesh assigned to MeshFilter: {mesh.vertexCount} vertices, {mesh.triangles.Length/3} triangles");
        Debug.Log($"[CSVGeometryLoader] Mesh bounds: center={mesh.bounds.center}, size={mesh.bounds.size}, min={mesh.bounds.min}, max={mesh.bounds.max}");
        Debug.Log($"[CSVGeometryLoader] GameObject position: {transform.position}, scale: {transform.localScale}");

        // Get or add MeshRenderer component
        MeshRenderer meshRenderer = GetComponent<MeshRenderer>();
        if (meshRenderer == null)
        {
            meshRenderer = gameObject.AddComponent<MeshRenderer>();
            Debug.Log("[CSVGeometryLoader] Created MeshRenderer component");
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
            meshRenderer.material = materialToUse;
            Debug.Log("[CSVGeometryLoader] ✓ RiverMaterial applied successfully.");
        }
        else
        {
            // Create a default material if none found - use a distinct color to make it visible
            Shader standardShader = Shader.Find("Standard");
            if (standardShader != null)
            {
                Material defaultMat = new Material(standardShader);
                defaultMat.color = new Color(0.8f, 0.2f, 0.2f, 1f); // Red color to distinguish from grid
                defaultMat.SetFloat("_Metallic", 0.3f);
                defaultMat.SetFloat("_Glossiness", 0.5f);
                meshRenderer.material = defaultMat;
                Debug.LogWarning($"[CSVGeometryLoader] RiverMaterial not found. Using default RED material. River geometry is at Y={defaultElevationY}. Please assign Materials/RiverMaterial in the Inspector.");
            }
            else
            {
                Debug.LogError("[CSVGeometryLoader] Could not find Standard shader or create default material.");
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
        
        Debug.Log($"[CSVGeometryLoader] ✓ MeshRenderer enabled. River geometry should be visible at Y={defaultElevationY}");
        Debug.Log($"[CSVGeometryLoader] River geometry mesh: {mesh.vertexCount} vertices forming a sinuous strip. Mesh size: {mesh.bounds.size.x:F3} x {mesh.bounds.size.z:F3} units.");
        Debug.Log($"[CSVGeometryLoader] TIP: If the river shape is hard to see, try increasing 'Scale Factor' in the Inspector (currently {scaleFactor}).");
    }
    
    void OnDrawGizmosSelected()
    {
        // Only draw gizmos when selected to avoid performance issues
        if (mesh != null && mesh.vertices != null && mesh.vertices.Length > 0)
        {
            Gizmos.color = Color.yellow;
            Gizmos.matrix = transform.localToWorldMatrix;
            
            // Draw the mesh bounds (lightweight)
            Gizmos.DrawWireCube(mesh.bounds.center, mesh.bounds.size);
            
            // Draw a simplified line along the left bank to show the river path
            // Only draw every Nth vertex to prevent freezing with large meshes (41k vertices)
            if (mesh.vertices.Length >= 4)
            {
                Gizmos.color = Color.cyan;
                // Only draw ~200 lines max to show the shape without freezing
                int step = Mathf.Max(2, (mesh.vertices.Length / 2) / 200); // Divide by 2 because we only want left bank
                for (int i = 0; i < mesh.vertices.Length - step * 2; i += step * 2)
                {
                    Vector3 v1 = transform.TransformPoint(mesh.vertices[i]);
                    Vector3 v2 = transform.TransformPoint(mesh.vertices[i + step * 2]);
                    Gizmos.DrawLine(v1, v2);
                }
            }
        }
    }

    /// <summary>
    /// Creates a mesh strip by connecting sequential cross-sections (L_i, R_i) to (L_{i+1}, R_{i+1}).
    /// Corrected winding order to ensure normals face upwards.
    /// </summary>
    private int[] StripTriangulate(int numCrossSections)
    {
        int numQuads = numCrossSections - 1;
        int[] triangles = new int[numQuads * 6];
        int t = 0;

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
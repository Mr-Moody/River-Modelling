using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Generates and updates a 3D terrain mesh around the river that shows erosion and deposition.
/// The terrain deforms as the river cuts into it.
/// </summary>
public class TerrainMesh : MonoBehaviour
{
    [Header("Terrain Configuration")]
    [Tooltip("Height of terrain above the riverbed. Higher values create more dramatic valleys.")]
    [Range(0.5f, 10f)]
    public float terrainHeight = 2.0f;
    
    [Tooltip("Smoothing factor for terrain elevation. Higher values create smoother terrain.")]
    [Range(0.1f, 5f)]
    public float terrainSmoothing = 1.0f;
    
    [Tooltip("Material for the terrain mesh.")]
    public Material terrainMaterial;
    
    [Header("References")]
    [Tooltip("Reference to RiverToGrid component to get grid bounds.")]
    public RiverToGrid riverToGrid;
    
    private Mesh terrainMesh;
    private MeshFilter meshFilter;
    private MeshRenderer meshRenderer;
    private Vector3[] terrainVertices;
    private int[] terrainTriangles;
    private float minX, maxX, minZ, maxZ;
    private int terrainResolutionX = 100;
    private int terrainResolutionZ = 100;
    private bool isInitialized = false;
    
    /// <summary>
    /// Initializes the terrain mesh based on river bounds.
    /// Uses the same positioning logic as MeshGrid - keeps GameObject at origin and uses offsets.
    /// Should only be called after RiverToGrid is initialized.
    /// </summary>
    public void InitializeTerrain()
    {
        if (isInitialized)
        {
            return; // Already initialized, skip
        }
        
        if (riverToGrid == null)
        {
            riverToGrid = GetComponent<RiverToGrid>();
            if (riverToGrid == null)
            {
                // Try to find it on parent
                riverToGrid = transform.parent?.GetComponent<RiverToGrid>();
                if (riverToGrid == null)
                {
                    Debug.LogError("[TerrainMesh] RiverToGrid component not found!");
                    return;
                }
            }
        }
        
        // Wait for RiverToGrid to be initialized before getting bounds
        if (!riverToGrid.isInitialized)
        {
            Debug.LogWarning("[TerrainMesh] RiverToGrid not initialized yet. Terrain will be initialized when grid is ready.");
            return;
        }
        
        // Get bounds from RiverToGrid (same as MeshGrid uses)
        // These bounds include padding and are in the same coordinate system as the simulation grid
        minX = riverToGrid.MinX;
        maxX = riverToGrid.MaxX;
        minZ = riverToGrid.MinZ;
        maxZ = riverToGrid.MaxZ;
        
        Debug.Log($"[TerrainMesh] Terrain bounds from RiverToGrid: X[{minX:F2}, {maxX:F2}], Z[{minZ:F2}, {maxZ:F2}]");
        
        // Note: The river mesh vertices are normalized to start at (0,0), but RiverToGrid's MinX/MinZ
        // include padding and may be negative. The terrain will use these bounds directly to align with
        // the simulation grid. The river mesh should be positioned to align with these bounds.
        
        // Keep GameObject at origin (same as MeshGrid) - vertices will use minX/minZ offsets directly
        transform.localPosition = Vector3.zero;
        transform.localRotation = Quaternion.identity;
        transform.localScale = Vector3.one;
        
        // Calculate resolution based on grid size
        float width = maxX - minX;
        float height = maxZ - minZ;
        terrainResolutionX = Mathf.Max(50, Mathf.RoundToInt(width / 0.5f));
        terrainResolutionZ = Mathf.Max(50, Mathf.RoundToInt(height / 0.5f));
        
        Debug.Log($"[TerrainMesh] Initializing terrain: {terrainResolutionX}x{terrainResolutionZ} vertices, Bounds: X[{minX:F2}, {maxX:F2}], Z[{minZ:F2}, {maxZ:F2}]");
        
        GenerateTerrainMesh();
        SetupRenderer();
        
        isInitialized = true;
    }
    
    /// <summary>
    /// Generates the initial terrain mesh with a flat surface.
    /// </summary>
    private void GenerateTerrainMesh()
    {
        terrainMesh = new Mesh();
        terrainMesh.name = "TerrainMesh";
        
        // Generate vertices
        terrainVertices = new Vector3[terrainResolutionX * terrainResolutionZ];
        Vector2[] uvs = new Vector2[terrainVertices.Length];
        
        float stepX = (maxX - minX) / (terrainResolutionX - 1);
        float stepZ = (maxZ - minZ) / (terrainResolutionZ - 1);
        
        for (int z = 0; z < terrainResolutionZ; z++)
        {
            for (int x = 0; x < terrainResolutionX; x++)
            {
                int index = z * terrainResolutionX + x;
                // Use same coordinate system as MeshGrid (using RiverToGrid bounds directly)
                float worldX = minX + x * stepX;
                float worldZ = minZ + z * stepZ;
                
                // Start with flat terrain at terrainHeight above origin
                terrainVertices[index] = new Vector3(worldX, terrainHeight, worldZ);
                uvs[index] = new Vector2((float)x / terrainResolutionX, (float)z / terrainResolutionZ);
            }
        }
        
        // Generate triangles
        List<int> triangles = new List<int>();
        for (int z = 0; z < terrainResolutionZ - 1; z++)
        {
            for (int x = 0; x < terrainResolutionX - 1; x++)
            {
                int i = z * terrainResolutionX + x;
                
                // Triangle 1
                triangles.Add(i);
                triangles.Add(i + terrainResolutionX);
                triangles.Add(i + 1);
                
                // Triangle 2
                triangles.Add(i + 1);
                triangles.Add(i + terrainResolutionX);
                triangles.Add(i + terrainResolutionX + 1);
            }
        }
        
        terrainTriangles = triangles.ToArray();
        
        terrainMesh.vertices = terrainVertices;
        terrainMesh.triangles = terrainTriangles;
        terrainMesh.uv = uvs;
        terrainMesh.RecalculateNormals();
        terrainMesh.RecalculateBounds();
        
        Debug.Log($"[TerrainMesh] Terrain mesh generated: {terrainVertices.Length} vertices, {terrainTriangles.Length / 3} triangles");
    }
    
    /// <summary>
    /// Sets up the MeshRenderer and MeshFilter components.
    /// </summary>
    private void SetupRenderer()
    {
        meshFilter = GetComponent<MeshFilter>();
        if (meshFilter == null)
        {
            meshFilter = gameObject.AddComponent<MeshFilter>();
        }
        meshFilter.mesh = terrainMesh;
        
        meshRenderer = GetComponent<MeshRenderer>();
        if (meshRenderer == null)
        {
            meshRenderer = gameObject.AddComponent<MeshRenderer>();
        }
        
        if (terrainMaterial == null)
        {
            // Create default terrain material
            Shader standardShader = Shader.Find("Standard");
            if (standardShader != null)
            {
                terrainMaterial = new Material(standardShader);
                terrainMaterial.color = new Color(0.4f, 0.3f, 0.2f); // Brown/earth color
                terrainMaterial.SetFloat("_Metallic", 0.0f);
                terrainMaterial.SetFloat("_Glossiness", 0.3f);
            }
        }
        
        meshRenderer.material = terrainMaterial;
        meshRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.On;
        meshRenderer.receiveShadows = true;
    }
    
    /// <summary>
    /// Updates the terrain mesh based on simulation results (bed elevation and erosion).
    /// </summary>
    public void UpdateTerrain(double[,] bedElevation, int[,] cellType, float cellSize, float gridMinX, float gridMinZ, int gridWidth, int gridHeight)
    {
        if (terrainMesh == null || terrainVertices == null)
        {
            return;
        }
        
        // Update each terrain vertex based on simulation
        // Use same coordinate system as MeshGrid (using RiverToGrid bounds directly)
        float stepX = (maxX - minX) / (terrainResolutionX - 1);
        float stepZ = (maxZ - minZ) / (terrainResolutionZ - 1);
        
        for (int z = 0; z < terrainResolutionZ; z++)
        {
            for (int x = 0; x < terrainResolutionX; x++)
            {
                int index = z * terrainResolutionX + x;
                // World space coordinates (same as MeshGrid uses)
                float worldX = minX + x * stepX;
                float worldZ = minZ + z * stepZ;
                
                // Convert world position to grid coordinates
                float gridX = (worldX - gridMinX) / cellSize;
                float gridZ = (worldZ - gridMinZ) / cellSize;
                
                // Clamp to valid grid bounds
                int xIdx = Mathf.Clamp(Mathf.RoundToInt(gridX), 0, gridWidth);
                int zIdx = Mathf.Clamp(Mathf.RoundToInt(gridZ), 0, gridHeight);
                
                // Get bed elevation from simulation
                float bedElev = 0f;
                if (xIdx < bedElevation.GetLength(0) && zIdx < bedElevation.GetLength(1))
                {
                    bedElev = (float)bedElevation[xIdx, zIdx];
                }
                
                // Determine terrain height based on cell type
                float finalHeight = terrainHeight;
                
                if (xIdx < cellType.GetLength(0) && zIdx < cellType.GetLength(1))
                {
                    if (cellType[xIdx, zIdx] == RiverCellType.FLUID)
                    {
                        // River channel - use bed elevation (river cuts into terrain)
                        // Make sure river is below surrounding terrain
                        finalHeight = bedElev - terrainHeight * 0.2f; // River channel is below terrain
                    }
                    else if (cellType[xIdx, zIdx] == RiverCellType.BANK)
                    {
                        // Bank area - transition between river and terrain
                        finalHeight = bedElev + terrainHeight * 0.1f;
                    }
                    else
                    {
                        // Terrain - use bed elevation + terrain height
                        // If bed elevation is negative (erosion), terrain follows it
                        finalHeight = Mathf.Max(bedElev + terrainHeight, terrainHeight * 0.5f);
                    }
                }
                else
                {
                    // Outside simulation area - use default terrain height
                    finalHeight = terrainHeight;
                }
                
                // Apply smoothing to prevent sharp edges
                if (x > 0 && x < terrainResolutionX - 1 && z > 0 && z < terrainResolutionZ - 1)
                {
                    // Average with neighbors for smoothing
                    float neighborAvg = (
                        terrainVertices[(z - 1) * terrainResolutionX + x].y +
                        terrainVertices[(z + 1) * terrainResolutionX + x].y +
                        terrainVertices[z * terrainResolutionX + (x - 1)].y +
                        terrainVertices[z * terrainResolutionX + (x + 1)].y
                    ) / 4f;
                    
                    finalHeight = Mathf.Lerp(finalHeight, neighborAvg, 1f / (1f + terrainSmoothing));
                }
                
                // Store in world space coordinates (same as MeshGrid)
                terrainVertices[index] = new Vector3(worldX, finalHeight, worldZ);
            }
        }
        
        // Apply updated vertices
        terrainMesh.vertices = terrainVertices;
        terrainMesh.RecalculateNormals();
        terrainMesh.RecalculateBounds();
    }
    
    /// <summary>
    /// Returns the terrain mesh.
    /// </summary>
    public Mesh GetMesh()
    {
        return terrainMesh;
    }
}

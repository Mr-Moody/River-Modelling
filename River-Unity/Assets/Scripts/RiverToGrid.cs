using UnityEngine;
using System;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// Converts unstructured 3D mesh geometry (from CsvGeometryLoader) into the 
/// structured 2D grid arrays (cellType, waterDepth) required by the Simulation.
/// </summary>
public class RiverToGrid : MonoBehaviour
{
    [Header("Configuration")]
    [Tooltip("Cell size for grid conversion. Minimum value is 0.1.")]
    [Range(0.1f, 10f)]
    public float cellSize = 0.5f;

    [Header("References")]
    [Tooltip("Reference to CSVGeometryLoader component on the same GameObject.")]
    public CSVGeometryLoader geometryLoader;

    // --- Events ---
    /// <summary>
    /// Event fired when grid is successfully initialized from mesh.
    /// Parameters: GridWidth, GridHeight, MinX, MinZ, MaxX, MaxZ
    /// </summary>
    public static event Action<int, int, float, float, float, float> OnGridInitialized;

    // Physical bounds of the loaded geometry (in world space, relative to (0,0,0))
    public float MinX { get; private set; }
    public float MaxX { get; private set; }
    public float MinZ { get; private set; }
    public float MaxZ { get; private set; }
    public int GridWidth { get; private set; }
    public int GridHeight { get; private set; }

    private List<Vector2> riverPolygon;
    public bool isInitialized { get; private set; } = false;

    void Awake()
    {
        // Try to get CSVGeometryLoader reference if not assigned
        if (geometryLoader == null)
        {
            geometryLoader = GetComponent<CSVGeometryLoader>();
        }
    }

    void OnEnable()
    {
        // Subscribe to mesh loaded event
        CSVGeometryLoader.OnMeshLoaded += HandleMeshLoaded;
    }

    void OnDisable()
    {
        // Unsubscribe from events
        CSVGeometryLoader.OnMeshLoaded -= HandleMeshLoaded;
    }

    /// <summary>
    /// Handles mesh loaded event from CSVGeometryLoader.
    /// </summary>
    private void HandleMeshLoaded(Mesh mesh)
    {
        if (mesh == null || mesh.vertices == null || mesh.vertices.Length == 0)
        {
            Debug.LogError("RiverToGrid: Received invalid mesh from CSVGeometryLoader.");
            return;
        }

        Initialize(mesh.vertices);
    }

    /// <summary>
    /// Initializes the converter, calculating bounds and setting up the river boundary polygon.
    /// </summary>
    private bool Initialize(Vector3[] meshVertices)
    {
        if (meshVertices == null || meshVertices.Length < 4 || meshVertices.Length % 2 != 0)
        {
            Debug.LogError("Mesh vertices are invalid for forming a strip polygon.");
            return false;
        }

        Vector3[] verticesToUse = meshVertices;

        if (verticesToUse == null || verticesToUse.Length < 4 || verticesToUse.Length % 2 != 0)
        {
            Debug.LogError("Mesh vertices are invalid for forming a strip polygon.");
            return false;
        }

        // 1. Calculate Bounds
        MinX = verticesToUse.Min(v => v.x);
        MaxX = verticesToUse.Max(v => v.x);
        MinZ = verticesToUse.Min(v => v.z);
        MaxZ = verticesToUse.Max(v => v.z);

        // 2. Calculate Grid Resolution
        float physicalWidth = MaxX - MinX;
        float physicalHeight = MaxZ - MinZ;

        // Force minimum cellSize to 0.1 if invalid
        if (cellSize <= 0f)
        {
            Debug.LogWarning($"RiverToGrid: Invalid cellSize ({cellSize}). Setting to minimum value of 0.1.");
            cellSize = 0.1f;
        }
        else if (cellSize < 0.1f)
        {
            Debug.LogWarning($"RiverToGrid: cellSize ({cellSize}) is below minimum. Setting to 0.1.");
            cellSize = 0.1f;
        }

        // Warn if cellSize is too large relative to mesh size (will create a very coarse grid)
        float recommendedCellSize = Mathf.Min(physicalWidth, physicalHeight) / 50f; // Aim for ~50 cells in smallest dimension
        if (cellSize > recommendedCellSize * 2f)
        {
            Debug.LogWarning($"[RiverToGrid] WARNING: cellSize ({cellSize}) is very large relative to mesh size ({physicalWidth:F3}x{physicalHeight:F3}). " +
                           $"This will create a coarse grid that may not capture the river's sinuous shape. " +
                           $"Recommended cellSize: ~{recommendedCellSize:F4} (will create ~{Mathf.CeilToInt(physicalWidth / recommendedCellSize)}x{Mathf.CeilToInt(physicalHeight / recommendedCellSize)} grid). " +
                           $"Current settings will create a {Mathf.CeilToInt(physicalWidth / cellSize)}x{Mathf.CeilToInt(physicalHeight / cellSize)} grid.");
        }

        // Calculate grid dimensions
        // Add a small padding (half a cell) to ensure grid covers the full river bounds
        float padding = cellSize * 0.5f;
        int calculatedWidth = Mathf.CeilToInt((physicalWidth + padding * 2) / cellSize);
        int calculatedHeight = Mathf.CeilToInt((physicalHeight + padding * 2) / cellSize);
        
        // Adjust bounds to center the padding
        MinX -= padding;
        MinZ -= padding;

        // Validate calculated dimensions before assigning
        if (calculatedWidth <= 0 || calculatedHeight <= 0)
        {
            Debug.LogError($"RiverToGrid: Invalid grid dimensions calculated: {calculatedWidth}x{calculatedHeight}. Physical size: {physicalWidth}x{physicalHeight}, CellSize: {cellSize}");
            return false;
        }

        GridWidth = calculatedWidth;
        GridHeight = calculatedHeight;

        // 3. Construct the River Boundary Polygon (Ordered list of 2D points)
        riverPolygon = new List<Vector2>();

        // Add Left Bank points in order (indices 0, 2, 4, ...)
        for (int i = 0; i < verticesToUse.Length; i += 2)
        {
            riverPolygon.Add(new Vector2(verticesToUse[i].x, verticesToUse[i].z));
        }

        // Add Right Bank points in reverse order (indices N-1, N-3, ...) to close the loop
        for (int i = verticesToUse.Length - 1; i >= 1; i -= 2)
        {
            riverPolygon.Add(new Vector2(verticesToUse[i].x, verticesToUse[i].z));
        }
        
        // Calculate bounding box for optimization
        CalculatePolygonBounds();

        isInitialized = true;

        // Fire event to notify other components that grid is initialized
        OnGridInitialized?.Invoke(GridWidth, GridHeight, MinX, MinZ, MaxX, MaxZ);

        Debug.Log($"[RiverToGrid] Grid initialized: {GridWidth}x{GridHeight} cells, CellSize: {cellSize}, Physical bounds: {MaxX - MinX:F1}m x {MaxZ - MinZ:F1}m");
        return true;
    }

    /// <summary>
    /// Public method to get cell size (used by other components).
    /// </summary>
    public float GetCellSize()
    {
        return cellSize;
    }

    /// <summary>
    /// Iterates over the PhysicsSolver's grid and initializes cell types and water depths
    /// based on the loaded river geometry.
    /// </summary>
    public void GenerateCellTypesAndDepths(PhysicsSolver solver, float initialWaterDepth)
    {
        // Check that solver grid dimensions match RiverToGrid calculated dimensions
        // Note: solver.nx = gridWidth + 1 and solver.ny = gridHeight + 1 (vertex count)
        // We compare gridWidth and gridHeight (cell count) which should match
        if (solver.gridWidth != GridWidth || solver.gridHeight != GridHeight)
        {
            Debug.LogError($"Solver grid dimensions ({solver.gridWidth}x{solver.gridHeight}) do not match RiverToGrid calculated dimensions ({GridWidth}x{GridHeight}).");
            return;
        }

        int totalCells = GridWidth * GridHeight;
        int fluidCells = 0;
        int bankCells = 0;

        // Note: solver arrays are (gridWidth+1) x (gridHeight+1) = (nx, ny)
        // We initialize cells [0..GridWidth-1, 0..GridHeight-1] based on geometry
        // Boundary cells at indices GridWidth and GridHeight will use default/nearest neighbor values
        
        for (int j = 0; j < GridHeight; j++) // Z-axis (height)
        {
            for (int i = 0; i < GridWidth; i++) // X-axis (width)
            {
                // Calculate the world-space center of this grid cell
                float worldX = MinX + (i * cellSize) + (cellSize / 2f);
                float worldZ = MinZ + (j * cellSize) + (cellSize / 2f);
                Vector2 gridPoint = new Vector2(worldX, worldZ);

                // Use the Point-in-Polygon test
                if (IsPointInPolygon(gridPoint))
                {
                    // Cell is inside the river boundary
                    solver.cellType[i, j] = RiverCellType.FLUID;
                    solver.waterDepth[i, j] = initialWaterDepth;
                    // Add initial velocity for visible flow with some variation
                    // Add slight variation based on position to create initial flow pattern
                    float flowVariation = 1.0f + 0.1f * Mathf.Sin(i * 0.1f) * Mathf.Cos(j * 0.1f);
                    solver.u[i, j] = 0.1 * flowVariation;
                    solver.v[i, j] = 0.01 * Mathf.Sin(i * 0.05f); // Small cross-flow component
                    fluidCells++;
                }
                else
                {
                    // Cell is outside the river boundary
                    solver.cellType[i, j] = RiverCellType.BANK;
                    solver.waterDepth[i, j] = 0.0f;
                    solver.u[i, j] = 0.0;
                    solver.v[i, j] = 0.0;
                    bankCells++;
                }

                // Initial Bed Elevation (h) is kept at 0, as no elevation data was provided in the CSV.
                solver.h[i, j] = 0.0f;
            }
        }
        
        // Initialize boundary cells (at indices GridWidth and GridHeight) using nearest neighbor values
        // This ensures the boundary cells have valid values for the solver
        for (int j = 0; j < GridHeight; j++)
        {
            // Right boundary (i = GridWidth)
            solver.cellType[GridWidth, j] = solver.cellType[GridWidth - 1, j];
            solver.waterDepth[GridWidth, j] = solver.waterDepth[GridWidth - 1, j];
            solver.u[GridWidth, j] = solver.u[GridWidth - 1, j];
            solver.v[GridWidth, j] = solver.v[GridWidth - 1, j];
            solver.h[GridWidth, j] = solver.h[GridWidth - 1, j];
        }
        
        for (int i = 0; i <= GridWidth; i++)
        {
            // Top boundary (j = GridHeight)
            solver.cellType[i, GridHeight] = solver.cellType[i, GridHeight - 1];
            solver.waterDepth[i, GridHeight] = solver.waterDepth[i, GridHeight - 1];
            solver.u[i, GridHeight] = solver.u[i, GridHeight - 1];
            solver.v[i, GridHeight] = solver.v[i, GridHeight - 1];
            solver.h[i, GridHeight] = solver.h[i, GridHeight - 1];
        }
        
        Debug.Log($"[RiverToGrid] Cell types generated: {fluidCells} fluid, {bankCells} bank cells");
        
        if (fluidCells == 0)
        {
            Debug.LogError($"[RiverToGrid] WARNING: No fluid cells found! The grid may be too coarse (cellSize={cellSize}) or the river polygon may be invalid. " +
                          $"Try reducing cellSize to capture the river's sinuous shape. Current grid: {GridWidth}x{GridHeight}.");
        }
        else if (fluidCells < totalCells * 0.01f)
        {
            Debug.LogWarning($"[RiverToGrid] WARNING: Very few fluid cells ({fluidCells}/{totalCells} = {fluidCells * 100f / totalCells:F1}%). " +
                           $"The grid may be too coarse. Consider reducing cellSize from {cellSize} to better capture the river shape.");
        }
    }

    // Cache polygon bounds for early rejection
    private float polygonMinX, polygonMaxX, polygonMinY, polygonMaxY;
    private bool boundsCalculated = false;

    /// <summary>
    /// Calculates and caches the bounding box of the polygon for fast rejection.
    /// </summary>
    private void CalculatePolygonBounds()
    {
        if (riverPolygon == null || riverPolygon.Count == 0)
        {
            boundsCalculated = false;
            return;
        }

        polygonMinX = riverPolygon.Min(p => p.x);
        polygonMaxX = riverPolygon.Max(p => p.x);
        polygonMinY = riverPolygon.Min(p => p.y);
        polygonMaxY = riverPolygon.Max(p => p.y);
        boundsCalculated = true;
    }

    /// <summary>
    /// Determines if a 2D point is inside the river boundary polygon using the Ray Casting Algorithm.
    /// Optimized with bounding box early rejection.
    /// </summary>
    private bool IsPointInPolygon(Vector2 point)
    {
        int count = riverPolygon.Count;
        if (count < 3) return false;

        // Early rejection using bounding box
        if (boundsCalculated)
        {
            if (point.x < polygonMinX || point.x > polygonMaxX || 
                point.y < polygonMinY || point.y > polygonMaxY)
            {
                return false; // Point is outside bounding box
            }
        }

        // Optimized ray casting - only check edges that could intersect
        int intersections = 0;
        for (int i = 0; i < count; i++)
        {
            Vector2 p1 = riverPolygon[i];
            Vector2 p2 = riverPolygon[(i + 1) % count];

            // Early skip if edge is completely above or below the point's y-coordinate
            if ((p1.y > point.y && p2.y > point.y) || (p1.y < point.y && p2.y < point.y))
            {
                continue;
            }

            // Check if the ray from 'point' intersects the segment (p1, p2)
            if (((p1.y <= point.y && point.y < p2.y) || (p2.y <= point.y && point.y < p1.y)) &&
                (point.x < (p2.x - p1.x) * (point.y - p1.y) / (p2.y - p1.y) + p1.x))
            {
                intersections++;
            }
        }

        return (intersections % 2 != 0);
    }
}
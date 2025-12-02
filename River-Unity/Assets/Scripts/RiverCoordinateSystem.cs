using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Handles coordinate transformations between world space and river-local space.
/// River-local coordinates:
/// - u: along the river (longitudinal, following the river's path)
/// - v: across the river (transverse, perpendicular to the river's path)
/// </summary>
public class RiverCoordinateSystem
{
    // River mesh structure
    private Vector3[] riverVertices;
    private int numCrossSections;  // Number of cross-sections along the river
    private int widthResolution;   // Number of points across the river width
    
    // Cached coordinate frames for each cross-section
    public struct CrossSectionFrame
    {
        public Vector3 center;           // Center point of cross-section
        public Vector3 longitudinalDir;   // Direction along river (normalized)
        public Vector3 transverseDir;     // Direction across river (normalized)
        public Vector3 normal;           // Upward normal
    }
    
    private CrossSectionFrame[] frames;
    private bool framesCalculated = false;
    
    /// <summary>
    /// Initializes the coordinate system from a structured river mesh.
    /// Mesh must be structured as: vertexIndex = rowIndex * widthResolution + columnIndex
    /// where rowIndex is along the river and columnIndex is across the river.
    /// </summary>
    public RiverCoordinateSystem(Vector3[] vertices, int numCrossSections, int widthResolution)
    {
        this.riverVertices = vertices;
        this.numCrossSections = numCrossSections;
        this.widthResolution = widthResolution;
        this.frames = new CrossSectionFrame[numCrossSections];
        
        if (vertices == null || vertices.Length != numCrossSections * widthResolution)
        {
            Debug.LogError($"[RiverCoordinateSystem] Invalid mesh structure. Expected {numCrossSections * widthResolution} vertices, got {(vertices?.Length ?? 0)}");
        }
    }
    
    /// <summary>
    /// Calculates coordinate frames for all cross-sections.
    /// </summary>
    public void CalculateFrames()
    {
        if (riverVertices == null || riverVertices.Length < widthResolution * 2)
        {
            Debug.LogError("[RiverCoordinateSystem] Cannot calculate frames - insufficient vertices");
            return;
        }
        
        // Calculate frames for each cross-section
        for (int i = 0; i < numCrossSections; i++)
        {
            CalculateFrameAtCrossSection(i);
        }
        
        framesCalculated = true;
    }
    
    /// <summary>
    /// Calculates the coordinate frame at a specific cross-section.
    /// </summary>
    private void CalculateFrameAtCrossSection(int crossSectionIndex)
    {
        if (crossSectionIndex < 0 || crossSectionIndex >= numCrossSections)
            return;
        
        // Get left and right bank vertices for this cross-section
        int leftIdx = crossSectionIndex * widthResolution;
        int rightIdx = crossSectionIndex * widthResolution + (widthResolution - 1);
        
        Vector3 leftBank = riverVertices[leftIdx];
        Vector3 rightBank = riverVertices[rightIdx];
        
        // Center of cross-section
        Vector3 center = (leftBank + rightBank) * 0.5f;
        
        // Transverse direction: from left bank to right bank (across river)
        Vector3 transverseDir = (rightBank - leftBank).normalized;
        
        // Longitudinal direction: along the river
        Vector3 longitudinalDir;
        if (crossSectionIndex == 0)
        {
            // First cross-section: use direction to next cross-section
            if (numCrossSections > 1)
            {
                int nextLeftIdx = (crossSectionIndex + 1) * widthResolution;
                int nextRightIdx = (crossSectionIndex + 1) * widthResolution + (widthResolution - 1);
                Vector3 nextCenter = (riverVertices[nextLeftIdx] + riverVertices[nextRightIdx]) * 0.5f;
                longitudinalDir = (nextCenter - center).normalized;
            }
            else
            {
                longitudinalDir = Vector3.right; // Fallback
            }
        }
        else if (crossSectionIndex == numCrossSections - 1)
        {
            // Last cross-section: use direction from previous cross-section
            int prevLeftIdx = (crossSectionIndex - 1) * widthResolution;
            int prevRightIdx = (crossSectionIndex - 1) * widthResolution + (widthResolution - 1);
            Vector3 prevCenter = (riverVertices[prevLeftIdx] + riverVertices[prevRightIdx]) * 0.5f;
            longitudinalDir = (center - prevCenter).normalized;
        }
        else
        {
            // Middle cross-sections: average direction from previous and to next
            int prevLeftIdx = (crossSectionIndex - 1) * widthResolution;
            int prevRightIdx = (crossSectionIndex - 1) * widthResolution + (widthResolution - 1);
            Vector3 prevCenter = (riverVertices[prevLeftIdx] + riverVertices[prevRightIdx]) * 0.5f;
            
            int nextLeftIdx = (crossSectionIndex + 1) * widthResolution;
            int nextRightIdx = (crossSectionIndex + 1) * widthResolution + (widthResolution - 1);
            Vector3 nextCenter = (riverVertices[nextLeftIdx] + riverVertices[nextRightIdx]) * 0.5f;
            
            Vector3 dirToNext = (nextCenter - center).normalized;
            Vector3 dirFromPrev = (center - prevCenter).normalized;
            longitudinalDir = ((dirToNext + dirFromPrev) * 0.5f).normalized;
        }
        
        // Ensure longitudinal and transverse are perpendicular
        // Project longitudinal onto plane perpendicular to transverse
        float dot = Vector3.Dot(longitudinalDir, transverseDir);
        longitudinalDir = (longitudinalDir - transverseDir * dot).normalized;
        
        // Normal: upward (cross product of longitudinal and transverse, then project to vertical)
        Vector3 normal = Vector3.Cross(longitudinalDir, transverseDir).normalized;
        // Ensure normal points upward (positive Y)
        if (normal.y < 0) normal = -normal;
        
        frames[crossSectionIndex] = new CrossSectionFrame
        {
            center = center,
            longitudinalDir = longitudinalDir,
            transverseDir = transverseDir,
            normal = normal
        };
    }
    
    /// <summary>
    /// Gets the coordinate frame at a specific cross-section index.
    /// </summary>
    public CrossSectionFrame GetFrame(int crossSectionIndex)
    {
        if (!framesCalculated)
        {
            CalculateFrames();
        }
        
        if (crossSectionIndex < 0 || crossSectionIndex >= numCrossSections)
        {
            Debug.LogWarning($"[RiverCoordinateSystem] Invalid cross-section index: {crossSectionIndex}");
            return frames[Mathf.Clamp(crossSectionIndex, 0, numCrossSections - 1)];
        }
        
        return frames[crossSectionIndex];
    }
    
    /// <summary>
    /// Converts world-space velocity to river-local velocity (u along river, v across river).
    /// </summary>
    public Vector2 WorldToLocalVelocity(Vector3 worldVelocity, int crossSectionIndex)
    {
        CrossSectionFrame frame = GetFrame(crossSectionIndex);
        
        // Project world velocity onto local coordinate axes
        float u = Vector3.Dot(worldVelocity, frame.longitudinalDir);  // Along river
        float v = Vector3.Dot(worldVelocity, frame.transverseDir);    // Across river
        
        return new Vector2(u, v);
    }
    
    /// <summary>
    /// Converts river-local velocity (u, v) to world-space velocity.
    /// </summary>
    public Vector3 LocalToWorldVelocity(float u, float v, int crossSectionIndex)
    {
        CrossSectionFrame frame = GetFrame(crossSectionIndex);
        
        // Combine local velocity components in world space
        Vector3 worldVelocity = frame.longitudinalDir * u + frame.transverseDir * v;
        
        return worldVelocity;
    }
    
    /// <summary>
    /// Gets the position of a vertex in river-local coordinates (s, w).
    /// s: distance along river (from start)
    /// w: position across river (0 = left bank, 1 = right bank)
    /// </summary>
    public Vector2 GetLocalCoordinates(int crossSectionIndex, int widthIndex)
    {
        if (crossSectionIndex < 0 || crossSectionIndex >= numCrossSections ||
            widthIndex < 0 || widthIndex >= widthResolution)
        {
            return Vector2.zero;
        }
        
        // Calculate distance along river (s)
        float s = 0f;
        for (int i = 1; i <= crossSectionIndex; i++)
        {
            int prevLeftIdx = (i - 1) * widthResolution;
            int prevRightIdx = (i - 1) * widthResolution + (widthResolution - 1);
            Vector3 prevCenter = (riverVertices[prevLeftIdx] + riverVertices[prevRightIdx]) * 0.5f;
            
            int currLeftIdx = i * widthResolution;
            int currRightIdx = i * widthResolution + (widthResolution - 1);
            Vector3 currCenter = (riverVertices[currLeftIdx] + riverVertices[currRightIdx]) * 0.5f;
            
            s += Vector3.Distance(prevCenter, currCenter);
        }
        
        // Position across river (w) - normalized 0 to 1
        float w = widthIndex / (float)(widthResolution - 1);
        
        return new Vector2(s, w);
    }
    
    /// <summary>
    /// Gets the vertex index from river-local coordinates (cross-section, width).
    /// </summary>
    public int GetVertexIndex(int crossSectionIndex, int widthIndex)
    {
        return crossSectionIndex * widthResolution + widthIndex;
    }
    
    /// <summary>
    /// Gets cross-section and width indices from vertex index.
    /// </summary>
    public (int crossSectionIndex, int widthIndex) GetIndicesFromVertex(int vertexIndex)
    {
        int crossSectionIndex = vertexIndex / widthResolution;
        int widthIndex = vertexIndex % widthResolution;
        return (crossSectionIndex, widthIndex);
    }
    
    public int NumCrossSections => numCrossSections;
    public int WidthResolution => widthResolution;
}


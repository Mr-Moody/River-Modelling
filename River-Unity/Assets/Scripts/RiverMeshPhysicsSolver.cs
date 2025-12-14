using UnityEngine;
using System;
using System.Collections.Generic;

/// <summary>
/// Physics solver that operates directly on the river mesh using river-local coordinates.
/// Uses u (along river) and v (across river) velocity components instead of world-space x/y.
/// </summary>
public class RiverMeshPhysicsSolver
{
    // River structure
    private int _numCrossSections;  // Number of cross-sections along river (resizable for reconnection)
    public int numCrossSections 
    { 
        get { return _numCrossSections; }
        private set { _numCrossSections = value; }
    }
    public readonly int widthResolution;    // Number of points across river width
    private RiverCoordinateSystem _coordinateSystem;
    public RiverCoordinateSystem coordinateSystem
    {
        get { return _coordinateSystem; }
        private set { _coordinateSystem = value; }
    }
    
    // Physics parameters
    public readonly double nu;              // Kinematic viscosity (m^2/s)
    public readonly double rho;             // Water density (kg/m^3)
    public readonly double g;               // Gravitational acceleration (m/s^2)
    public readonly double sedimentDensity;
    public readonly double porosity;
    public readonly double criticalShear;
    private readonly double _baseTransportCoefficient;
    public double transportCoefficient
    {
        get
        {
            if (MonteCarloParameters.UseMonteCarloTransportCoefficient)
            {
                return MonteCarloParameters.CurrentTransportCoefficient;
            }
            return _baseTransportCoefficient;
        }
    }
    public readonly double bankCriticalShear;
    public readonly double bankErosionRate;
    
    // Flow fields (river-local coordinates)
    // Arrays indexed as: [crossSectionIndex, widthIndex]
    public double[,] u;              // Velocity along river (longitudinal)
    public double[,] v;              // Velocity across river (transverse)
    public double[,] h;              // Bed elevation (changes from initial state)
    public double[,] waterDepth;    // Water depth
    public int[,] cellType;         // Cell type (FLUID, BANK, TERRAIN)
    
    // Initial bed elevation (stored separately to apply changes relative to initial state)
    private double[,] initialBedElevation;
    
    // Unit conversion: Unity units to meters (1 / scaleFactor, where scaleFactor converts meters to Unity)
    private double unityToMetersScale;
    
    // Bank migration tracking
    private double[,] cumulativeBankErosion;  // Cumulative erosion at bank cells (for migration)
    public double bankMigrationThreshold = 0.01;  // Erosion threshold before converting FLUID to BANK (meters)
    
    // Current erosion rate (stored from last ExnerEquation call)
    private double[,] current_dh_dt;  // Current bed elevation change rate
    
    // Debug logging for bank migration
    private static int debugLogCounter = 0;
    private const int DEBUG_LOG_INTERVAL = 100; // Log every N steps
    
    // Physical spacing (approximate, will be calculated from mesh)
    private double[] ds;             // Distance between cross-sections along river
    private double[] dw;             // Distance across river at each cross-section
    
    // Width-based velocity constraint
    private double[] initialChannelWidths;  // Initial channel width for each cross-section (meters)
    private bool initialWidthsStored = false;  // Flag to ensure initial widths are stored once
    
    public RiverMeshPhysicsSolver(Vector3[] riverVertices, int numCrossSections, int widthResolution,
                                  double nu = 1e-6, double rho = 1000.0, double g = 9.81,
                                  double sedimentDensity = 2650.0, double porosity = 0.4,
                                  double criticalShear = 0.05, double transportCoefficient = 0.1,
                                  double bankCriticalShear = 0.15, double bankErosionRate = 0.3,
                                  double unityToMetersScale = 2000.0)
    {
        this._numCrossSections = numCrossSections;
        this.widthResolution = widthResolution;
        this.nu = nu;
        this.rho = rho;
        this.g = g;
        this.sedimentDensity = sedimentDensity;
        this.porosity = porosity;
        this.criticalShear = criticalShear;
        this._baseTransportCoefficient = transportCoefficient;
        this.bankCriticalShear = bankCriticalShear;
        this.bankErosionRate = bankErosionRate;
        this.unityToMetersScale = unityToMetersScale;
        
        // Initialize coordinate system
        _coordinateSystem = new RiverCoordinateSystem(riverVertices, numCrossSections, widthResolution);
        _coordinateSystem.CalculateFrames();
        
        // Initialize arrays
        u = new double[numCrossSections, widthResolution];
        v = new double[numCrossSections, widthResolution];
        h = new double[numCrossSections, widthResolution];
        waterDepth = new double[numCrossSections, widthResolution];
        cellType = new int[numCrossSections, widthResolution];
        cumulativeBankErosion = new double[numCrossSections, widthResolution];
        current_dh_dt = new double[numCrossSections, widthResolution];
        
        // Initialize cumulative bank erosion and dh_dt to zero
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                cumulativeBankErosion[i, w] = 0.0;
                current_dh_dt[i, w] = 0.0;
            }
        }
        
        // Calculate spacing
        ds = new double[numCrossSections];
        dw = new double[numCrossSections];
        CalculateSpacing(riverVertices);
        
        InitializeFields();
    }
    
    /// <summary>
    /// Calculates spacing between cross-sections (ds) and across each cross-section (dw).
    /// Converts from Unity units to meters for physics calculations.
    /// </summary>
    private void CalculateSpacing(Vector3[] vertices)
    {
        // Calculate ds (distance along river between cross-sections)
        for (int i = 0; i < numCrossSections - 1; i++)
        {
            int currLeftIdx = i * widthResolution;
            int currRightIdx = i * widthResolution + (widthResolution - 1);
            Vector3 currCenter = (vertices[currLeftIdx] + vertices[currRightIdx]) * 0.5f;
            
            int nextLeftIdx = (i + 1) * widthResolution;
            int nextRightIdx = (i + 1) * widthResolution + (widthResolution - 1);
            Vector3 nextCenter = (vertices[nextLeftIdx] + vertices[nextRightIdx]) * 0.5f;
            
            // Convert Unity units to meters
            double distanceUnity = Vector3.Distance(currCenter, nextCenter);
            ds[i] = distanceUnity * unityToMetersScale;
        }
        // Last cross-section uses previous spacing
        if (numCrossSections > 1)
        {
            ds[numCrossSections - 1] = ds[numCrossSections - 2];
        }
        else
        {
            ds[0] = 1.0 * unityToMetersScale; // Default in meters
        }
        
        // Calculate dw (width across river at each cross-section)
        if (initialChannelWidths == null)
        {
            initialChannelWidths = new double[numCrossSections];
        }
        
        for (int i = 0; i < numCrossSections; i++)
        {
            int leftIdx = i * widthResolution;
            int rightIdx = i * widthResolution + (widthResolution - 1);
            // Convert Unity units to meters
            double widthUnity = Vector3.Distance(vertices[leftIdx], vertices[rightIdx]);
            double widthMeters = widthUnity * unityToMetersScale;
            dw[i] = widthMeters / (widthResolution - 1); // Average spacing across width in meters
            
            // Store initial channel width (only on first call, or if not yet stored)
            if (!initialWidthsStored || initialChannelWidths[i] <= 0)
            {
                initialChannelWidths[i] = widthMeters;
            }
        }
        
        // Mark as stored after first complete calculation
        if (!initialWidthsStored)
        {
            initialWidthsStored = true;
        }
    }
    
    /// <summary>
    /// Calculates current channel widths from mesh vertex positions.
    /// Returns width in meters for each cross-section.
    /// </summary>
    private double[] CalculateCurrentChannelWidths(Vector3[] vertices)
    {
        if (vertices == null || vertices.Length != numCrossSections * widthResolution)
        {
            // Fallback: return current dw values scaled by widthResolution
            double[] widths = new double[numCrossSections];
            for (int i = 0; i < numCrossSections; i++)
            {
                widths[i] = dw[i] * (widthResolution - 1);
            }
            return widths;
        }
        
        double[] currentWidths = new double[numCrossSections];
        for (int i = 0; i < numCrossSections; i++)
        {
            int leftIdx = i * widthResolution;
            int rightIdx = i * widthResolution + (widthResolution - 1);
            
            if (leftIdx < vertices.Length && rightIdx < vertices.Length)
            {
                // Calculate width in Unity units, then convert to meters
                double widthUnity = Vector3.Distance(vertices[leftIdx], vertices[rightIdx]);
                double widthMeters = widthUnity * unityToMetersScale;
                currentWidths[i] = widthMeters;
            }
            else
            {
                // Fallback to spacing-based calculation
                currentWidths[i] = dw[i] * (widthResolution - 1);
            }
        }
        
        return currentWidths;
    }
    
    /// <summary>
    /// Initializes flow fields.
    /// </summary>
    private void InitializeFields()
    {
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                // Assume all cells are fluid initially (can be updated based on geometry)
                cellType[i, w] = RiverCellType.FLUID;
                waterDepth[i, w] = 0.5;
                u[i, w] = 0.1;  // Initial flow along river
                v[i, w] = 0.0;  // No initial cross-flow
                h[i, w] = 0.0;
            }
        }
    }
    
    /// <summary>
    /// Updates cell types based on river geometry (identifies banks).
    /// </summary>
    public void UpdateCellTypes(Vector3[] vertices)
    {
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                // Identify banks (first and last width indices)
                if (w == 0 || w == widthResolution - 1)
                {
                    cellType[i, w] = RiverCellType.BANK;
                    waterDepth[i, w] = 0.0;
                }
                else
                {
                    cellType[i, w] = RiverCellType.FLUID;
                }
            }
        }
    }
    
    /// <summary>
    /// Initializes bed elevation h from mesh vertex Y coordinates.
    /// Converts Unity units to meters and stores initial elevations separately.
    /// Sets h to 0 (changes from initial state, in meters).
    /// </summary>
    public void InitializeBedElevationFromMesh(Vector3[] vertices)
    {
        // Initialize the initialBedElevation array
        initialBedElevation = new double[numCrossSections, widthResolution];
        
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                int vertexIdx = i * widthResolution + w;
                if (vertexIdx < vertices.Length)
                {
                    // Convert Unity Y coordinate to meters and store initial elevation
                    double unityY = vertices[vertexIdx].y;
                    initialBedElevation[i, w] = unityY * unityToMetersScale;
                    // Initialize h to 0 (bed elevation changes start at zero, in meters)
                    h[i, w] = 0.0;
                }
                else
                {
                    initialBedElevation[i, w] = 0.0;
                    h[i, w] = 0.0;
                }
            }
        }
    }
    
    /// <summary>
    /// Gets the initial bed elevation at the specified cross-section and width index (in meters).
    /// </summary>
    public double GetInitialBedElevation(int i, int w)
    {
        if (initialBedElevation != null && i >= 0 && i < numCrossSections && w >= 0 && w < widthResolution)
        {
            return initialBedElevation[i, w];
        }
        return 0.0;
    }
    
    /// <summary>
    /// Gets the scale factor to convert meters to Unity units (1 / unityToMetersScale).
    /// </summary>
    public double GetMetersToUnityScale()
    {
        return (unityToMetersScale > 0.0) ? (1.0 / unityToMetersScale) : 0.0005;
    }
    
    /// <summary>
    /// Performs one Navier-Stokes time step using river-local coordinates.
    /// </summary>
    public void NavierStokesStep(float dt)
    {
        double[,] uNew = (double[,])u.Clone();
        double[,] vNew = (double[,])v.Clone();
        
        // Calculate water surface elevation
        // h represents changes from initial, so actual bed elevation = initialBedElevation + h
        double[,] waterSurface = new double[numCrossSections, widthResolution];
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                double actualBedElevation = (initialBedElevation != null) 
                    ? initialBedElevation[i, w] + h[i, w] 
                    : h[i, w]; // Fallback if initialBedElevation not initialized
                waterSurface[i, w] = actualBedElevation + waterDepth[i, w];
            }
        }
        
        // Solve velocity equations for interior cells
        for (int i = 1; i < numCrossSections - 1; i++)
        {
            for (int w = 1; w < widthResolution - 1; w++)
            {
                if (cellType[i, w] != RiverCellType.FLUID)
                {
                    continue; // Skip non-fluid cells
                }
                
                double ds_curr = ds[i];
                double dw_curr = dw[i];
                
                // Advection terms (upwind)
                double u_adv_s = u[i, w] > 0 ? 
                    u[i, w] * (u[i, w] - u[i - 1, w]) / ds_curr : 
                    u[i, w] * (u[i + 1, w] - u[i, w]) / ds_curr;
                
                double u_adv_w = v[i, w] > 0 ? 
                    u[i, w] * (u[i, w] - u[i, w - 1]) / dw_curr : 
                    u[i, w] * (u[i, w + 1] - u[i, w]) / dw_curr;
                
                double v_adv_s = u[i, w] > 0 ? 
                    u[i, w] * (v[i, w] - v[i - 1, w]) / ds_curr : 
                    u[i, w] * (v[i + 1, w] - v[i, w]) / ds_curr;
                
                double v_adv_w = v[i, w] > 0 ? 
                    v[i, w] * (v[i, w] - v[i, w - 1]) / dw_curr : 
                    v[i, w] * (v[i, w + 1] - v[i, w]) / dw_curr;
                
                // Pressure gradient (water surface slope)
                double p_grad_s = -g * (waterSurface[i + 1, w] - waterSurface[i - 1, w]) / (2 * ds_curr);
                double p_grad_w = -g * (waterSurface[i, w + 1] - waterSurface[i, w - 1]) / (2 * dw_curr);
                
                // Viscous diffusion (Laplacian)
                double u_laplacian = ((u[i + 1, w] - 2 * u[i, w] + u[i - 1, w]) / (ds_curr * ds_curr)) +
                                     ((u[i, w + 1] - 2 * u[i, w] + u[i, w - 1]) / (dw_curr * dw_curr));
                
                double v_laplacian = ((v[i + 1, w] - 2 * v[i, w] + v[i - 1, w]) / (ds_curr * ds_curr)) +
                                     ((v[i, w + 1] - 2 * v[i, w] + v[i, w - 1]) / (dw_curr * dw_curr));
                
                // Bed slope effect
                double h_slope_s = -g * waterDepth[i, w] * (h[i + 1, w] - h[i - 1, w]) / (2 * ds_curr);
                double h_slope_w = -g * waterDepth[i, w] * (h[i, w + 1] - h[i, w - 1]) / (2 * dw_curr);
                
                // Centrifugal force effect at bends (increases velocity at outer bank)
                // In real rivers, water at outer bank moves faster due to centrifugal effects
                // Calculate curvature: how much the river is turning
                double curvature = CalculateCurvature(i);
                double centrifugal_effect = 0.0;
                if (Math.Abs(curvature) > 1e-6)
                {
                    // Normalized position across width: 0 = left bank, 1 = right bank
                    double normalized_w = (double)w / (widthResolution - 1);
                    // Centrifugal force: F = v^2 / R, where R is radius of curvature
                    // For a bend, outer bank should have higher velocity
                    double u_mag = Math.Sqrt(u[i, w] * u[i, w] + v[i, w] * v[i, w]);
                    double radius = 1.0 / Math.Max(Math.Abs(curvature), 1e-6); // Avoid division by zero
                    
                    // Determine which side is outer bank based on curvature direction
                    // Positive curvature = left turn (outer bank on right, w near 1)
                    // Negative curvature = right turn (outer bank on left, w near 0)
                    double outer_bank_factor = curvature > 0 ? 
                        (normalized_w - 0.5) :  // Left turn: right side is outer
                        (0.5 - normalized_w);   // Right turn: left side is outer
                    
                    // Centrifugal acceleration: pushes water toward outer bank
                    // This increases longitudinal velocity (u) at outer bank
                    double centrifugal_accel = (u_mag * u_mag) / Math.Max(radius, 1.0);
                    // Apply as additional acceleration in longitudinal direction at outer bank
                    centrifugal_effect = centrifugal_accel * outer_bank_factor * 0.2; // Scale factor
                }
                
                // Update velocities (add centrifugal effect to longitudinal velocity at outer bank)
                uNew[i, w] = u[i, w] + dt * (-u_adv_s - u_adv_w + p_grad_s + nu * u_laplacian + h_slope_s + centrifugal_effect);
                vNew[i, w] = v[i, w] + dt * (-v_adv_s - v_adv_w + p_grad_w + nu * v_laplacian + h_slope_w);
                
                // Clamp velocities
                double maxVelocity = 10.0;
                uNew[i, w] = Math.Max(-maxVelocity, Math.Min(maxVelocity, uNew[i, w]));
                vNew[i, w] = Math.Max(-maxVelocity, Math.Min(maxVelocity, vNew[i, w]));
            }
        }
        
        // Apply boundary conditions
        ApplyBoundaryConditions(uNew, vNew);
        
        // Update water depth (continuity equation)
        waterDepth = SolveContinuityEquation(waterDepth, uNew, vNew, dt);
        
        // Final assignment
        u = uNew;
        v = vNew;
        
        // Final safety clamp
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                double maxVelocity = 10.0;
                u[i, w] = Math.Max(-maxVelocity, Math.Min(maxVelocity, u[i, w]));
                v[i, w] = Math.Max(-maxVelocity, Math.Min(maxVelocity, v[i, w]));
                waterDepth[i, w] = Math.Max(0.0, Math.Min(10.0, waterDepth[i, w]));
            }
        }
    }
    
    /// <summary>
    /// Applies width-based velocity constraint: reduces velocity when channel widens.
    /// Implements constant discharge principle: Q = width × depth × velocity.
    /// When width increases, velocity decreases proportionally to maintain constant discharge.
    /// </summary>
    /// <param name="vertices">Current mesh vertex positions (optional, will use spacing if null)</param>
    /// <param name="feedbackStrength">Strength of feedback (0.0 = disabled, 1.0 = full inverse relationship)</param>
    public void ApplyWidthBasedVelocityConstraint(Vector3[] vertices = null, double feedbackStrength = 1.0)
    {
        // Skip if feedback is disabled
        if (feedbackStrength <= 0.0)
            return;
        
        // Ensure initial widths are stored
        if (initialChannelWidths == null || !initialWidthsStored)
        {
            if (vertices != null)
            {
                CalculateSpacing(vertices);
            }
            else
            {
                // Cannot apply constraint without initial widths or vertices
                return;
            }
        }
        
        // Validate initial widths array matches current cross-section count
        if (initialChannelWidths.Length != numCrossSections)
        {
            // Recalculate initial widths if array size doesn't match (e.g., after reconnection)
            if (vertices != null)
            {
                initialWidthsStored = false;
                CalculateSpacing(vertices);
            }
            else
            {
                return;
            }
        }
        
        // Calculate current channel widths
        double[] currentWidths = CalculateCurrentChannelWidths(vertices);
        
        // Apply velocity reduction factor for each cross-section
        for (int i = 0; i < numCrossSections; i++)
        {
            if (i >= initialChannelWidths.Length || i >= currentWidths.Length)
                continue;
                
            double initialWidth = initialChannelWidths[i];
            double currentWidth = currentWidths[i];
            
            // Skip if initial width is invalid or current width is invalid
            if (initialWidth <= 0 || currentWidth <= 0)
                continue;
            
            // Calculate velocity reduction factor: v_new = v_old × (W_initial / W_current)
            // Clamp minimum width to 10% of initial to prevent division by zero
            double minWidth = initialWidth * 0.1;
            double safeCurrentWidth = Math.Max(currentWidth, minWidth);
            double baseVelocityFactor = initialWidth / safeCurrentWidth;
            
            // Apply feedback strength: interpolate between no change (1.0) and full inverse (baseVelocityFactor)
            double velocityFactor = 1.0 + (baseVelocityFactor - 1.0) * feedbackStrength;
            
            // Clamp velocity factor to reasonable range (0.1 to 10.0) to prevent extreme values
            velocityFactor = Math.Max(0.1, Math.Min(10.0, velocityFactor));
            
            // Apply to all FLUID cells in this cross-section
            for (int w = 0; w < widthResolution; w++)
            {
                if (cellType[i, w] == RiverCellType.FLUID)
                {
                    u[i, w] *= velocityFactor;
                    v[i, w] *= velocityFactor;
                }
            }
        }
    }
    
    /// <summary>
    /// Calculates curvature at a cross-section (how much the river is turning).
    /// Returns curvature magnitude (positive value).
    /// </summary>
    private double CalculateCurvature(int crossSectionIndex)
    {
        if (crossSectionIndex < 1 || crossSectionIndex >= numCrossSections - 1)
            return 0.0;
        
        // Get coordinate frames
        var framePrev = coordinateSystem.GetFrame(crossSectionIndex - 1);
        var frameCurr = coordinateSystem.GetFrame(crossSectionIndex);
        var frameNext = coordinateSystem.GetFrame(crossSectionIndex + 1);
        
        // Calculate change in direction along river
        Vector3 dirChange = frameNext.longitudinalDir - framePrev.longitudinalDir;
        
        // Curvature is related to the magnitude of direction change
        // Use cross product to determine turn direction
        Vector3 cross = Vector3.Cross(framePrev.longitudinalDir, frameCurr.longitudinalDir);
        double curvature = Math.Sqrt(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z);
        
        return curvature;
    }
    
    /// <summary>
    /// Applies boundary conditions at banks and ends of river.
    /// Includes bank collision effects with partial reflection.
    /// </summary>
    private void ApplyBoundaryConditions(double[,] uNew, double[,] vNew)
    {
        // Bank boundaries (w = 0 and w = widthResolution - 1): no-slip with collision effects
        for (int i = 0; i < numCrossSections; i++)
        {
            // Left bank (w = 0)
            uNew[i, 0] = 0.0;
            vNew[i, 0] = 0.0;
            
            // Right bank (w = widthResolution - 1)
            uNew[i, widthResolution - 1] = 0.0;
            vNew[i, widthResolution - 1] = 0.0;
            
            // Bank collision effects will be applied after setting bank velocities to zero
        }
        
        // Upstream boundary (i = 0): maintain inflow
        for (int w = 1; w < widthResolution - 1; w++)
        {
            uNew[0, w] = u[0, w]; // Keep current value or set to inflow
            vNew[0, w] = 0.0;     // No cross-flow at inlet
        }
        
        // Downstream boundary (i = numCrossSections - 1): zero gradient
        for (int w = 1; w < widthResolution - 1; w++)
        {
            if (numCrossSections > 1)
            {
                uNew[numCrossSections - 1, w] = uNew[numCrossSections - 2, w];
                vNew[numCrossSections - 1, w] = vNew[numCrossSections - 2, w];
            }
        }
        
        // Bank collision effects: partial reflection of fluid velocity near banks
        // This models water hitting the bank and bouncing back with energy loss
        // Matches physics.py: modifies FLUID cells adjacent to banks
        double reflectionCoeff = 0.3; // 30% reflection, 70% energy loss
        
        for (int i = 0; i < numCrossSections; i++)
        {
            // Left bank (w=0) collision: check right neighbor (w=1) for fluid flowing left (negative v)
            if (widthResolution > 1 && cellType[i, 1] == RiverCellType.FLUID)
            {
                if (vNew[i, 1] < 0) // Flowing toward left bank
                {
                    vNew[i, 1] *= -reflectionCoeff; // Partial reflection with energy loss
                }
            }
            
            // Right bank (w=widthResolution-1) collision: check left neighbor for fluid flowing right (positive v)
            if (widthResolution > 1 && cellType[i, widthResolution - 2] == RiverCellType.FLUID)
            {
                if (vNew[i, widthResolution - 2] > 0) // Flowing toward right bank
                {
                    vNew[i, widthResolution - 2] *= -reflectionCoeff; // Partial reflection with energy loss
                }
            }
        }
    }
    
    /// <summary>
    /// Solves the continuity equation to update water depth.
    /// </summary>
    private double[,] SolveContinuityEquation(double[,] h_w, double[,] u, double[,] v, float dt)
    {
        double[,] h_w_new = (double[,])h_w.Clone();
        double equilibriumDepth = 0.5;
        double relaxationTime = 0.01;
        
        for (int i = 1; i < numCrossSections - 1; i++)
        {
            for (int w = 1; w < widthResolution - 1; w++)
            {
                if (cellType[i, w] != RiverCellType.FLUID)
                {
                    h_w_new[i, w] = 0.0;
                    continue;
                }
                
                double ds_curr = ds[i];
                double dw_curr = dw[i];
                
                // Divergence in river-local coordinates
                double div_u = (u[i + 1, w] - u[i - 1, w]) / (2 * ds_curr) +
                               (v[i, w + 1] - v[i, w - 1]) / (2 * dw_curr);
                
                // Relaxation term
                double relaxationTerm = -(h_w[i, w] - equilibriumDepth) / relaxationTime;
                
                // Flow adjustment
                double flowAdjustment = -div_u * 0.01;
                flowAdjustment = Math.Max(-0.01, Math.Min(0.01, flowAdjustment));
                
                double dh_w_dt = 0.95 * relaxationTerm + 0.05 * flowAdjustment;
                
                // Water depth change rate - scale for large time steps
                double maxChangeRate = 0.005;
                if (dt > 1.0)
                {
                    // For large time steps, allow faster water depth changes
                    maxChangeRate = 0.05; // 10x faster for large steps
                }
                else if (dt > 0.1)
                {
                    maxChangeRate = 0.015; // 3x faster for medium steps
                }
                double maxChange = maxChangeRate * equilibriumDepth;
                dh_w_dt = Math.Max(-maxChange / dt, Math.Min(maxChange / dt, dh_w_dt));
                
                h_w_new[i, w] = h_w[i, w] + dh_w_dt * dt;
                h_w_new[i, w] = Math.Max(0.01, Math.Min(2.0, h_w_new[i, w]));
            }
        }
        
        return h_w_new;
    }
    
    /// <summary>
    /// Converts river-local velocities to world-space velocities for a vertex.
    /// </summary>
    public Vector3 GetWorldVelocity(int crossSectionIndex, int widthIndex)
    {
        return coordinateSystem.LocalToWorldVelocity((float)u[crossSectionIndex, widthIndex],
                                                      (float)v[crossSectionIndex, widthIndex],
                                                      crossSectionIndex);
    }
    
    /// <summary>
    /// Gets velocity magnitude at a vertex.
    /// </summary>
    public double GetVelocityMagnitude(int crossSectionIndex, int widthIndex)
    {
        return Math.Sqrt(u[crossSectionIndex, widthIndex] * u[crossSectionIndex, widthIndex] +
                         v[crossSectionIndex, widthIndex] * v[crossSectionIndex, widthIndex]);
    }
    
    /// <summary>
    /// Gets the current bed elevation change rate (dh_dt) at a vertex.
    /// Positive values indicate deposition (bed rising), negative values indicate erosion (bed lowering).
    /// </summary>
    public double GetErosionRate(int crossSectionIndex, int widthIndex)
    {
        if (current_dh_dt == null) return 0.0;
        return current_dh_dt[crossSectionIndex, widthIndex];
    }
    
    /// <summary>
    /// Gets the cumulative bank erosion at a vertex.
    /// Higher values indicate more bank erosion and potential bank migration.
    /// </summary>
    public double GetCumulativeBankErosion(int crossSectionIndex, int widthIndex)
    {
        if (cumulativeBankErosion == null) return 0.0;
        return cumulativeBankErosion[crossSectionIndex, widthIndex];
    }
    
    /// <summary>
    /// Checks if a cell is experiencing active bank migration (cumulative erosion near threshold).
    /// </summary>
    public bool IsBankMigrating(int crossSectionIndex, int widthIndex)
    {
        if (cumulativeBankErosion == null) return false;
        // Consider bank migrating if cumulative erosion is above 50% of threshold
        return cumulativeBankErosion[crossSectionIndex, widthIndex] >= bankMigrationThreshold * 0.5;
    }
    
    /// <summary>
    /// Gets the bank edge positions for a given cross-section.
    /// Returns the width indices where the left and right banks are located.
    /// </summary>
    /// <param name="crossSectionIndex">The cross-section index</param>
    /// <returns>Tuple of (leftBankEdge, rightBankEdge) width indices. Returns (-1, -1) if not found.</returns>
    public (int leftBankEdge, int rightBankEdge) GetBankEdges(int crossSectionIndex)
    {
        int leftBankEdge = -1;
        int rightBankEdge = -1;
        
        // Find the left bank edge: find the rightmost BANK cell (innermost from left)
        for (int w = 0; w < widthResolution - 1; w++)
        {
            if (cellType[crossSectionIndex, w] == RiverCellType.BANK && cellType[crossSectionIndex, w + 1] == RiverCellType.FLUID)
            {
                leftBankEdge = w;
                break;
            }
        }
        
        // If no bank edge found at boundary, check if w=0 is bank
        if (leftBankEdge == -1 && cellType[crossSectionIndex, 0] == RiverCellType.BANK)
        {
            leftBankEdge = 0;
        }
        
        // Find the right bank edge: find the leftmost BANK cell (innermost from right)
        for (int w = widthResolution - 1; w > 0; w--)
        {
            if (cellType[crossSectionIndex, w] == RiverCellType.BANK && cellType[crossSectionIndex, w - 1] == RiverCellType.FLUID)
            {
                rightBankEdge = w;
                break;
            }
        }
        
        // If no bank edge found at boundary, check if rightmost is bank
        if (rightBankEdge == -1 && cellType[crossSectionIndex, widthResolution - 1] == RiverCellType.BANK)
        {
            rightBankEdge = widthResolution - 1;
        }
        
        return (leftBankEdge, rightBankEdge);
    }
    
    // --- Sediment and Erosion Methods (from physics.py) ---
    
    /// <summary>
    /// Computes bed and bank shear stress from velocity field.
    /// Includes enhanced stress at bank walls due to water-wall collisions.
    /// </summary>
    public double[,] ComputeShearStress()
    {
        double[,] tau = new double[numCrossSections, widthResolution];
        double n = 0.03;  // Manning's roughness for bed
        double n_bank = 0.05;  // Manning's roughness for banks
        
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                if (cellType[i, w] == RiverCellType.FLUID)
                {
                    // For FLUID cells, compute normal shear stress
                    double u_mag = GetVelocityMagnitude(i, w);
                    double R = Math.Max(waterDepth[i, w], 0.001);  // Minimum depth to avoid division by zero
                    
                    // Compute friction velocity (u_star) using Manning's roughness
                    double u_star = Math.Sqrt(g * n * n * u_mag * u_mag / Math.Pow(R, 1.0 / 3.0));
                    
                    // Base bed shear stress
                    tau[i, w] = rho * u_star * u_star;
                }
                else
                {
                    // For BANK cells, compute stress based on adjacent FLUID cells
                    // Banks experience stress from water flowing past them, not from their own (zero) velocity
                    double maxAdjacentStress = 0.0;
                    double maxAdjacentVelocity = 0.0;
                    double maxAdjacentDepth = 0.0;
                    
                    // Check left neighbor (if fluid)
                    if (w > 0 && cellType[i, w - 1] == RiverCellType.FLUID)
                    {
                        double u_mag_adj = GetVelocityMagnitude(i, w - 1);
                        double R_adj = Math.Max(waterDepth[i, w - 1], 0.001);
                        double u_star_adj = Math.Sqrt(g * n_bank * n_bank * u_mag_adj * u_mag_adj / Math.Pow(R_adj, 1.0 / 3.0));
                        double stress_adj = rho * u_star_adj * u_star_adj;
                        if (stress_adj > maxAdjacentStress) 
                        {
                            maxAdjacentStress = stress_adj;
                            maxAdjacentVelocity = u_mag_adj;
                            maxAdjacentDepth = waterDepth[i, w - 1];
                        }
                    }
                    
                    // Check right neighbor (if fluid)
                    if (w < widthResolution - 1 && cellType[i, w + 1] == RiverCellType.FLUID)
                    {
                        double u_mag_adj = GetVelocityMagnitude(i, w + 1);
                        double R_adj = Math.Max(waterDepth[i, w + 1], 0.001);
                        double u_star_adj = Math.Sqrt(g * n_bank * n_bank * u_mag_adj * u_mag_adj / Math.Pow(R_adj, 1.0 / 3.0));
                        double stress_adj = rho * u_star_adj * u_star_adj;
                        if (stress_adj > maxAdjacentStress) 
                        {
                            maxAdjacentStress = stress_adj;
                            maxAdjacentVelocity = u_mag_adj;
                            maxAdjacentDepth = waterDepth[i, w + 1];
                        }
                    }
                    
                    // Use maximum stress from adjacent fluid cells
                    // If no adjacent fluid, use minimum stress based on minimum flow assumption
                    tau[i, w] = maxAdjacentStress > 0.0 ? maxAdjacentStress : 0.0;
                }
            }
        }
        
        // Add bank collision stress (additional stress from water hitting bank walls)
        tau = AddBankCollisionStress(tau);
        
        return tau;
    }
    
    /// <summary>
    /// Adds additional shear stress at bank walls due to water-wall collisions.
    /// </summary>
    private double[,] AddBankCollisionStress(double[,] tau_base)
    {
        double[,] tau = (double[,])tau_base.Clone();
        double collisionFactor = 1.5;
        
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                if (cellType[i, w] == RiverCellType.BANK)
                {
                    // Check left neighbor (w-1) for fluid flowing right (positive v)
                    if (w > 0 && cellType[i, w - 1] == RiverCellType.FLUID && v[i, w - 1] > 0)
                    {
                        double v_normal = Math.Abs(v[i, w - 1]);
                        double collisionStress = rho * collisionFactor * v_normal * v_normal;
                        tau[i, w] += collisionStress;
                    }
                    // Check right neighbor (w+1) for fluid flowing left (negative v)
                    if (w < widthResolution - 1 && cellType[i, w + 1] == RiverCellType.FLUID && v[i, w + 1] < 0)
                    {
                        double v_normal = Math.Abs(v[i, w + 1]);
                        double collisionStress = rho * collisionFactor * v_normal * v_normal;
                        tau[i, w] += collisionStress;
                    }
                }
            }
        }
        return tau;
    }
    
    /// <summary>
    /// Computes sediment transport flux magnitude from bed shear stress.
    /// Uses Meyer-Peter and Müller type formulation with bed slope effects.
    /// </summary>
    public double[,] SedimentFlux(double[,] tau)
    {
        double[,] qs_magnitude = new double[numCrossSections, widthResolution];
        
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                // Compute excess shear stress
                double tau_excess = Math.Max(tau[i, w] - criticalShear, 0.0);
                
                // Base sediment flux (Meyer-Peter and Müller type)
                qs_magnitude[i, w] = transportCoefficient * Math.Pow(tau_excess, 1.5);
                
                // Apply bed slope effect
                double ds_curr = ds[i];
                double dw_curr = dw[i];
                
                // Compute bed slope
                double h_slope_s, h_slope_w;
                if (i > 0 && i < numCrossSections - 1)
                    h_slope_s = (h[i + 1, w] - h[i - 1, w]) / (2 * ds_curr);
                else if (i == 0)
                    h_slope_s = (h[1, w] - h[0, w]) / ds_curr;
                else
                    h_slope_s = (h[numCrossSections - 1, w] - h[numCrossSections - 2, w]) / ds_curr;
                
                if (w > 0 && w < widthResolution - 1)
                    h_slope_w = (h[i, w + 1] - h[i, w - 1]) / (2 * dw_curr);
                else if (w == 0)
                    h_slope_w = (h[i, 1] - h[i, 0]) / dw_curr;
                else
                    h_slope_w = (h[i, widthResolution - 1] - h[i, widthResolution - 2]) / dw_curr;
                
                double slope_mag = Math.Sqrt(h_slope_s * h_slope_s + h_slope_w * h_slope_w);
                
                // Slope factor: reduces transport on steep slopes
                double max_slope = 0.3;
                double slope_factor = 1.0 / (1.0 + slope_mag / max_slope);
                
                qs_magnitude[i, w] *= slope_factor;
            }
        }
        
        return qs_magnitude;
    }
    
    /// <summary>
    /// Computes sediment flux vector (direction based on velocity and bed slope).
    /// </summary>
    public (double[,] qs_s, double[,] qs_w) ComputeSedimentFluxVector(double[,] tau)
    {
        double[,] qs_magnitude = SedimentFlux(tau);
        double[,] qs_s = new double[numCrossSections, widthResolution];
        double[,] qs_w = new double[numCrossSections, widthResolution];
        
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                double u_mag = GetVelocityMagnitude(i, w);
                
                // Flow direction vector (normalized)
                double u_norm = u_mag > 1e-10 ? u[i, w] / u_mag : 0.0;
                double v_norm = u_mag > 1e-10 ? v[i, w] / u_mag : 0.0;
                
                // Bed slope direction vector (negative gradient = downhill)
                double ds_curr = ds[i];
                double dw_curr = dw[i];
                
                double h_slope_s_raw, h_slope_w_raw;
                
                // S-slope (along river)
                if (i > 0 && i < numCrossSections - 1)
                    h_slope_s_raw = (h[i + 1, w] - h[i - 1, w]) / (2 * ds_curr);
                else if (i == 0)
                    h_slope_s_raw = (h[1, w] - h[0, w]) / ds_curr;
                else
                    h_slope_s_raw = (h[numCrossSections - 1, w] - h[numCrossSections - 2, w]) / ds_curr;
                
                // W-slope (across river)
                if (w > 0 && w < widthResolution - 1)
                    h_slope_w_raw = (h[i, w + 1] - h[i, w - 1]) / (2 * dw_curr);
                else if (w == 0)
                    h_slope_w_raw = (h[i, 1] - h[i, 0]) / dw_curr;
                else
                    h_slope_w_raw = (h[i, widthResolution - 1] - h[i, widthResolution - 2]) / dw_curr;
                
                // Downhill direction = negative gradient
                double h_slope_s = -h_slope_s_raw;
                double h_slope_w = -h_slope_w_raw;
                
                double slope_mag = Math.Sqrt(h_slope_s * h_slope_s + h_slope_w * h_slope_w);
                double slope_s_norm = slope_mag > 1e-10 ? h_slope_s / slope_mag : 0.0;
                double slope_w_norm = slope_mag > 1e-10 ? h_slope_w / slope_mag : 0.0;
                
                // Weighted average of flow (70%) and slope (30%)
                double slopeFactor = 0.3;
                double flowFactor = 1.0 - slopeFactor;
                
                double u_combined = flowFactor * u_norm + slopeFactor * slope_s_norm;
                double v_combined = flowFactor * v_norm + slopeFactor * slope_w_norm;
                
                // Normalize combined direction
                double combined_mag = Math.Sqrt(u_combined * u_combined + v_combined * v_combined);
                double u_final_norm = combined_mag > 1e-10 ? u_combined / combined_mag : 0.0;
                double v_final_norm = combined_mag > 1e-10 ? v_combined / combined_mag : 0.0;
                
                // Compute final flux vector in river-local coordinates
                qs_s[i, w] = qs_magnitude[i, w] * u_final_norm;
                qs_w[i, w] = qs_magnitude[i, w] * v_final_norm;
            }
        }
        
        return (qs_s, qs_w);
    }
    
    /// <summary>
    /// Computes bank erosion rate based on shear stress.
    /// </summary>
    public double[,] ComputeBankErosion(double[,] tau)
    {
        double[,] bank_erosion = new double[numCrossSections, widthResolution];
        
        // Debug tracking
        double maxTau = 0.0;
        double maxTauExcess = 0.0;
        double maxBankErosion = 0.0;
        int bankCellCount = 0;
        int bankEdgeCount = 0;
        
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                // Compute excess shear stress for banks
                double tau_excess = Math.Max(tau[i, w] - bankCriticalShear, 0.0);
                
                // Erosion rate only applies to BANK cells
                if (cellType[i, w] != RiverCellType.FLUID)
                {
                    bankCellCount++;
                    
                    // Check if bank edge
                    bool isEdge = (w > 0 && cellType[i, w - 1] == RiverCellType.FLUID) ||
                                  (w < widthResolution - 1 && cellType[i, w + 1] == RiverCellType.FLUID);
                    if (isEdge) bankEdgeCount++;
                    
                    // Bank erosion rate scales the base transport coefficient
                    // Add scaling factor to convert from stress-based formula to realistic erosion rates (m/s)
                    // Typical bank erosion: 0.1-10 m/year = 3e-9 to 3e-7 m/s
                    // With high tau values (~1000 Pa), formula produces very large values, so scale down significantly
                    // Scaling factor converts stress-based units to m/s erosion rate
                    double erosionScalingFactor = 1e-6; // Conservative scaling for realistic erosion rates
                    bank_erosion[i, w] = bankErosionRate * transportCoefficient * Math.Pow(tau_excess, 1.5) * erosionScalingFactor;
                    
                    // Track max values
                    if (tau[i, w] > maxTau) maxTau = tau[i, w];
                    if (tau_excess > maxTauExcess) maxTauExcess = tau_excess;
                    if (bank_erosion[i, w] > maxBankErosion) maxBankErosion = bank_erosion[i, w];
                }
                else
                {
                    bank_erosion[i, w] = 0.0;
                }
            }
        }
        
        // Debug logging - also check adjacent fluid velocities for bank cells
        double maxAdjacentFluidVelocity = 0.0;
        double maxAdjacentFluidDepth = 0.0;
        if (debugLogCounter % DEBUG_LOG_INTERVAL == 0)
        {
            // Sample a few bank cells to check adjacent fluid properties
            int sampleCount = 0;
            for (int i = 0; i < numCrossSections && sampleCount < 10; i++)
            {
                for (int w = 0; w < widthResolution && sampleCount < 10; w++)
                {
                    if (cellType[i, w] == RiverCellType.BANK)
                    {
                        // Check adjacent fluid
                        if (w > 0 && cellType[i, w - 1] == RiverCellType.FLUID)
                        {
                            double vel = GetVelocityMagnitude(i, w - 1);
                            double depth = waterDepth[i, w - 1];
                            if (vel > maxAdjacentFluidVelocity) maxAdjacentFluidVelocity = vel;
                            if (depth > maxAdjacentFluidDepth) maxAdjacentFluidDepth = depth;
                            sampleCount++;
                        }
                        if (w < widthResolution - 1 && cellType[i, w + 1] == RiverCellType.FLUID)
                        {
                            double vel = GetVelocityMagnitude(i, w + 1);
                            double depth = waterDepth[i, w + 1];
                            if (vel > maxAdjacentFluidVelocity) maxAdjacentFluidVelocity = vel;
                            if (depth > maxAdjacentFluidDepth) maxAdjacentFluidDepth = depth;
                            sampleCount++;
                        }
                    }
                }
            }
        }
        
        // Debug logging
        if (debugLogCounter % DEBUG_LOG_INTERVAL == 0)
        {
            Debug.Log($"[BankErosion] Step {debugLogCounter}: BankCells={bankCellCount}, BankEdges={bankEdgeCount}, " +
                $"MaxTau={maxTau:F6}, MaxTauExcess={maxTauExcess:F6}, MaxErosionRate={maxBankErosion:F6}, " +
                $"MaxAdjacentFluidVel={maxAdjacentFluidVelocity:F6}, MaxAdjacentFluidDepth={maxAdjacentFluidDepth:F6}, " +
                $"BankErosionRate={bankErosionRate}, TransportCoeff={transportCoefficient}, BankCriticalShear={bankCriticalShear}");
        }
        
        return bank_erosion;
    }
    
    /// <summary>
    /// Solves Exner equation for bed evolution with zero-flux boundary conditions.
    /// dh/dt = -1/(1-porosity) * div(qs) + bank_erosion
    /// </summary>
    public (double[,] dh_dt, double[,] h_new) ExnerEquation(double[,] qs_s, double[,] qs_w, float dt)
    {
        double[,] h_new = (double[,])h.Clone();
        double porosityFactor = 1.0 / (1.0 - porosity);
        double max_change_rate = 0.001; // Reduced from 0.02 for more realistic, gradual bed evolution
        
        // Compute shear stress and bank erosion
        double[,] tau = ComputeShearStress();
        double[,] bank_erosion_rate = ComputeBankErosion(tau);
        
        // Apply flux boundary conditions (zero flux at boundaries)
        double[,] qs_s_bc = (double[,])qs_s.Clone();
        double[,] qs_w_bc = (double[,])qs_w.Clone();
        
        // Set flux to zero at boundaries
        for (int w = 0; w < widthResolution; w++)
        {
            qs_s_bc[0, w] = 0.0;  // Upstream boundary
            qs_s_bc[numCrossSections - 1, w] = 0.0;  // Downstream boundary
        }
        for (int i = 0; i < numCrossSections; i++)
        {
            qs_w_bc[i, 0] = 0.0;  // Left bank
            qs_w_bc[i, widthResolution - 1] = 0.0;  // Right bank
        }
        
        // Compute divergence
        double[,] div_qs = new double[numCrossSections, widthResolution];
        
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                double dqs_s_ds, dqs_w_dw;
                double ds_curr = ds[i];
                double dw_curr = dw[i];
                
                // S-divergence (along river)
                if (i > 0 && i < numCrossSections - 1)
                    dqs_s_ds = (qs_s_bc[i + 1, w] - qs_s_bc[i - 1, w]) / (2 * ds_curr);
                else if (i == 0)
                    dqs_s_ds = (qs_s_bc[1, w] - qs_s_bc[0, w]) / ds_curr;
                else
                    dqs_s_ds = (qs_s_bc[numCrossSections - 1, w] - qs_s_bc[numCrossSections - 2, w]) / ds_curr;
                
                // W-divergence (across river)
                if (w > 0 && w < widthResolution - 1)
                    dqs_w_dw = (qs_w_bc[i, w + 1] - qs_w_bc[i, w - 1]) / (2 * dw_curr);
                else if (w == 0)
                    dqs_w_dw = (qs_w_bc[i, 1] - qs_w_bc[i, 0]) / dw_curr;
                else
                    dqs_w_dw = (qs_w_bc[i, widthResolution - 1] - qs_w_bc[i, widthResolution - 2]) / dw_curr;
                
                div_qs[i, w] = dqs_s_ds + dqs_w_dw;
                
                // Explicitly set divergence to zero for BANK cells
                // Banks should not be affected by sediment transport (only horizontal bank migration, not vertical erosion)
                if (cellType[i, w] != RiverCellType.FLUID)
                {
                    div_qs[i, w] = 0.0;
                }
            }
        }
        
        // Corners: zero flux
        div_qs[0, 0] = 0.0;
        div_qs[0, widthResolution - 1] = 0.0;
        div_qs[numCrossSections - 1, 0] = 0.0;
        div_qs[numCrossSections - 1, widthResolution - 1] = 0.0;
        
        // Exner equation: dh/dt = -1/(1-porosity) * div(qs) + bank_erosion
        double[,] dh_dt = new double[numCrossSections, widthResolution];
        
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                // Only apply Exner equation to FLUID cells
                if (cellType[i, w] == RiverCellType.FLUID)
                {
                    dh_dt[i, w] = -porosityFactor * div_qs[i, w];
                }
                else
                {
                    // BANK cells: prevent vertical erosion (banks should erode horizontally, not vertically)
                    // Keep bank elevation stable or maintain minimum height relative to adjacent fluid
                    dh_dt[i, w] = 0.0;
                }
                
                // Update bed elevation and apply stability constraint
                double h_change = dh_dt[i, w] * dt;
                
                // For very large time steps (days/weeks), disable max_change_rate constraint
                // The maxDeviation cap below provides sufficient safety for long-term evolution
                // For small/medium steps, keep the constraint to prevent numerical instability
                if (dt > 86400.0) // > 1 day: disable constraint entirely for long-term evolution
                {
                    // No constraint - allow natural evolution, maxDeviation cap below will prevent unbounded growth
                }
                else if (dt > 1.0) // > 1 second but < 1 day: scale up significantly
                {
                    // For large dt, scale max_change_rate to allow meaningful evolution
                    double effective_max_change_rate = max_change_rate * 100.0; // Very permissive (effectively disabled)
                    double max_change = effective_max_change_rate * dt;
                    h_change = Math.Max(-max_change, Math.Min(max_change, h_change));
                }
                else if (dt > 0.1) // Medium dt: moderate scaling
                {
                    double effective_max_change_rate = max_change_rate * 3.0;
                    double max_change = effective_max_change_rate * dt;
                    h_change = Math.Max(-max_change, Math.Min(max_change, h_change));
                }
                else // Small dt: apply normal constraint
                {
                    double max_change = max_change_rate * dt;
                    h_change = Math.Max(-max_change, Math.Min(max_change, h_change));
                }
                
                h_new[i, w] = h[i, w] + h_change;
                
                // Final safety clip - limit total change from initial elevation to prevent unbounded growth
                // Since h represents changes from initial state (starts at 0), limit h_new to ±maxDeviation
                // For large time steps, scale maxDeviation to allow meaningful long-term evolution
                double maxDeviation = 2.0; // Base maximum change from initial elevation (in solver units)
                if (dt > 1.0)
                {
                    // For large time steps (weeks/months), allow much larger deviations
                    // Scale with time step: allow up to 10 meters per week of simulation
                    maxDeviation = 10.0 * (dt / 604800.0); // Scale by weeks (604800 seconds = 1 week)
                    maxDeviation = Math.Max(maxDeviation, 10.0); // Minimum 10 meters for large steps
                }
                else if (dt > 0.1)
                {
                    // For medium steps, allow moderate scaling
                    maxDeviation = 5.0;
                }
                
                if (Math.Abs(h_new[i, w]) > maxDeviation)
                {
                    // Clamp h_new to ±maxDeviation (since h is the change from initial)
                    h_new[i, w] = Math.Sign(h_new[i, w]) * maxDeviation;
                }
            }
        }
        
        // Apply boundary conditions: maintain bank elevations and prevent edge falling
        ApplyBedElevationBoundaryConditions(h_new);
        
        // Accumulate bank erosion for migration tracking
        // Only accumulate at bank edge cells (BANK cells adjacent to FLUID cells)
        double maxCumulativeErosion = 0.0;
        int accumulatingEdges = 0;
        double totalErosionThisStep = 0.0;
        
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                if (cellType[i, w] == RiverCellType.BANK)
                {
                    // Check if this is a bank edge cell (adjacent to FLUID)
                    bool isBankEdge = false;
                    
                    // Check left neighbor
                    if (w > 0 && cellType[i, w - 1] == RiverCellType.FLUID)
                        isBankEdge = true;
                    
                    // Check right neighbor
                    if (w < widthResolution - 1 && cellType[i, w + 1] == RiverCellType.FLUID)
                        isBankEdge = true;
                    
                    // Only accumulate erosion at bank edges (where water contacts bank)
                    if (isBankEdge)
                    {
                        accumulatingEdges++;
                        double erosionIncrement = bank_erosion_rate[i, w] * dt;
                        
                        // Cap cumulative erosion to prevent overflow and ensure reasonable migration rates
                        // Reset when it exceeds threshold by a large margin (e.g., 10x threshold)
                        double maxCumulativeBeforeReset = bankMigrationThreshold * 10.0;
                        if (cumulativeBankErosion[i, w] >= maxCumulativeBeforeReset)
                        {
                            // Reset to threshold value to maintain migration pressure without overflow
                            cumulativeBankErosion[i, w] = bankMigrationThreshold * 1.5;
                        }
                        else
                        {
                            cumulativeBankErosion[i, w] += erosionIncrement;
                        }
                        
                        totalErosionThisStep += erosionIncrement;
                        
                        if (cumulativeBankErosion[i, w] > maxCumulativeErosion)
                            maxCumulativeErosion = cumulativeBankErosion[i, w];
                    }
                }
            }
        }
        
        // Debug logging
        if (debugLogCounter % DEBUG_LOG_INTERVAL == 0)
        {
            Debug.Log($"[BankAccumulation] Step {debugLogCounter}: dt={dt}, AccumulatingEdges={accumulatingEdges}, " +
                $"TotalErosionThisStep={totalErosionThisStep:F6}, MaxCumulativeErosion={maxCumulativeErosion:F6}, " +
                $"MigrationThreshold={bankMigrationThreshold}");
        }
        
        debugLogCounter++;
        
        // NOTE: Bank migration now works by moving mesh vertices outward/inward
        // instead of converting FLUID cells to BANK. The grid always keeps w=0 and w=widthResolution-1 as banks.
        // ProcessBankMigration() is disabled - mesh expansion/contraction is handled in UpdateMeshVerticesForBankMigration()
        
        // Apply spatial smoothing to reduce numerical noise
        // Conservative smoothing to prevent mesh peaking while preserving physical behavior
        h_new = ApplySpatialSmoothing(h_new, smoothingFactor: 0.05);
        
        // Store current erosion rate for visualization
        current_dh_dt = dh_dt;
        
        return (dh_dt, h_new);
    }
    
    /// <summary>
    /// Updates mesh vertices to widen/narrow cross-sections based on bank erosion.
    /// When banks erode, the mesh expands outward. When banks deposit, the mesh contracts inward.
    /// The grid always keeps w=0 and w=widthResolution-1 as banks, but their positions move.
    /// </summary>
    /// <param name="vertices">The mesh vertices array to update (modified in place)</param>
    /// <param name="dt">Time step for erosion rate</param>
    public void UpdateMeshVerticesForBankMigration(Vector3[] vertices, float dt)
    {
        if (vertices == null || vertices.Length != numCrossSections * widthResolution)
        {
            Debug.LogWarning("[RiverMeshPhysicsSolver] Cannot update mesh vertices - invalid vertex array");
            return;
        }
        
        // Compute bank erosion rate for this step
        double[,] tau = ComputeShearStress();
        double[,] bank_erosion_rate = ComputeBankErosion(tau);
        
        // First pass: Calculate desired movements for all cross-sections
        float[] leftBankMovements = new float[numCrossSections];
        float[] rightBankMovements = new float[numCrossSections];
        
        // Process each cross-section to calculate movements
        for (int i = 0; i < numCrossSections; i++)
        {
            // Get current bank positions (left and right edges)
            int leftBankIdx = i * widthResolution;
            int rightBankIdx = i * widthResolution + (widthResolution - 1);
            
            Vector3 leftBankPos = vertices[leftBankIdx];
            Vector3 rightBankPos = vertices[rightBankIdx];
            
            // Calculate current cross-section center and direction
            Vector3 center = (leftBankPos + rightBankPos) * 0.5f;
            Vector3 bankDirection = (rightBankPos - leftBankPos).normalized;
            float currentWidth = Vector3.Distance(leftBankPos, rightBankPos);
            
            // Compute net bank movement for this cross-section
            // Use cumulative erosion for more stable migration
            // Positive erosion rate = bank erodes = move outward (widen)
            // Negative erosion rate = bank deposits = move inward (narrow)
            
            // Get cumulative erosion at bank edges (more stable than instantaneous rate)
            double leftCumulativeErosion = cumulativeBankErosion != null ? cumulativeBankErosion[i, 0] : 0.0;
            double rightCumulativeErosion = cumulativeBankErosion != null ? cumulativeBankErosion[i, widthResolution - 1] : 0.0;
            
            // Also use instantaneous erosion rate for immediate response
            double leftBankErosionRate = bank_erosion_rate[i, 0] * dt;
            double rightBankErosionRate = bank_erosion_rate[i, widthResolution - 1] * dt;
            
            // Normalize cumulative erosion to threshold scale to prevent overflow issues
            // Use cumulative as a multiplier on the threshold, but cap it
            double leftCumulativeFactor = Math.Min(leftCumulativeErosion / bankMigrationThreshold, 10.0);
            double rightCumulativeFactor = Math.Min(rightCumulativeErosion / bankMigrationThreshold, 10.0);
            
            // Combine cumulative factor with instantaneous erosion
            // Use cumulative as multiplier, instantaneous as base
            // Reduced scaling from 100.0 to 1.0 for more reasonable movement rates
            // The erosion rates are already in m/s, so we just need to convert to Unity units
            double leftTotalErosion = leftBankErosionRate * (1.0 + leftCumulativeFactor * 0.1); // Much smaller multiplier
            double rightTotalErosion = rightBankErosionRate * (1.0 + rightCumulativeFactor * 0.1);
            
            // FAVOR MEANDERING OVER WIDENING:
            // If both banks are eroding outward (widening), reduce the movement significantly
            // If one bank erodes more than the other (meandering), allow normal movement
            bool bothBanksEroding = leftTotalErosion > 0 && rightTotalErosion > 0;
            bool bothBanksDepositing = leftTotalErosion < 0 && rightTotalErosion < 0;
            
            // Calculate asymmetry factor (difference between left and right erosion)
            double erosionAsymmetry = Math.Abs(leftTotalErosion - rightTotalErosion);
            double totalErosionMagnitude = Math.Abs(leftTotalErosion) + Math.Abs(rightTotalErosion);
            double asymmetryRatio = totalErosionMagnitude > 0 ? erosionAsymmetry / totalErosionMagnitude : 0.0;
            
            // If both banks are eroding outward (symmetric widening), heavily penalize it
            if (bothBanksEroding && asymmetryRatio < 0.3) // Less than 30% asymmetry = symmetric widening
            {
                // Reduce outward movement by 90% to favor meandering
                leftTotalErosion *= 0.1;
                rightTotalErosion *= 0.1;
            }
            // If one bank erodes much more than the other (asymmetric = meandering), allow it
            else if (asymmetryRatio > 0.5) // More than 50% asymmetry = strong meandering
            {
                // Boost meandering movement slightly (10% increase)
                if (leftTotalErosion > rightTotalErosion)
                {
                    leftTotalErosion *= 1.1;
                    rightTotalErosion *= 0.9; // Opposite bank moves less
                }
                else
                {
                    rightTotalErosion *= 1.1;
                    leftTotalErosion *= 0.9;
                }
            }
            
            // Convert erosion from meters to Unity units
            double metersToUnityScale = 1.0 / unityToMetersScale;
            float leftBankMovement = (float)(leftTotalErosion * metersToUnityScale);
            float rightBankMovement = (float)(rightTotalErosion * metersToUnityScale);
            
            // Apply movement limits to prevent excessive expansion/contraction
            // Reduced to 1% per step for outward movement, 2% for inward (meandering)
            float maxOutwardMovementPerStep = currentWidth * 0.01f; // Max 1% width increase per step (widening)
            float maxInwardMovementPerStep = currentWidth * 0.02f; // Max 2% width decrease per step (narrowing/meandering)
            
            // Clamp outward movement more strictly than inward
            if (leftBankMovement > 0)
                leftBankMovement = Mathf.Clamp(leftBankMovement, 0, maxOutwardMovementPerStep);
            else
                leftBankMovement = Mathf.Clamp(leftBankMovement, -maxInwardMovementPerStep, 0);
                
            if (rightBankMovement > 0)
                rightBankMovement = Mathf.Clamp(rightBankMovement, 0, maxOutwardMovementPerStep);
            else
                rightBankMovement = Mathf.Clamp(rightBankMovement, -maxInwardMovementPerStep, 0);
            
            // Store movements for smoothing (don't apply yet)
            leftBankMovements[i] = leftBankMovement;
            rightBankMovements[i] = rightBankMovement;
        }
        
        // Second pass: Apply spatial smoothing to movements to prevent noisy width changes
        // Smooth movements between adjacent cross-sections to prevent abrupt width jumps
        // Use extended neighborhood for stronger smoothing
        float[] smoothedLeftMovements = new float[numCrossSections];
        float[] smoothedRightMovements = new float[numCrossSections];
        
        for (int i = 0; i < numCrossSections; i++)
        {
            // Extended smoothing: include current + 2 neighbors on each side (5-point average)
            // Much stronger smoothing: 40% current, 60% neighbors
            float centerWeight = 0.4f; // 40% current
            float neighborWeight = 0.15f; // 15% per immediate neighbor
            float extendedNeighborWeight = 0.075f; // 7.5% per extended neighbor
            
            float leftSmoothed = leftBankMovements[i] * centerWeight;
            float rightSmoothed = rightBankMovements[i] * centerWeight;
            
            // Add contribution from immediate upstream neighbor (i-1)
            if (i > 0)
            {
                leftSmoothed += leftBankMovements[i - 1] * neighborWeight;
                rightSmoothed += rightBankMovements[i - 1] * neighborWeight;
            }
            else
            {
                leftSmoothed += leftBankMovements[i] * neighborWeight;
                rightSmoothed += rightBankMovements[i] * neighborWeight;
            }
            
            // Add contribution from immediate downstream neighbor (i+1)
            if (i < numCrossSections - 1)
            {
                leftSmoothed += leftBankMovements[i + 1] * neighborWeight;
                rightSmoothed += rightBankMovements[i + 1] * neighborWeight;
            }
            else
            {
                leftSmoothed += leftBankMovements[i] * neighborWeight;
                rightSmoothed += rightBankMovements[i] * neighborWeight;
            }
            
            // Add contribution from extended upstream neighbor (i-2)
            if (i > 1)
            {
                leftSmoothed += leftBankMovements[i - 2] * extendedNeighborWeight;
                rightSmoothed += rightBankMovements[i - 2] * extendedNeighborWeight;
            }
            else if (i > 0)
            {
                // Use i-1 if i-2 doesn't exist
                leftSmoothed += leftBankMovements[i - 1] * extendedNeighborWeight;
                rightSmoothed += rightBankMovements[i - 1] * extendedNeighborWeight;
            }
            else
            {
                leftSmoothed += leftBankMovements[i] * extendedNeighborWeight;
                rightSmoothed += rightBankMovements[i] * extendedNeighborWeight;
            }
            
            // Add contribution from extended downstream neighbor (i+2)
            if (i < numCrossSections - 2)
            {
                leftSmoothed += leftBankMovements[i + 2] * extendedNeighborWeight;
                rightSmoothed += rightBankMovements[i + 2] * extendedNeighborWeight;
            }
            else if (i < numCrossSections - 1)
            {
                // Use i+1 if i+2 doesn't exist
                leftSmoothed += leftBankMovements[i + 1] * extendedNeighborWeight;
                rightSmoothed += rightBankMovements[i + 1] * extendedNeighborWeight;
            }
            else
            {
                leftSmoothed += leftBankMovements[i] * extendedNeighborWeight;
                rightSmoothed += rightBankMovements[i] * extendedNeighborWeight;
            }
            
            smoothedLeftMovements[i] = leftSmoothed;
            smoothedRightMovements[i] = rightSmoothed;
        }
        
        // Additional smoothing pass: apply a second round of smoothing for even smoother transitions
        float[] finalLeftMovements = new float[numCrossSections];
        float[] finalRightMovements = new float[numCrossSections];
        
        for (int i = 0; i < numCrossSections; i++)
        {
            // Second smoothing pass: 3-point average (current + immediate neighbors)
            float secondPassWeight = 0.5f; // 50% current
            float secondPassNeighborWeight = 0.25f; // 25% per neighbor
            
            float leftFinal = smoothedLeftMovements[i] * secondPassWeight;
            float rightFinal = smoothedRightMovements[i] * secondPassWeight;
            
            if (i > 0)
            {
                leftFinal += smoothedLeftMovements[i - 1] * secondPassNeighborWeight;
                rightFinal += smoothedRightMovements[i - 1] * secondPassNeighborWeight;
            }
            else
            {
                leftFinal += smoothedLeftMovements[i] * secondPassNeighborWeight;
                rightFinal += smoothedRightMovements[i] * secondPassNeighborWeight;
            }
            
            if (i < numCrossSections - 1)
            {
                leftFinal += smoothedLeftMovements[i + 1] * secondPassNeighborWeight;
                rightFinal += smoothedRightMovements[i + 1] * secondPassNeighborWeight;
            }
            else
            {
                leftFinal += smoothedLeftMovements[i] * secondPassNeighborWeight;
                rightFinal += smoothedRightMovements[i] * secondPassNeighborWeight;
            }
            
            finalLeftMovements[i] = leftFinal;
            finalRightMovements[i] = rightFinal;
        }
        
        // Use the final smoothed movements
        smoothedLeftMovements = finalLeftMovements;
        smoothedRightMovements = finalRightMovements;
        
        // Additional smoothing: Limit maximum width change between adjacent cross-sections
        // This prevents sudden jumps in width
        for (int i = 1; i < numCrossSections; i++)
        {
            // Get current widths
            int prevLeftIdx = (i - 1) * widthResolution;
            int prevRightIdx = (i - 1) * widthResolution + (widthResolution - 1);
            int currLeftIdx = i * widthResolution;
            int currRightIdx = i * widthResolution + (widthResolution - 1);
            
            float prevWidth = Vector3.Distance(vertices[prevLeftIdx], vertices[prevRightIdx]);
            float currWidth = Vector3.Distance(vertices[currLeftIdx], vertices[currRightIdx]);
            
            // Calculate what the new width would be after movement
            Vector3 prevBankDir = (vertices[prevRightIdx] - vertices[prevLeftIdx]).normalized;
            Vector3 currBankDir = (vertices[currRightIdx] - vertices[currLeftIdx]).normalized;
            
            // Estimate new width after movement (approximate)
            float prevNewWidth = prevWidth + (smoothedLeftMovements[i - 1] + smoothedRightMovements[i - 1]);
            float currNewWidth = currWidth + (smoothedLeftMovements[i] + smoothedRightMovements[i]);
            
            // Limit width change to max 2% difference between adjacent cross-sections (stricter limit)
            float maxWidthChange = prevWidth * 0.02f;
            float widthDiff = Mathf.Abs(currNewWidth - prevNewWidth);
            
            if (widthDiff > maxWidthChange)
            {
                // Scale down movements to limit width change
                float scaleFactor = maxWidthChange / widthDiff;
                smoothedLeftMovements[i] *= scaleFactor;
                smoothedRightMovements[i] *= scaleFactor;
            }
        }
        
        // Third pass: Apply smoothed movements to vertices
        for (int i = 0; i < numCrossSections; i++)
        {
            // Get current bank positions
            int leftBankIdx = i * widthResolution;
            int rightBankIdx = i * widthResolution + (widthResolution - 1);
            
            Vector3 leftBankPos = vertices[leftBankIdx];
            Vector3 rightBankPos = vertices[rightBankIdx];
            
            // Calculate current cross-section center and direction
            Vector3 bankDirection = (rightBankPos - leftBankPos).normalized;
            
            // Use smoothed movements
            float leftBankMovement = smoothedLeftMovements[i];
            float rightBankMovement = smoothedRightMovements[i];
            
            // Move banks outward (negative direction for left, positive for right)
            Vector3 leftBankNewPos = leftBankPos - bankDirection * leftBankMovement;
            Vector3 rightBankNewPos = rightBankPos + bankDirection * rightBankMovement;
            
            // Update bank vertex positions
            vertices[leftBankIdx] = leftBankNewPos;
            vertices[rightBankIdx] = rightBankNewPos;
            
            // Update all intermediate vertices proportionally
            // Keep the same relative spacing but adjust to new width
            float newWidth = Vector3.Distance(leftBankNewPos, rightBankNewPos);
            
            for (int w = 1; w < widthResolution - 1; w++)
            {
                // Normalized position across width (0 = left bank, 1 = right bank)
                float normalizedPos = (float)w / (float)(widthResolution - 1);
                
                // Interpolate new position between new bank positions
                Vector3 newPos = Vector3.Lerp(leftBankNewPos, rightBankNewPos, normalizedPos);
                
                // Preserve the elevation (Y coordinate) from current position
                int vertexIdx = i * widthResolution + w;
                if (vertexIdx < vertices.Length)
                {
                    float currentElevation = vertices[vertexIdx].y;
                    newPos.y = currentElevation; // Keep elevation, only change horizontal position
                    vertices[vertexIdx] = newPos;
                }
            }
            
            // Also preserve elevation for bank vertices (only move horizontally)
            vertices[leftBankIdx].y = leftBankPos.y;
            vertices[rightBankIdx].y = rightBankPos.y;
        }
        
        // Fourth pass: Smooth bank vertex positions along river length for continuous lines
        // This creates smooth, continuous bank lines instead of discrete cross-sections
        Vector3[] leftBankPositions = new Vector3[numCrossSections];
        Vector3[] rightBankPositions = new Vector3[numCrossSections];
        
        // Collect bank positions
        for (int i = 0; i < numCrossSections; i++)
        {
            leftBankPositions[i] = vertices[i * widthResolution];
            rightBankPositions[i] = vertices[i * widthResolution + (widthResolution - 1)];
        }
        
        // Smooth left bank positions along river length
        Vector3[] smoothedLeftBanks = SmoothBankPositions(leftBankPositions);
        // Smooth right bank positions along river length
        Vector3[] smoothedRightBanks = SmoothBankPositions(rightBankPositions);
        
        // Update bank vertices with smoothed positions
        for (int i = 0; i < numCrossSections; i++)
        {
            int leftBankIdx = i * widthResolution;
            int rightBankIdx = i * widthResolution + (widthResolution - 1);
            
            // Preserve elevation from original position, only smooth horizontal (X, Z)
            Vector3 smoothedLeft = smoothedLeftBanks[i];
            Vector3 smoothedRight = smoothedRightBanks[i];
            smoothedLeft.y = leftBankPositions[i].y; // Preserve elevation
            smoothedRight.y = rightBankPositions[i].y; // Preserve elevation
            
            vertices[leftBankIdx] = smoothedLeft;
            vertices[rightBankIdx] = smoothedRight;
        }
        
        // Recalculate intermediate vertices proportionally after bank smoothing
        for (int i = 0; i < numCrossSections; i++)
        {
            int leftBankIdx = i * widthResolution;
            int rightBankIdx = i * widthResolution + (widthResolution - 1);
            
            Vector3 leftBankPos = vertices[leftBankIdx];
            Vector3 rightBankPos = vertices[rightBankIdx];
            
            // Update all intermediate vertices proportionally
            for (int w = 1; w < widthResolution - 1; w++)
            {
                // Normalized position across width (0 = left bank, 1 = right bank)
                float normalizedPos = (float)w / (float)(widthResolution - 1);
                
                // Interpolate new position between smoothed bank positions
                Vector3 newPos = Vector3.Lerp(leftBankPos, rightBankPos, normalizedPos);
                
                // Preserve the elevation (Y coordinate) from current position
                int vertexIdx = i * widthResolution + w;
                if (vertexIdx < vertices.Length)
                {
                    float currentElevation = vertices[vertexIdx].y;
                    newPos.y = currentElevation; // Keep elevation, only change horizontal position
                    vertices[vertexIdx] = newPos;
                }
            }
        }
        
        // Update spacing and coordinate system after vertex changes
        CalculateSpacing(vertices);
        _coordinateSystem = new RiverCoordinateSystem(vertices, numCrossSections, widthResolution);
        _coordinateSystem.CalculateFrames();
        
        // Debug logging for migration
        if (debugLogCounter % DEBUG_LOG_INTERVAL == 0)
        {
            double totalMovement = 0.0;
            double maxMovement = 0.0;
            int movedSections = 0;
            for (int i = 0; i < numCrossSections; i++)
            {
                int leftIdx = i * widthResolution;
                int rightIdx = i * widthResolution + (widthResolution - 1);
                if (leftIdx < vertices.Length && rightIdx < vertices.Length)
                {
                    float width = Vector3.Distance(vertices[leftIdx], vertices[rightIdx]);
                    totalMovement += width;
                    if (width > maxMovement) maxMovement = width;
                    movedSections++;
                }
            }
            double avgWidth = movedSections > 0 ? totalMovement / movedSections : 0.0;
            
            // Sample a few cross-sections to show actual movement
            double sampleLeftErosion = cumulativeBankErosion != null && numCrossSections > 0 ? cumulativeBankErosion[0, 0] : 0.0;
            double sampleRightErosion = cumulativeBankErosion != null && numCrossSections > 0 ? cumulativeBankErosion[0, widthResolution - 1] : 0.0;
            
            Debug.Log($"[BankMigration] Mesh vertices updated. Avg width: {avgWidth:F3}, Max width: {maxMovement:F3}, " +
                     $"Sample cumulative erosion: L={sampleLeftErosion:E3}, R={sampleRightErosion:E3}, Threshold={bankMigrationThreshold}");
        }
    }
    
    /// <summary>
    /// Smooths bank vertex positions along the river length (s-direction) to create continuous smooth bank lines.
    /// Uses a 5-point smoothing kernel with weighted neighbors.
    /// </summary>
    /// <param name="bankPositions">Array of bank vertex positions along the river</param>
    /// <returns>Array of smoothed bank positions</returns>
    private Vector3[] SmoothBankPositions(Vector3[] bankPositions)
    {
        if (bankPositions == null || bankPositions.Length == 0)
            return bankPositions;
        
        Vector3[] smoothed = new Vector3[bankPositions.Length];
        
        // 5-point smoothing kernel: 40% current, 20% each immediate neighbor (i±1), 10% each extended neighbor (i±2)
        float centerWeight = 0.4f;
        float immediateNeighborWeight = 0.2f;
        float extendedNeighborWeight = 0.1f;
        
        for (int i = 0; i < bankPositions.Length; i++)
        {
            Vector3 smoothedPos = bankPositions[i] * centerWeight;
            
            // Add contribution from immediate upstream neighbor (i-1)
            if (i > 0)
            {
                smoothedPos += bankPositions[i - 1] * immediateNeighborWeight;
            }
            else
            {
                // At boundary, use current value
                smoothedPos += bankPositions[i] * immediateNeighborWeight;
            }
            
            // Add contribution from immediate downstream neighbor (i+1)
            if (i < bankPositions.Length - 1)
            {
                smoothedPos += bankPositions[i + 1] * immediateNeighborWeight;
            }
            else
            {
                // At boundary, use current value
                smoothedPos += bankPositions[i] * immediateNeighborWeight;
            }
            
            // Add contribution from extended upstream neighbor (i-2)
            if (i > 1)
            {
                smoothedPos += bankPositions[i - 2] * extendedNeighborWeight;
            }
            else if (i > 0)
            {
                // Use i-1 if i-2 doesn't exist
                smoothedPos += bankPositions[i - 1] * extendedNeighborWeight;
            }
            else
            {
                // At boundary, use current value
                smoothedPos += bankPositions[i] * extendedNeighborWeight;
            }
            
            // Add contribution from extended downstream neighbor (i+2)
            if (i < bankPositions.Length - 2)
            {
                smoothedPos += bankPositions[i + 2] * extendedNeighborWeight;
            }
            else if (i < bankPositions.Length - 1)
            {
                // Use i+1 if i+2 doesn't exist
                smoothedPos += bankPositions[i + 1] * extendedNeighborWeight;
            }
            else
            {
                // At boundary, use current value
                smoothedPos += bankPositions[i] * extendedNeighborWeight;
            }
            
            smoothed[i] = smoothedPos;
        }
        
        return smoothed;
    }
    
    /// <summary>
    /// Processes bank migration by converting FLUID cells to BANK cells when cumulative erosion exceeds threshold.
    /// This simulates banks eroding and the river channel narrowing over time.
    /// The bank edge moves inward as adjacent FLUID cells are converted to BANK.
    /// </summary>
    /// <summary>
    /// Processes bank migration by converting FLUID cells to BANK cells when cumulative erosion exceeds threshold.
    /// This simulates banks eroding and the river channel narrowing over time.
    /// The bank edge moves inward as adjacent FLUID cells are converted to BANK.
    /// </summary>
    private void ProcessBankMigration()
    {
        // Process each cross-section
        const int minChannelWidth = 2; // Minimum FLUID cells to maintain (prevent complete closure)
        int totalMigrationsThisStep = 0;
        int preventedMigrations = 0;
        int criticalWidthSections = 0;
        
        for (int i = 0; i < numCrossSections; i++)
        {
            // Find the left bank edge: find the rightmost BANK cell (innermost from left)
            int leftBankEdge = -1;
            for (int w = 0; w < widthResolution - 1; w++)
            {
                if (cellType[i, w] == RiverCellType.BANK && cellType[i, w + 1] == RiverCellType.FLUID)
                {
                    leftBankEdge = w;
                    break;
                }
            }
            
            // If no bank edge found at boundary, check if w=0 is bank
            if (leftBankEdge == -1 && cellType[i, 0] == RiverCellType.BANK)
            {
                leftBankEdge = 0;
            }
            
            // Find the right bank edge: find the leftmost BANK cell (innermost from right)
            int rightBankEdge = -1;
            for (int w = widthResolution - 1; w > 0; w--)
            {
                if (cellType[i, w] == RiverCellType.BANK && cellType[i, w - 1] == RiverCellType.FLUID)
                {
                    rightBankEdge = w;
                    break;
                }
            }
            
            // If no bank edge found at boundary, check if rightmost is bank
            if (rightBankEdge == -1 && cellType[i, widthResolution - 1] == RiverCellType.BANK)
            {
                rightBankEdge = widthResolution - 1;
            }
            
            // Calculate current channel width (number of FLUID cells)
            int channelWidth = -1;
            if (leftBankEdge >= 0 && rightBankEdge >= 0 && leftBankEdge < rightBankEdge)
            {
                channelWidth = rightBankEdge - leftBankEdge - 1;
            }
            else if (leftBankEdge == -1 && rightBankEdge == -1)
            {
                // No bank edges found - check if all cells are FLUID or all are BANK
                bool hasFluid = false;
                for (int w = 0; w < widthResolution; w++)
                {
                    if (cellType[i, w] == RiverCellType.FLUID)
                    {
                        hasFluid = true;
                        break;
                    }
                }
                channelWidth = hasFluid ? widthResolution : 0; // All FLUID or all BANK
            }
            
            // Check if channel width is critical
            if (channelWidth >= 0 && channelWidth <= minChannelWidth)
            {
                criticalWidthSections++;
                if (debugLogCounter % DEBUG_LOG_INTERVAL == 0)
                {
                    Debug.LogWarning($"[BankMigration] CrossSection {i}: Channel width is critical ({channelWidth} FLUID cells). Migration may be prevented.");
                }
            }
            
            // Check if we should migrate the left bank inward
            if (leftBankEdge >= 0 && leftBankEdge < widthResolution - 1)
            {
                double cumulativeErosion = cumulativeBankErosion[i, leftBankEdge];
                
                // Debug logging for threshold check
                if (debugLogCounter % DEBUG_LOG_INTERVAL == 0 && cumulativeErosion > 0)
                {
                    Debug.Log($"[BankMigration] CrossSection {i}, LeftEdge w={leftBankEdge}: " +
                        $"CumulativeErosion={cumulativeErosion:F6}, Threshold={bankMigrationThreshold}, " +
                        $"Ratio={cumulativeErosion / bankMigrationThreshold:F2}x, ChannelWidth={channelWidth}");
                }
                
                // Check cumulative erosion at the bank edge
                if (cumulativeErosion >= bankMigrationThreshold)
                {
                    // Find the adjacent FLUID cell (moving inward from left)
                    int nextFluidW = leftBankEdge + 1;
                    if (nextFluidW < widthResolution && cellType[i, nextFluidW] == RiverCellType.FLUID)
                    {
                        // Channel closure safeguard: prevent migration if it would close the channel
                        int newChannelWidth = (rightBankEdge >= 0 && rightBankEdge > nextFluidW) ? 
                            rightBankEdge - nextFluidW - 1 : -1;
                        
                        if (newChannelWidth >= 0 && newChannelWidth < minChannelWidth)
                        {
                            // Migration would close channel - prevent it
                            preventedMigrations++;
                            if (debugLogCounter % DEBUG_LOG_INTERVAL == 0)
                            {
                                Debug.LogWarning($"[BankMigration] CrossSection {i}: Prevented left bank migration - would reduce channel to {newChannelWidth} FLUID cells (minimum: {minChannelWidth})");
                            }
                        }
                        else
                        {
                            // Safe to migrate
                            // Convert FLUID to BANK: bank migrates inward
                            cellType[i, nextFluidW] = RiverCellType.BANK;
                            waterDepth[i, nextFluidW] = 0.0;
                            u[i, nextFluidW] = 0.0;
                            v[i, nextFluidW] = 0.0;
                            
                            // Transfer cumulative erosion to new bank position (carry over excess)
                            // Cap excess erosion to prevent immediate re-migration cascade
                            double excessErosion = cumulativeErosion - bankMigrationThreshold;
                            double maxExcessErosion = bankMigrationThreshold * 0.5; // Allow up to 50% of threshold
                            cumulativeBankErosion[i, nextFluidW] = Math.Min(excessErosion, maxExcessErosion);
                            cumulativeBankErosion[i, leftBankEdge] = 0.0;
                            totalMigrationsThisStep++;
                        }
                    }
                }
            }
            
            // Check if we should migrate the right bank inward
            if (rightBankEdge > 0 && rightBankEdge < widthResolution)
            {
                double cumulativeErosion = cumulativeBankErosion[i, rightBankEdge];
                
                // Debug logging for threshold check
                if (debugLogCounter % DEBUG_LOG_INTERVAL == 0 && cumulativeErosion > 0)
                {
                    Debug.Log($"[BankMigration] CrossSection {i}, RightEdge w={rightBankEdge}: " +
                        $"CumulativeErosion={cumulativeErosion:F6}, Threshold={bankMigrationThreshold}, " +
                        $"Ratio={cumulativeErosion / bankMigrationThreshold:F2}x, ChannelWidth={channelWidth}");
                }
                
                // Check cumulative erosion at the bank edge
                if (cumulativeErosion >= bankMigrationThreshold)
                {
                    // Find the adjacent FLUID cell (moving inward from right)
                    int nextFluidW = rightBankEdge - 1;
                    if (nextFluidW >= 0 && cellType[i, nextFluidW] == RiverCellType.FLUID)
                    {
                        // Channel closure safeguard: prevent migration if it would close the channel
                        int newChannelWidth = (leftBankEdge >= 0 && leftBankEdge < nextFluidW) ? 
                            nextFluidW - leftBankEdge - 1 : -1;
                        
                        if (newChannelWidth >= 0 && newChannelWidth < minChannelWidth)
                        {
                            // Migration would close channel - prevent it
                            preventedMigrations++;
                            if (debugLogCounter % DEBUG_LOG_INTERVAL == 0)
                            {
                                Debug.LogWarning($"[BankMigration] CrossSection {i}: Prevented right bank migration - would reduce channel to {newChannelWidth} FLUID cells (minimum: {minChannelWidth})");
                            }
                        }
                        else
                        {
                            // Safe to migrate
                            // Convert FLUID to BANK: bank migrates inward
                            cellType[i, nextFluidW] = RiverCellType.BANK;
                            waterDepth[i, nextFluidW] = 0.0;
                            u[i, nextFluidW] = 0.0;
                            v[i, nextFluidW] = 0.0;
                            
                            // Transfer cumulative erosion to new bank position (carry over excess)
                            // Cap excess erosion to prevent immediate re-migration cascade
                            double excessErosion = cumulativeErosion - bankMigrationThreshold;
                            double maxExcessErosion = bankMigrationThreshold * 0.5; // Allow up to 50% of threshold
                            cumulativeBankErosion[i, nextFluidW] = Math.Min(excessErosion, maxExcessErosion);
                            cumulativeBankErosion[i, rightBankEdge] = 0.0;
                            totalMigrationsThisStep++;
                        }
                    }
                }
            }
        }
        
        // OUTWARD MIGRATION: Convert BANK to FLUID to widen channel (meandering)
        // This allows the channel to expand outward, especially on outer banks of bends
        int outwardMigrations = 0;
        for (int i = 0; i < numCrossSections; i++)
        {
            // Find bank edges again (may have changed from inward migration)
            int leftBankEdge = -1;
            for (int w = 0; w < widthResolution - 1; w++)
            {
                if (cellType[i, w] == RiverCellType.BANK && cellType[i, w + 1] == RiverCellType.FLUID)
                {
                    leftBankEdge = w;
                    break;
                }
            }
            if (leftBankEdge == -1 && cellType[i, 0] == RiverCellType.BANK)
            {
                leftBankEdge = 0;
            }
            
            int rightBankEdge = -1;
            for (int w = widthResolution - 1; w > 0; w--)
            {
                if (cellType[i, w] == RiverCellType.BANK && cellType[i, w - 1] == RiverCellType.FLUID)
                {
                    rightBankEdge = w;
                    break;
                }
            }
            if (rightBankEdge == -1 && cellType[i, widthResolution - 1] == RiverCellType.BANK)
            {
                rightBankEdge = widthResolution - 1;
            }
            
            // Check left bank for outward migration (channel widens left)
            if (leftBankEdge >= 0 && leftBankEdge > 0)
            {
                // Check if there's significant erosion on the left bank edge
                double leftBankErosion = cumulativeBankErosion[i, leftBankEdge];
                
                // For outward migration, use a MUCH higher threshold to prevent excessive widening
                // Only widen when erosion is extremely high (e.g., 50x the migration threshold)
                // This heavily favors meandering (inward migration) over widening
                double outwardMigrationThreshold = bankMigrationThreshold * 50.0;
                
                if (leftBankErosion >= outwardMigrationThreshold)
                {
                    // Convert leftmost BANK cell to FLUID (channel widens outward)
                    int outwardBankW = leftBankEdge - 1;
                    if (outwardBankW >= 0 && cellType[i, outwardBankW] == RiverCellType.BANK)
                    {
                        // Convert BANK to FLUID: channel widens
                        cellType[i, outwardBankW] = RiverCellType.FLUID;
                        waterDepth[i, outwardBankW] = 0.5; // Default water depth (same as initialization)
                        u[i, outwardBankW] = 0.0; // Will be set by flow
                        v[i, outwardBankW] = 0.0;
                        
                        // Reset erosion for the newly created FLUID cell
                        cumulativeBankErosion[i, outwardBankW] = 0.0;
                        
                        // Reduce erosion on the bank edge (some erosion was "used" to widen channel)
                        cumulativeBankErosion[i, leftBankEdge] = leftBankErosion - outwardMigrationThreshold;
                        outwardMigrations++;
                    }
                }
            }
            
            // Check right bank for outward migration (channel widens right)
            if (rightBankEdge >= 0 && rightBankEdge < widthResolution - 1)
            {
                // Check if there's significant erosion on the right bank edge
                double rightBankErosion = cumulativeBankErosion[i, rightBankEdge];
                
                // For outward migration, use a MUCH higher threshold to prevent excessive widening
                // Only widen when erosion is extremely high (e.g., 50x the migration threshold)
                // This heavily favors meandering (inward migration) over widening
                double outwardMigrationThreshold = bankMigrationThreshold * 50.0;
                
                if (rightBankErosion >= outwardMigrationThreshold)
                {
                    // Convert rightmost BANK cell to FLUID (channel widens outward)
                    int outwardBankW = rightBankEdge + 1;
                    if (outwardBankW < widthResolution && cellType[i, outwardBankW] == RiverCellType.BANK)
                    {
                        // Convert BANK to FLUID: channel widens
                        cellType[i, outwardBankW] = RiverCellType.FLUID;
                        waterDepth[i, outwardBankW] = 0.5; // Default water depth (same as initialization)
                        u[i, outwardBankW] = 0.0; // Will be set by flow
                        v[i, outwardBankW] = 0.0;
                        
                        // Reset erosion for the newly created FLUID cell
                        cumulativeBankErosion[i, outwardBankW] = 0.0;
                        
                        // Reduce erosion on the bank edge (some erosion was "used" to widen channel)
                        cumulativeBankErosion[i, rightBankEdge] = rightBankErosion - outwardMigrationThreshold;
                        outwardMigrations++;
                    }
                }
            }
        }
        
        // Debug logging for migration statistics
        if (debugLogCounter % DEBUG_LOG_INTERVAL == 0)
        {
            Debug.Log($"[BankMigration] Step {debugLogCounter}: InwardMigrations={totalMigrationsThisStep}, " +
                $"OutwardMigrations={outwardMigrations}, PreventedMigrations={preventedMigrations}, CriticalWidthSections={criticalWidthSections}");
        }
    }
    
    /// <summary>
    /// Detects meander cutoffs by checking when the channel physically narrows to near-zero width
    /// OR when two non-adjacent river segments collide (come very close together in 3D space).
    /// With vertex-based migration, banks are always at w=0 and w=widthResolution-1, so we detect
    /// cutoffs by measuring the distance between bank vertices rather than cell type overlap.
    /// </summary>
    /// <param name="vertices">Current mesh vertices array</param>
    /// <returns>List of cutoff points: (crossSectionIndex, leftEdge, rightEdge)</returns>
    public List<(int crossSection, int leftEdge, int rightEdge)> DetectCutoff(Vector3[] vertices = null)
    {
        List<(int crossSection, int leftEdge, int rightEdge)> cutoffs = new List<(int crossSection, int leftEdge, int rightEdge)>();
        
        // If vertices not provided, we can't detect cutoffs by distance - return empty list
        // This maintains backward compatibility but requires vertices for proper detection
        if (vertices == null || vertices.Length != numCrossSections * widthResolution)
        {
            // Fallback: check if all cells are BANK (channel completely closed)
            for (int i = 0; i < numCrossSections; i++)
            {
                bool hasFluid = false;
                for (int w = 0; w < widthResolution; w++)
                {
                    if (cellType[i, w] == RiverCellType.FLUID)
                    {
                        hasFluid = true;
                        break;
                    }
                }
                if (!hasFluid)
                {
                    cutoffs.Add((i, 0, widthResolution - 1));
                }
            }
            return cutoffs;
        }
        
        // Calculate typical channel width for threshold (use average of first few cross-sections)
        double typicalWidth = 0.0;
        int sampleCount = Math.Min(5, numCrossSections);
        for (int i = 0; i < sampleCount; i++)
        {
            int leftIdx = i * widthResolution;
            int rightIdx = i * widthResolution + (widthResolution - 1);
            if (leftIdx < vertices.Length && rightIdx < vertices.Length)
            {
                Vector3 leftBank = vertices[leftIdx];
                Vector3 rightBank = vertices[rightIdx];
                // Convert Unity units to meters for threshold calculation
                double widthUnity = Vector3.Distance(leftBank, rightBank);
                double widthMeters = widthUnity * unityToMetersScale;
                typicalWidth += widthMeters;
            }
        }
        typicalWidth = sampleCount > 0 ? typicalWidth / sampleCount : 10.0; // Default 10m if no samples
        
        // Threshold: 20% of typical width or absolute minimum of 0.3m (in meters)
        // More sensitive threshold to detect cutoffs earlier
        double minWidthThreshold = Math.Max(typicalWidth * 0.2, 0.3);
        
        // METHOD 1: Detect narrow channel width (existing method)
        for (int i = 0; i < numCrossSections; i++)
        {
            int leftIdx = i * widthResolution;
            int rightIdx = i * widthResolution + (widthResolution - 1);
            
            if (leftIdx >= vertices.Length || rightIdx >= vertices.Length)
                continue;
            
            Vector3 leftBank = vertices[leftIdx];
            Vector3 rightBank = vertices[rightIdx];
            
            // Calculate channel width in Unity units, then convert to meters
            float widthUnity = Vector3.Distance(leftBank, rightBank);
            double widthMeters = widthUnity * unityToMetersScale;
            
            // Cutoff detected when channel width falls below threshold
            if (widthMeters < minWidthThreshold)
            {
                cutoffs.Add((i, 0, widthResolution - 1));
            }
        }
        
        // METHOD 2: Detect bank collision - when two non-adjacent cross-sections are very close
        // This handles the case where meander loops come together but each cross-section is still wide
        double collisionThreshold = typicalWidth * 0.5; // Banks collide when distance < 50% of typical width
        double collisionThresholdMeters = Math.Max(collisionThreshold, 2.0); // Minimum 2m threshold
        float collisionThresholdSq = (float)(collisionThresholdMeters / unityToMetersScale);
        collisionThresholdSq *= collisionThresholdSq; // Use squared distance to avoid sqrt
        
        // OPTIMIZATION: Only check every Nth cross-section to reduce computation
        int checkInterval = Math.Max(1, numCrossSections / 100); // Check ~100 cross-sections max
        int maxSearchDistance = Math.Min(numCrossSections / 4, 200); // Limit search window
        
        // Pre-calculate centers for sampled cross-sections
        List<(int idx, Vector3 center, Vector3 leftBank, Vector3 rightBank)> sampledSections = 
            new List<(int, Vector3, Vector3, Vector3)>();
        
        for (int i = 0; i < numCrossSections - 5; i += checkInterval)
        {
            int leftIdx1 = i * widthResolution;
            int rightIdx1 = i * widthResolution + (widthResolution - 1);
            
            if (leftIdx1 >= vertices.Length || rightIdx1 >= vertices.Length)
                continue;
            
            Vector3 leftBank = vertices[leftIdx1];
            Vector3 rightBank = vertices[rightIdx1];
            Vector3 center = (leftBank + rightBank) * 0.5f;
            sampledSections.Add((i, center, leftBank, rightBank));
        }
        
        // Check distances between sampled cross-sections
        for (int i = 0; i < sampledSections.Count; i++)
        {
            var section1 = sampledSections[i];
            
            // Only check against sections that are far enough in index but potentially close in space
            int minJ = i + Math.Max(10 / checkInterval, 1); // At least 10 indices apart
            int maxJ = Math.Min(sampledSections.Count, i + maxSearchDistance / checkInterval);
            
            for (int j = minJ; j < maxJ; j++)
            {
                var section2 = sampledSections[j];
                
                // Early exit: use squared distance to avoid expensive sqrt
                Vector3 diff = section1.center - section2.center;
                float distanceSq = diff.sqrMagnitude;
                
                if (distanceSq > collisionThresholdSq * 2.0f) // 2x threshold for early exit
                    continue; // Too far, skip expensive bank distance checks
                
                // Now calculate actual distance (only for close pairs)
                float distanceUnity = Mathf.Sqrt(distanceSq);
                double distanceMeters = distanceUnity * unityToMetersScale;
                
                if (distanceMeters >= collisionThresholdMeters)
                    continue;
                
                // Check bank-to-bank distances (only for close pairs)
                Vector3 leftToRight = section1.leftBank - section2.rightBank;
                Vector3 rightToLeft = section1.rightBank - section2.leftBank;
                float leftToRightDistSq = leftToRight.sqrMagnitude;
                float rightToLeftDistSq = rightToLeft.sqrMagnitude;
                
                // Use squared distance comparison first
                float bankThresholdSq = collisionThresholdSq * 0.25f; // 0.5^2 = 0.25
                if (leftToRightDistSq > bankThresholdSq && rightToLeftDistSq > bankThresholdSq)
                    continue;
                
                float minBankDist = Mathf.Min(Mathf.Sqrt(leftToRightDistSq), Mathf.Sqrt(rightToLeftDistSq));
                double minBankDistanceMeters = minBankDist * unityToMetersScale;
                
                // Collision detected: cross-sections are close AND banks are very close
                if (minBankDistanceMeters < collisionThresholdMeters * 0.5)
                {
                    // Find the midpoint between the two colliding sections
                    int midpointIdx = (section1.idx + section2.idx) / 2;
                    
                    // Check if this cutoff hasn't already been added
                    bool alreadyAdded = false;
                    foreach (var existing in cutoffs)
                    {
                        if (Math.Abs(existing.crossSection - midpointIdx) < 5)
                        {
                            alreadyAdded = true;
                            break;
                        }
                    }
                    
                    if (!alreadyAdded && midpointIdx < numCrossSections)
                    {
                        cutoffs.Add((midpointIdx, 0, widthResolution - 1));
                        Debug.Log($"[CutoffDetection] Bank collision detected between cross-sections {section1.idx} and {section2.idx} " +
                                 $"(distance: {distanceMeters:F2}m, bank distance: {minBankDistanceMeters:F2}m). " +
                                 $"Cutoff at midpoint: {midpointIdx}");
                    }
                }
            }
        }
        
        return cutoffs;
    }
    
    /// <summary>
    /// Breaks through banks when two river segments collide, creating a flow path.
    /// WARNING: This method modifies cell types but does NOT update vertex positions.
    /// Vertex positions must be updated separately via UpdateMeshVerticesForBankMigration.
    /// This method is now DISABLED by default to prevent mesh corruption - use cutoff detection instead.
    /// </summary>
    /// <param name="vertices">Current mesh vertex positions</param>
    /// <returns>True if any banks were broken, false otherwise</returns>
    public bool BreakBanksOnCollision(Vector3[] vertices)
    {
        // DISABLED: Bank breaking causes mesh corruption because it modifies cell types
        // without properly updating vertex positions. Use cutoff detection instead.
        // If re-enabled, must ensure vertex positions are updated to match cell type changes.
        return false;
        
        /* DISABLED CODE - causes mesh corruption
        if (vertices == null || vertices.Length != numCrossSections * widthResolution)
            return false;
        
        bool anyBanksBroken = false;
        
        // Calculate typical channel width for collision threshold
        double typicalWidth = 0.0;
        int sampleCount = Math.Min(5, numCrossSections);
        for (int i = 0; i < sampleCount; i++)
        {
            int leftIdx = i * widthResolution;
            int rightIdx = i * widthResolution + (widthResolution - 1);
            if (leftIdx < vertices.Length && rightIdx < vertices.Length)
            {
                Vector3 leftBank = vertices[leftIdx];
                Vector3 rightBank = vertices[rightIdx];
                double widthUnity = Vector3.Distance(leftBank, rightBank);
                double widthMeters = widthUnity * unityToMetersScale;
                typicalWidth += widthMeters;
            }
        }
        typicalWidth = sampleCount > 0 ? typicalWidth / sampleCount : 10.0;
        
        // Collision threshold: banks break when segments are within 60% of typical width
        double breakThresholdMeters = Math.Max(typicalWidth * 0.6, 3.0); // Minimum 3m
        float breakThresholdSq = (float)(breakThresholdMeters / unityToMetersScale);
        breakThresholdSq *= breakThresholdSq; // Use squared distance to avoid sqrt
        
        // OPTIMIZATION: Only check every Nth cross-section and limit search window
        int checkInterval = Math.Max(1, numCrossSections / 100); // Check ~100 cross-sections max
        int maxSearchDistance = Math.Min(numCrossSections / 4, 200); // Limit search window
        
        // Pre-calculate centers for sampled cross-sections
        List<(int idx, Vector3 center, Vector3 leftBank, Vector3 rightBank)> sampledSections = 
            new List<(int, Vector3, Vector3, Vector3)>();
        
        for (int i = 0; i < numCrossSections - 5; i += checkInterval)
        {
            int leftIdx1 = i * widthResolution;
            int rightIdx1 = i * widthResolution + (widthResolution - 1);
            
            if (leftIdx1 >= vertices.Length || rightIdx1 >= vertices.Length)
                continue;
            
            Vector3 leftBank = vertices[leftIdx1];
            Vector3 rightBank = vertices[rightIdx1];
            Vector3 center = (leftBank + rightBank) * 0.5f;
            sampledSections.Add((i, center, leftBank, rightBank));
        }
        
        // Check for collisions between sampled cross-sections
        for (int i = 0; i < sampledSections.Count; i++)
        {
            var section1 = sampledSections[i];
            
            // Only check against sections that are far enough in index
            int minJ = i + Math.Max(5 / checkInterval, 1); // At least 5 indices apart
            int maxJ = Math.Min(sampledSections.Count, i + maxSearchDistance / checkInterval);
            
            for (int j = minJ; j < maxJ; j++)
            {
                var section2 = sampledSections[j];
                
                // Early exit: use squared distance to avoid expensive sqrt
                Vector3 diff = section1.center - section2.center;
                float distanceSq = diff.sqrMagnitude;
                
                if (distanceSq > breakThresholdSq * 2.0f) // 2x threshold for early exit
                    continue; // Too far, skip expensive bank distance checks
                
                // Now calculate actual distance (only for close pairs)
                float distanceUnity = Mathf.Sqrt(distanceSq);
                double distanceMeters = distanceUnity * unityToMetersScale;
                
                if (distanceMeters >= breakThresholdMeters)
                    continue;
                
                // Check bank-to-bank distances (only for close pairs)
                Vector3 leftToRight = section1.leftBank - section2.rightBank;
                Vector3 rightToLeft = section1.rightBank - section2.leftBank;
                float leftToRightDistSq = leftToRight.sqrMagnitude;
                float rightToLeftDistSq = rightToLeft.sqrMagnitude;
                
                // Use squared distance comparison first
                float bankThresholdSq = breakThresholdSq * 0.49f; // 0.7^2 ≈ 0.49
                if (leftToRightDistSq > bankThresholdSq && rightToLeftDistSq > bankThresholdSq)
                    continue;
                
                float minBankDist = Mathf.Min(Mathf.Sqrt(leftToRightDistSq), Mathf.Sqrt(rightToLeftDistSq));
                double minBankDistanceMeters = minBankDist * unityToMetersScale;
                
                // Bank breaking: segments are close AND banks are very close
                if (minBankDistanceMeters < breakThresholdMeters * 0.7)
                {
                    // Determine which banks are colliding
                    bool leftToRightCollision = leftToRightDistSq < rightToLeftDistSq;
                    
                    // Break through banks in the region between the two colliding cross-sections
                    // Create a flow path connecting the two segments
                    int startIdx = Math.Max(0, section1.idx - 3);
                    int endIdx = Math.Min(numCrossSections - 1, section2.idx + 3);
                    
                    // Calculate midpoint for creating connection
                    int midpointIdx = (section1.idx + section2.idx) / 2;
                    
                    for (int k = startIdx; k <= endIdx; k++)
                    {
                        // Calculate interpolation factor (0 at start, 1 at end, 0.5 at midpoint)
                        float t = (float)(k - startIdx) / Math.Max(1, endIdx - startIdx);
                        
                        // Determine which side to break through based on collision type
                        int breakStartW, breakEndW;
                        if (leftToRightCollision)
                        {
                            // Left bank of section i colliding with right bank of section j
                            // Break through left side, creating connection that widens toward midpoint
                            breakStartW = 0;
                            breakEndW = Math.Min(widthResolution / 3, widthResolution - 1);
                        }
                        else
                        {
                            // Right bank of section i colliding with left bank of section j
                            // Break through right side
                            breakStartW = Math.Max(widthResolution * 2 / 3, 0);
                            breakEndW = widthResolution - 1;
                        }
                        
                        // Break through more cells near the midpoint for better connection
                        int numCellsToBreak = (int)(3 + 5 * (1.0f - Math.Abs(t - 0.5f) * 2.0f)); // More cells at midpoint
                        numCellsToBreak = Math.Min(numCellsToBreak, breakEndW - breakStartW + 1);
                        
                        // Convert BANK cells to FLUID to create flow path
                        for (int w = breakStartW; w <= breakEndW && (w - breakStartW) < numCellsToBreak; w++)
                        {
                            if (cellType[k, w] == RiverCellType.BANK)
                            {
                                cellType[k, w] = RiverCellType.FLUID;
                                waterDepth[k, w] = 0.4; // Add water depth for flow
                                
                                // Set initial velocity to encourage flow through the break
                                // Velocity direction depends on which segment is upstream
                                if (k < midpointIdx)
                                {
                                    // Upstream of midpoint - flow toward break
                                    u[k, w] = 0.2; // Flow along river
                                    v[k, w] = leftToRightCollision ? -0.1f : 0.1f; // Flow toward break
                                }
                                else
                                {
                                    // Downstream of midpoint - flow away from break
                                    u[k, w] = 0.2;
                                    v[k, w] = leftToRightCollision ? 0.1f : -0.1f;
                                }
                                
                                anyBanksBroken = true;
                            }
                        }
                    }
                    
                    if (anyBanksBroken)
                    {
                        Debug.Log($"[BankBreaking] Banks broken between cross-sections {section1.idx} and {section2.idx} " +
                                 $"(distance: {distanceMeters:F2}m, bank distance: {minBankDistanceMeters:F2}m). " +
                                 $"Converted BANK to FLUID in region {startIdx}-{endIdx}");
                        
                        // Ensure flow connectivity: propagate water depth and velocity from adjacent FLUID cells
                        EnsureFlowConnectivity(startIdx, endIdx);
                    }
                }
            }
        }
        
        return anyBanksBroken;
        */
    }
    
    /// <summary>
    /// Ensures flow connectivity after bank breaking by propagating water properties from adjacent FLUID cells.
    /// This helps water flow through newly broken banks.
    /// </summary>
    private void EnsureFlowConnectivity(int startIdx, int endIdx)
    {
        for (int i = startIdx; i <= endIdx && i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                if (cellType[i, w] == RiverCellType.FLUID && waterDepth[i, w] < 0.1)
                {
                    // Find adjacent FLUID cells with water
                    double avgWaterDepth = 0.0;
                    double avgU = 0.0;
                    double avgV = 0.0;
                    int neighborCount = 0;
                    
                    // Check neighbors
                    if (i > 0 && cellType[i - 1, w] == RiverCellType.FLUID && waterDepth[i - 1, w] > 0.1)
                    {
                        avgWaterDepth += waterDepth[i - 1, w];
                        avgU += u[i - 1, w];
                        avgV += v[i - 1, w];
                        neighborCount++;
                    }
                    if (i < numCrossSections - 1 && cellType[i + 1, w] == RiverCellType.FLUID && waterDepth[i + 1, w] > 0.1)
                    {
                        avgWaterDepth += waterDepth[i + 1, w];
                        avgU += u[i + 1, w];
                        avgV += v[i + 1, w];
                        neighborCount++;
                    }
                    if (w > 0 && cellType[i, w - 1] == RiverCellType.FLUID && waterDepth[i, w - 1] > 0.1)
                    {
                        avgWaterDepth += waterDepth[i, w - 1];
                        avgU += u[i, w - 1];
                        avgV += v[i, w - 1];
                        neighborCount++;
                    }
                    if (w < widthResolution - 1 && cellType[i, w + 1] == RiverCellType.FLUID && waterDepth[i, w + 1] > 0.1)
                    {
                        avgWaterDepth += waterDepth[i, w + 1];
                        avgU += u[i, w + 1];
                        avgV += v[i, w + 1];
                        neighborCount++;
                    }
                    
                    // Propagate properties from neighbors
                    if (neighborCount > 0)
                    {
                        waterDepth[i, w] = Math.Max(waterDepth[i, w], avgWaterDepth / neighborCount * 0.8);
                        u[i, w] = avgU / neighborCount;
                        v[i, w] = avgV / neighborCount;
                    }
                }
            }
        }
    }
    
    /// <summary>
    /// Finds disconnected sections between cutoff points.
    /// Uses connectivity analysis to identify which cross-sections are part of the oxbow.
    /// </summary>
    /// <param name="cutoffStart">Cross-section index where cutoff starts</param>
    /// <param name="cutoffEnd">Cross-section index where cutoff ends</param>
    /// <returns>List of disconnected cross-section ranges: (start, end) inclusive</returns>
    public List<(int start, int end)> FindDisconnectedSections(int cutoffStart, int cutoffEnd)
    {
        List<(int, int)> disconnectedRanges = new List<(int, int)>();
        
        // Use flood-fill from upstream (before cutoffStart) to find connected sections
        HashSet<int> connectedSections = GetConnectedFluidSections(0); // Start from upstream
        
        // Find ranges of cross-sections that are NOT connected to upstream
        int currentStart = -1;
        for (int i = cutoffStart; i <= cutoffEnd && i < numCrossSections; i++)
        {
            bool isConnected = connectedSections.Contains(i);
            
            if (!isConnected && currentStart == -1)
            {
                // Start of disconnected range
                currentStart = i;
            }
            else if (isConnected && currentStart != -1)
            {
                // End of disconnected range
                disconnectedRanges.Add((currentStart, i - 1));
                currentStart = -1;
            }
        }
        
        // Handle case where disconnected range extends to end
        if (currentStart != -1)
        {
            disconnectedRanges.Add((currentStart, Math.Min(cutoffEnd, numCrossSections - 1)));
        }
        
        return disconnectedRanges;
    }
    
    /// <summary>
    /// Applies boundary conditions to bed elevation to prevent edges from falling.
    /// Banks are constrained to maintain minimum height relative to adjacent fluid cells.
    /// </summary>
    private void ApplyBedElevationBoundaryConditions(double[,] h_new)
    {
        // For BANK cells, maintain elevation at or above adjacent fluid cells
        // This prevents bank edges from eroding downward
        double bankHeightBuffer = 0.05; // Minimum height difference between bank and adjacent fluid
        
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                if (cellType[i, w] == RiverCellType.BANK)
                {
                    // Find maximum elevation of adjacent fluid cells
                    double maxAdjacentFluidElevation = double.MinValue;
                    bool hasAdjacentFluid = false;
                    
                    // Check adjacent cells along river (s-direction)
                    if (i > 0 && cellType[i - 1, w] == RiverCellType.FLUID)
                    {
                        maxAdjacentFluidElevation = Math.Max(maxAdjacentFluidElevation, h_new[i - 1, w]);
                        hasAdjacentFluid = true;
                    }
                    if (i < numCrossSections - 1 && cellType[i + 1, w] == RiverCellType.FLUID)
                    {
                        maxAdjacentFluidElevation = Math.Max(maxAdjacentFluidElevation, h_new[i + 1, w]);
                        hasAdjacentFluid = true;
                    }
                    
                    // Check adjacent cells across river (w-direction)
                    if (w > 0 && cellType[i, w - 1] == RiverCellType.FLUID)
                    {
                        maxAdjacentFluidElevation = Math.Max(maxAdjacentFluidElevation, h_new[i, w - 1]);
                        hasAdjacentFluid = true;
                    }
                    if (w < widthResolution - 1 && cellType[i, w + 1] == RiverCellType.FLUID)
                    {
                        maxAdjacentFluidElevation = Math.Max(maxAdjacentFluidElevation, h_new[i, w + 1]);
                        hasAdjacentFluid = true;
                    }
                    
                    // Constrain bank elevation to be at least buffer height above adjacent fluid
                    if (hasAdjacentFluid)
                    {
                        double minBankElevation = maxAdjacentFluidElevation + bankHeightBuffer;
                        h_new[i, w] = Math.Max(h_new[i, w], minBankElevation);
                    }
                    
                    // Alternative: Keep bank elevation stable (preserve original height)
                    // This prevents any vertical erosion on banks
                    // h_new[i, w] = h[i, w];
                }
            }
        }
        
        // Apply boundary conditions at river ends to prevent edge instability
        // Upstream boundary: maintain initial elevation or smooth with adjacent
        for (int w = 0; w < widthResolution; w++)
        {
            if (numCrossSections > 1 && cellType[0, w] == RiverCellType.FLUID)
            {
                // Smooth with downstream neighbor
                h_new[0, w] = 0.7 * h_new[0, w] + 0.3 * h_new[1, w];
            }
        }
        
        // Downstream boundary: smooth with upstream neighbor
        for (int w = 0; w < widthResolution; w++)
        {
            if (numCrossSections > 1 && cellType[numCrossSections - 1, w] == RiverCellType.FLUID)
            {
                h_new[numCrossSections - 1, w] = 0.7 * h_new[numCrossSections - 1, w] + 0.3 * h_new[numCrossSections - 2, w];
            }
        }
    }
    
    /// <summary>
    /// Applies Laplacian smoothing to reduce numerical noise in bed elevation.
    /// This helps prevent mesh "peaking" and instability.
    /// Uses a conservative smoothing approach that only reduces sharp variations.
    /// </summary>
    private double[,] ApplySpatialSmoothing(double[,] h_data, double smoothingFactor = 0.1)
    {
        double[,] h_smoothed = (double[,])h_data.Clone();
        
        // Apply Laplacian smoothing: h_new = h_old + smoothingFactor * Laplacian(h_old)
        // This diffuses sharp peaks and valleys
        for (int i = 1; i < numCrossSections - 1; i++)
        {
            for (int w = 1; w < widthResolution - 1; w++)
            {
                double ds_curr = Math.Max(ds[i], 1e-6); // Avoid division by zero
                double dw_curr = Math.Max(dw[i], 1e-6);
                
                // Smooth FLUID cells
                if (cellType[i, w] == RiverCellType.FLUID)
                {
                    // Compute Laplacian using only FLUID neighbors
                    double laplacian_s = 0.0;
                    double laplacian_w = 0.0;
                    
                    // Along-river direction (s)
                    if (cellType[i + 1, w] == RiverCellType.FLUID && cellType[i - 1, w] == RiverCellType.FLUID)
                    {
                        laplacian_s = (h_data[i + 1, w] - 2 * h_data[i, w] + h_data[i - 1, w]) / (ds_curr * ds_curr);
                    }
                    
                    // Across-river direction (w)
                    if (cellType[i, w + 1] == RiverCellType.FLUID && cellType[i, w - 1] == RiverCellType.FLUID)
                    {
                        laplacian_w = (h_data[i, w + 1] - 2 * h_data[i, w] + h_data[i, w - 1]) / (dw_curr * dw_curr);
                    }
                    
                    double laplacian = laplacian_s + laplacian_w;
                    
                    // Apply conservative smoothing (only reduce peaks, don't amplify)
                    double smoothed_value = h_data[i, w] + smoothingFactor * laplacian;
                    
                    // Limit smoothing to prevent unrealistic values
                    double max_change = 0.01; // Maximum change per smoothing step
                    smoothed_value = Math.Max(h_data[i, w] - max_change, Math.Min(h_data[i, w] + max_change, smoothed_value));
                    
                    h_smoothed[i, w] = smoothed_value;
                }
                // Smooth BANK cells at boundaries to prevent sharp edges
                else if (cellType[i, w] == RiverCellType.BANK)
                {
                    // Apply light smoothing to bank cells to prevent sharp drops
                    // Average with adjacent cells (weighted by cell type)
                    double weightedSum = h_data[i, w];
                    double totalWeight = 1.0;
                    
                    // Check neighbors and include them if they exist
                    if (i > 0)
                    {
                        weightedSum += h_data[i - 1, w];
                        totalWeight += 1.0;
                    }
                    if (i < numCrossSections - 1)
                    {
                        weightedSum += h_data[i + 1, w];
                        totalWeight += 1.0;
                    }
                    if (w > 0)
                    {
                        weightedSum += h_data[i, w - 1];
                        totalWeight += 1.0;
                    }
                    if (w < widthResolution - 1)
                    {
                        weightedSum += h_data[i, w + 1];
                        totalWeight += 1.0;
                    }
                    
                    // Apply very conservative smoothing to banks (only 10% of difference)
                    double neighborAvg = weightedSum / totalWeight;
                    double smoothed_value = 0.9 * h_data[i, w] + 0.1 * neighborAvg;
                    
                    // Ensure bank doesn't drop below adjacent fluid cells
                    double minAdjacentFluid = h_data[i, w];
                    if (i > 0 && cellType[i - 1, w] == RiverCellType.FLUID)
                        minAdjacentFluid = Math.Min(minAdjacentFluid, h_data[i - 1, w]);
                    if (i < numCrossSections - 1 && cellType[i + 1, w] == RiverCellType.FLUID)
                        minAdjacentFluid = Math.Min(minAdjacentFluid, h_data[i + 1, w]);
                    if (w > 0 && cellType[i, w - 1] == RiverCellType.FLUID)
                        minAdjacentFluid = Math.Min(minAdjacentFluid, h_data[i, w - 1]);
                    if (w < widthResolution - 1 && cellType[i, w + 1] == RiverCellType.FLUID)
                        minAdjacentFluid = Math.Min(minAdjacentFluid, h_data[i, w + 1]);
                    
                    smoothed_value = Math.Max(smoothed_value, minAdjacentFluid + 0.05); // Keep bank above fluid
                    
                    h_smoothed[i, w] = smoothed_value;
                }
            }
        }
        
        return h_smoothed;
    }
    
    /// <summary>
    /// Performs flood-fill/BFS from a starting cross-section to find all connected FLUID sections.
    /// Two cross-sections are connected if they have adjacent FLUID cells.
    /// </summary>
    /// <param name="startCrossSection">Starting cross-section index (typically 0 for upstream)</param>
    /// <returns>Set of cross-section indices that are connected to the start</returns>
    public HashSet<int> GetConnectedFluidSections(int startCrossSection)
    {
        HashSet<int> connected = new HashSet<int>();
        if (startCrossSection < 0 || startCrossSection >= numCrossSections)
            return connected;
        
        Queue<int> queue = new Queue<int>();
        queue.Enqueue(startCrossSection);
        connected.Add(startCrossSection);
        
        while (queue.Count > 0)
        {
            int current = queue.Dequeue();
            
            // Check adjacent cross-sections (upstream and downstream)
            int[] neighbors = { current - 1, current + 1 };
            
            foreach (int neighbor in neighbors)
            {
                if (neighbor < 0 || neighbor >= numCrossSections)
                    continue;
                
                if (connected.Contains(neighbor))
                    continue;
                
                // Check if there's a FLUID connection between current and neighbor
                bool hasFluidConnection = false;
                for (int w = 0; w < widthResolution; w++)
                {
                    if (cellType[current, w] == RiverCellType.FLUID && 
                        cellType[neighbor, w] == RiverCellType.FLUID)
                    {
                        hasFluidConnection = true;
                        break;
                    }
                }
                
                if (hasFluidConnection)
                {
                    connected.Add(neighbor);
                    queue.Enqueue(neighbor);
                }
            }
        }
        
        return connected;
    }
    
    /// <summary>
    /// Extracts data for an oxbow section from the main river solver.
    /// </summary>
    /// <param name="startCrossSection">Starting cross-section index (inclusive)</param>
    /// <param name="endCrossSection">Ending cross-section index (inclusive)</param>
    /// <param name="vertices">Current mesh vertices</param>
    /// <returns>OxbowSectionData containing all necessary data to create a new solver</returns>
    public OxbowSectionData ExtractOxbowSection(int startCrossSection, int endCrossSection, Vector3[] vertices)
    {
        int oxbowLength = endCrossSection - startCrossSection + 1;
        
        // Extract vertices
        Vector3[] oxbowVertices = new Vector3[oxbowLength * widthResolution];
        for (int i = 0; i < oxbowLength; i++)
        {
            int sourceIdx = (startCrossSection + i) * widthResolution;
            int destIdx = i * widthResolution;
            for (int w = 0; w < widthResolution; w++)
            {
                oxbowVertices[destIdx + w] = vertices[sourceIdx + w];
            }
        }
        
        // Extract physics state
        double[,] oxbowH = new double[oxbowLength, widthResolution];
        double[,] oxbowWaterDepth = new double[oxbowLength, widthResolution];
        int[,] oxbowCellType = new int[oxbowLength, widthResolution];
        double[,] oxbowU = new double[oxbowLength, widthResolution];
        double[,] oxbowV = new double[oxbowLength, widthResolution];
        double[,] oxbowInitialElevation = new double[oxbowLength, widthResolution];
        
        for (int i = 0; i < oxbowLength; i++)
        {
            int sourceIdx = startCrossSection + i;
            for (int w = 0; w < widthResolution; w++)
            {
                oxbowH[i, w] = h[sourceIdx, w];
                oxbowWaterDepth[i, w] = waterDepth[sourceIdx, w];
                oxbowCellType[i, w] = cellType[sourceIdx, w];
                oxbowU[i, w] = u[sourceIdx, w];
                oxbowV[i, w] = v[sourceIdx, w];
                if (initialBedElevation != null)
                {
                    oxbowInitialElevation[i, w] = initialBedElevation[sourceIdx, w];
                }
            }
        }
        
        return new OxbowSectionData
        {
            startCrossSection = startCrossSection,
            endCrossSection = endCrossSection,
            vertices = oxbowVertices,
            h = oxbowH,
            waterDepth = oxbowWaterDepth,
            cellType = oxbowCellType,
            u = oxbowU,
            v = oxbowV,
            initialBedElevation = oxbowInitialElevation
        };
    }
    
    /// <summary>
    /// Interpolates a new cross-section between upstream and downstream cross-sections.
    /// Creates smooth transition for reconnecting the channel after cutoff.
    /// </summary>
    /// <param name="upstreamIdx">Upstream cross-section index</param>
    /// <param name="downstreamIdx">Downstream cross-section index</param>
    /// <param name="t">Interpolation parameter (0 = upstream, 1 = downstream)</param>
    /// <param name="vertices">Current vertex array</param>
    /// <returns>New cross-section data (vertices, h, waterDepth, cellType, etc.)</returns>
    private CrossSectionData InterpolateCrossSection(int upstreamIdx, int downstreamIdx, float t, Vector3[] vertices)
    {
        CrossSectionData result = new CrossSectionData
        {
            vertices = new Vector3[widthResolution],
            h = new double[widthResolution],
            waterDepth = new double[widthResolution],
            cellType = new int[widthResolution],
            u = new double[widthResolution],
            v = new double[widthResolution],
            initialBedElevation = new double[widthResolution]
        };
        
        for (int w = 0; w < widthResolution; w++)
        {
            int upstreamVertexIdx = upstreamIdx * widthResolution + w;
            int downstreamVertexIdx = downstreamIdx * widthResolution + w;
            
            // Interpolate vertex positions
            result.vertices[w] = Vector3.Lerp(vertices[upstreamVertexIdx], vertices[downstreamVertexIdx], t);
            
            // Interpolate bed elevation
            result.h[w] = h[upstreamIdx, w] * (1.0 - t) + h[downstreamIdx, w] * t;
            
            // Interpolate water depth
            result.waterDepth[w] = waterDepth[upstreamIdx, w] * (1.0 - t) + waterDepth[downstreamIdx, w] * t;
            
            // Interpolate velocities
            result.u[w] = u[upstreamIdx, w] * (1.0 - t) + u[downstreamIdx, w] * t;
            result.v[w] = v[upstreamIdx, w] * (1.0 - t) + v[downstreamIdx, w] * t;
            
            // Interpolate initial elevation
            if (initialBedElevation != null)
            {
                result.initialBedElevation[w] = initialBedElevation[upstreamIdx, w] * (1.0 - t) + 
                                                initialBedElevation[downstreamIdx, w] * t;
            }
            
            // Determine cell type: if either is FLUID, make it FLUID; otherwise BANK
            if (cellType[upstreamIdx, w] == RiverCellType.FLUID || cellType[downstreamIdx, w] == RiverCellType.FLUID)
            {
                result.cellType[w] = RiverCellType.FLUID;
            }
            else
            {
                result.cellType[w] = RiverCellType.BANK;
            }
        }
        
        // Enforce bank structure: w=0 and w=widthResolution-1 must always be BANK
        // This ensures reconnected sections maintain the correct structure for vertex-based migration
        result.cellType[0] = RiverCellType.BANK;
        result.cellType[widthResolution - 1] = RiverCellType.BANK;
        
        return result;
    }
    
    /// <summary>
    /// Reconnects the main channel after a cutoff by removing disconnected sections
    /// and interpolating new cross-sections between upstream and downstream.
    /// NOTE: This method requires numCrossSections to be resizable (will be implemented next).
    /// </summary>
    /// <param name="cutoffStart">Start of cutoff range (inclusive)</param>
    /// <param name="cutoffEnd">End of cutoff range (inclusive)</param>
    /// <param name="mainChannelVertices">Current vertex array (will be updated)</param>
    /// <returns>New vertex array with reconnected channel</returns>
    public Vector3[] ReconnectChannel(int cutoffStart, int cutoffEnd, Vector3[] mainChannelVertices)
    {
        if (cutoffStart < 0 || cutoffEnd >= numCrossSections || cutoffStart > cutoffEnd)
        {
            Debug.LogError($"[RiverMeshPhysicsSolver] Invalid cutoff range: {cutoffStart} to {cutoffEnd}");
            return mainChannelVertices; // Return original vertices unchanged
        }
        
        // Validate input vertices
        if (mainChannelVertices == null || mainChannelVertices.Length != numCrossSections * widthResolution)
        {
            Debug.LogError($"[RiverMeshPhysicsSolver] Invalid vertex array for reconnection. Expected {numCrossSections * widthResolution}, got {(mainChannelVertices?.Length ?? 0)}");
            return mainChannelVertices; // Return original vertices unchanged
        }
        
        // Validate vertices are finite
        for (int i = 0; i < mainChannelVertices.Length; i++)
        {
            if (!float.IsFinite(mainChannelVertices[i].x) || 
                !float.IsFinite(mainChannelVertices[i].y) || 
                !float.IsFinite(mainChannelVertices[i].z))
            {
                Debug.LogError($"[RiverMeshPhysicsSolver] Invalid vertex at index {i} before reconnection");
                return mainChannelVertices; // Return original vertices unchanged
            }
        }
        
        // Find upstream and downstream cross-sections (before cutoffStart and after cutoffEnd)
        int upstreamIdx = Math.Max(0, cutoffStart - 1);
        int downstreamIdx = Math.Min(numCrossSections - 1, cutoffEnd + 1);
        
        // Number of cross-sections to remove
        int sectionsToRemove = cutoffEnd - cutoffStart + 1;
        
        // Number of interpolated cross-sections to add (smooth transition)
        int interpolatedSections = Math.Max(1, Math.Min(3, sectionsToRemove / 2)); // Add 1-3 sections
        
        // Calculate new number of cross-sections
        int newNumCrossSections = numCrossSections - sectionsToRemove + interpolatedSections;
        
        // Create new vertex array
        Vector3[] newVertices = new Vector3[newNumCrossSections * widthResolution];
        
        // Copy vertices before cutoff
        for (int i = 0; i < upstreamIdx + 1; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                int srcIdx = i * widthResolution + w;
                int dstIdx = i * widthResolution + w;
                if (srcIdx < mainChannelVertices.Length && dstIdx < newVertices.Length)
                {
                    Vector3 v = mainChannelVertices[srcIdx];
                    // Validate vertex before copying
                    if (float.IsFinite(v.x) && float.IsFinite(v.y) && float.IsFinite(v.z))
                    {
                        newVertices[dstIdx] = v;
                    }
                    else
                    {
                        Debug.LogWarning($"[RiverMeshPhysicsSolver] Invalid vertex at srcIdx {srcIdx}, using default");
                        newVertices[dstIdx] = new Vector3(w * 0.1f, 0.0f, i * 0.1f);
                    }
                }
            }
        }
        
        // Add interpolated cross-sections
        int newIdx = upstreamIdx + 1;
        for (int interp = 0; interp < interpolatedSections; interp++)
        {
            float t = (float)(interp + 1) / (interpolatedSections + 1);
            CrossSectionData interpSection = InterpolateCrossSection(upstreamIdx, downstreamIdx, t, mainChannelVertices);
            
            for (int w = 0; w < widthResolution; w++)
            {
                int dstIdx = newIdx * widthResolution + w;
                if (dstIdx < newVertices.Length && w < interpSection.vertices.Length)
                {
                    Vector3 v = interpSection.vertices[w];
                    // Validate interpolated vertex
                    if (float.IsFinite(v.x) && float.IsFinite(v.y) && float.IsFinite(v.z))
                    {
                        newVertices[dstIdx] = v;
                    }
                    else
                    {
                        Debug.LogWarning($"[RiverMeshPhysicsSolver] Invalid interpolated vertex at dstIdx {dstIdx}, using default");
                        newVertices[dstIdx] = new Vector3(w * 0.1f, 0.0f, newIdx * 0.1f);
                    }
                }
            }
            newIdx++;
        }
        
        // Copy vertices after cutoff (adjusting indices)
        int sourceStartIdx = downstreamIdx;
        int destStartIdx = newIdx;
        for (int i = 0; i < numCrossSections - downstreamIdx; i++)
        {
            int sourceIdx = sourceStartIdx + i;
            int destIdx = destStartIdx + i;
            if (destIdx < newNumCrossSections)
            {
                for (int w = 0; w < widthResolution; w++)
                {
                    int srcIdx = sourceIdx * widthResolution + w;
                    int dstIdx = destIdx * widthResolution + w;
                    if (srcIdx < mainChannelVertices.Length && dstIdx < newVertices.Length)
                    {
                        Vector3 v = mainChannelVertices[srcIdx];
                        // Validate vertex before copying
                        if (float.IsFinite(v.x) && float.IsFinite(v.y) && float.IsFinite(v.z))
                        {
                            newVertices[dstIdx] = v;
                        }
                        else
                        {
                            Debug.LogWarning($"[RiverMeshPhysicsSolver] Invalid vertex at srcIdx {srcIdx}, using default");
                            newVertices[dstIdx] = new Vector3(w * 0.1f, 0.0f, destIdx * 0.1f);
                        }
                    }
                }
            }
        }
        
        // Final validation of output vertices
        for (int i = 0; i < newVertices.Length; i++)
        {
            if (!float.IsFinite(newVertices[i].x) || 
                !float.IsFinite(newVertices[i].y) || 
                !float.IsFinite(newVertices[i].z))
            {
                Debug.LogError($"[RiverMeshPhysicsSolver] Invalid vertex created at index {i} after reconnection");
                // Replace with safe default
                int crossSection = i / widthResolution;
                int width = i % widthResolution;
                newVertices[i] = new Vector3(width * 0.1f, 0.0f, crossSection * 0.1f);
            }
        }
        
        // Resize arrays to match new structure
        ResizeArrays(newNumCrossSections, cutoffStart, cutoffEnd, upstreamIdx, downstreamIdx, interpolatedSections, newVertices);
        
        // Update cell types to ensure banks are at edges (w=0 and w=widthResolution-1)
        // This is critical for vertex-based migration system where banks must always be at edges
        UpdateCellTypes(newVertices);
        
        return newVertices;
    }
    
    /// <summary>
    /// Resizes all physics arrays to match new number of cross-sections after reconnection.
    /// Preserves data for remaining cross-sections and updates interpolated sections.
    /// </summary>
    private void ResizeArrays(int newNumCrossSections, int cutoffStart, int cutoffEnd, 
                              int upstreamIdx, int downstreamIdx, int interpolatedSections, Vector3[] newVertices)
    {
        int oldNumCrossSections = _numCrossSections;
        int sectionsToRemove = cutoffEnd - cutoffStart + 1;
        
        // Create new arrays
        double[,] newU = new double[newNumCrossSections, widthResolution];
        double[,] newV = new double[newNumCrossSections, widthResolution];
        double[,] newH = new double[newNumCrossSections, widthResolution];
        double[,] newWaterDepth = new double[newNumCrossSections, widthResolution];
        int[,] newCellType = new int[newNumCrossSections, widthResolution];
        double[,] newCumulativeBankErosion = new double[newNumCrossSections, widthResolution];
        double[,] newCurrentDhDt = new double[newNumCrossSections, widthResolution];
        double[,] newInitialBedElevation = new double[newNumCrossSections, widthResolution];
        double[] newDs = new double[newNumCrossSections];
        double[] newDw = new double[newNumCrossSections];
        double[] newInitialChannelWidths = new double[newNumCrossSections];
        
        // Copy data before cutoff
        for (int i = 0; i <= upstreamIdx && i < newNumCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                newU[i, w] = u[i, w];
                newV[i, w] = v[i, w];
                newH[i, w] = h[i, w];
                newWaterDepth[i, w] = waterDepth[i, w];
                newCellType[i, w] = cellType[i, w];
                newCumulativeBankErosion[i, w] = cumulativeBankErosion[i, w];
                newCurrentDhDt[i, w] = current_dh_dt[i, w];
                if (initialBedElevation != null)
                {
                    newInitialBedElevation[i, w] = initialBedElevation[i, w];
                }
            }
            if (i < oldNumCrossSections - 1)
            {
                newDs[i] = ds[i];
            }
            if (i < oldNumCrossSections)
            {
                newDw[i] = dw[i];
                if (initialChannelWidths != null && i < initialChannelWidths.Length)
                {
                    newInitialChannelWidths[i] = initialChannelWidths[i];
                }
            }
        }
        
        // Fill interpolated sections
        int newIdx = upstreamIdx + 1;
        for (int interp = 0; interp < interpolatedSections && newIdx < newNumCrossSections; interp++)
        {
            float t = (float)(interp + 1) / (interpolatedSections + 1);
            CrossSectionData interpSection = InterpolateCrossSection(upstreamIdx, downstreamIdx, t, newVertices);
            
            for (int w = 0; w < widthResolution; w++)
            {
                newH[newIdx, w] = interpSection.h[w];
                newWaterDepth[newIdx, w] = interpSection.waterDepth[w];
                newCellType[newIdx, w] = interpSection.cellType[w];
                newU[newIdx, w] = interpSection.u[w];
                newV[newIdx, w] = interpSection.v[w];
                newCumulativeBankErosion[newIdx, w] = 0.0; // Reset for new section
                newCurrentDhDt[newIdx, w] = 0.0;
                newInitialBedElevation[newIdx, w] = interpSection.initialBedElevation[w];
            }
            
            // Interpolate spacing
            if (upstreamIdx < oldNumCrossSections - 1 && downstreamIdx < oldNumCrossSections - 1)
            {
                newDs[newIdx] = ds[upstreamIdx] * (1.0 - t) + ds[downstreamIdx] * t;
            }
            else if (upstreamIdx < oldNumCrossSections - 1)
            {
                newDs[newIdx] = ds[upstreamIdx];
            }
            
            // Interpolate width spacing and initial channel width
            if (upstreamIdx < oldNumCrossSections && downstreamIdx < oldNumCrossSections)
            {
                newDw[newIdx] = dw[upstreamIdx] * (1.0 - t) + dw[downstreamIdx] * t;
                if (initialChannelWidths != null && upstreamIdx < initialChannelWidths.Length && downstreamIdx < initialChannelWidths.Length)
                {
                    newInitialChannelWidths[newIdx] = initialChannelWidths[upstreamIdx] * (1.0 - t) + initialChannelWidths[downstreamIdx] * t;
                }
                else
                {
                    // Fallback: use interpolated width spacing
                    newInitialChannelWidths[newIdx] = newDw[newIdx] * (widthResolution - 1);
                }
            }
            else if (upstreamIdx < oldNumCrossSections)
            {
                newDw[newIdx] = dw[upstreamIdx];
                if (initialChannelWidths != null && upstreamIdx < initialChannelWidths.Length)
                {
                    newInitialChannelWidths[newIdx] = initialChannelWidths[upstreamIdx];
                }
                else
                {
                    newInitialChannelWidths[newIdx] = newDw[newIdx] * (widthResolution - 1);
                }
            }
            
            newIdx++;
        }
        
        // Copy data after cutoff (adjusting indices)
        int sourceStartIdx = downstreamIdx;
        int destStartIdx = newIdx;
        for (int i = 0; i < oldNumCrossSections - downstreamIdx && destStartIdx + i < newNumCrossSections; i++)
        {
            int sourceIdx = sourceStartIdx + i;
            int destIdx = destStartIdx + i;
            
            for (int w = 0; w < widthResolution; w++)
            {
                newU[destIdx, w] = u[sourceIdx, w];
                newV[destIdx, w] = v[sourceIdx, w];
                newH[destIdx, w] = h[sourceIdx, w];
                newWaterDepth[destIdx, w] = waterDepth[sourceIdx, w];
                newCellType[destIdx, w] = cellType[sourceIdx, w];
                newCumulativeBankErosion[destIdx, w] = cumulativeBankErosion[sourceIdx, w];
                newCurrentDhDt[destIdx, w] = current_dh_dt[sourceIdx, w];
                if (initialBedElevation != null && sourceIdx < oldNumCrossSections)
                {
                    newInitialBedElevation[destIdx, w] = initialBedElevation[sourceIdx, w];
                }
            }
            
            if (sourceIdx < oldNumCrossSections - 1)
            {
                newDs[destIdx] = ds[sourceIdx];
            }
            if (sourceIdx < oldNumCrossSections)
            {
                newDw[destIdx] = dw[sourceIdx];
                if (initialChannelWidths != null && sourceIdx < initialChannelWidths.Length)
                {
                    newInitialChannelWidths[destIdx] = initialChannelWidths[sourceIdx];
                }
                else
                {
                    newInitialChannelWidths[destIdx] = newDw[destIdx] * (widthResolution - 1);
                }
            }
        }
        
        // Update arrays
        u = newU;
        v = newV;
        h = newH;
        waterDepth = newWaterDepth;
        cellType = newCellType;
        cumulativeBankErosion = newCumulativeBankErosion;
        current_dh_dt = newCurrentDhDt;
        initialBedElevation = newInitialBedElevation;
        ds = newDs;
        dw = newDw;
        initialChannelWidths = newInitialChannelWidths;
        
        // Update numCrossSections
        _numCrossSections = newNumCrossSections;
        
        // Recalculate coordinate system with new vertices
        _coordinateSystem = new RiverCoordinateSystem(newVertices, newNumCrossSections, widthResolution);
        _coordinateSystem.CalculateFrames();
        
        // Recalculate spacing for any missing values
        CalculateSpacing(newVertices);
    }
    
    /// <summary>
    /// Helper structure for cross-section interpolation.
    /// </summary>
    private struct CrossSectionData
    {
        public Vector3[] vertices;
        public double[] h;
        public double[] waterDepth;
        public int[] cellType;
        public double[] u;
        public double[] v;
        public double[] initialBedElevation;
    }
}

/// <summary>
/// Data structure for extracted oxbow lake section.
/// </summary>
public struct OxbowSectionData
{
    public int startCrossSection;
    public int endCrossSection;
    public Vector3[] vertices;
    public double[,] h;
    public double[,] waterDepth;
    public int[,] cellType;
    public double[,] u;
    public double[,] v;
    public double[,] initialBedElevation;
}


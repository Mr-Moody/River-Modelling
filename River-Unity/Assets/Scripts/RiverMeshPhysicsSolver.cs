using UnityEngine;
using System;

/// <summary>
/// Physics solver that operates directly on the river mesh using river-local coordinates.
/// Uses u (along river) and v (across river) velocity components instead of world-space x/y.
/// </summary>
public class RiverMeshPhysicsSolver
{
    // River structure
    public readonly int numCrossSections;  // Number of cross-sections along river
    public readonly int widthResolution;    // Number of points across river width
    public readonly RiverCoordinateSystem coordinateSystem;
    
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
    public double[,] h;              // Bed elevation
    public double[,] waterDepth;    // Water depth
    public int[,] cellType;         // Cell type (FLUID, BANK, TERRAIN)
    
    // Bank migration tracking
    private double[,] cumulativeBankErosion;  // Cumulative erosion at bank cells (for migration)
    public double bankMigrationThreshold = 0.01;  // Erosion threshold before converting FLUID to BANK (meters)
    
    // Current erosion rate (stored from last ExnerEquation call)
    private double[,] current_dh_dt;  // Current bed elevation change rate
    
    // Physical spacing (approximate, will be calculated from mesh)
    private double[] ds;             // Distance between cross-sections along river
    private double[] dw;             // Distance across river at each cross-section
    
    public RiverMeshPhysicsSolver(Vector3[] riverVertices, int numCrossSections, int widthResolution,
                                  double nu = 1e-6, double rho = 1000.0, double g = 9.81,
                                  double sedimentDensity = 2650.0, double porosity = 0.4,
                                  double criticalShear = 0.05, double transportCoefficient = 0.1,
                                  double bankCriticalShear = 0.15, double bankErosionRate = 0.3)
    {
        this.numCrossSections = numCrossSections;
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
        
        // Initialize coordinate system
        coordinateSystem = new RiverCoordinateSystem(riverVertices, numCrossSections, widthResolution);
        coordinateSystem.CalculateFrames();
        
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
            
            ds[i] = Vector3.Distance(currCenter, nextCenter);
        }
        // Last cross-section uses previous spacing
        if (numCrossSections > 1)
        {
            ds[numCrossSections - 1] = ds[numCrossSections - 2];
        }
        else
        {
            ds[0] = 1.0; // Default
        }
        
        // Calculate dw (width across river at each cross-section)
        for (int i = 0; i < numCrossSections; i++)
        {
            int leftIdx = i * widthResolution;
            int rightIdx = i * widthResolution + (widthResolution - 1);
            double width = Vector3.Distance(vertices[leftIdx], vertices[rightIdx]);
            dw[i] = width / (widthResolution - 1); // Average spacing across width
        }
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
    /// Performs one Navier-Stokes time step using river-local coordinates.
    /// </summary>
    public void NavierStokesStep(float dt)
    {
        double[,] uNew = (double[,])u.Clone();
        double[,] vNew = (double[,])v.Clone();
        
        // Calculate water surface elevation
        double[,] waterSurface = new double[numCrossSections, widthResolution];
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                waterSurface[i, w] = h[i, w] + waterDepth[i, w];
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
                
                double maxChangeRate = 0.005;
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
                double u_mag = GetVelocityMagnitude(i, w);
                double roughness = (cellType[i, w] == RiverCellType.FLUID) ? n : n_bank;
                double R = Math.Max(waterDepth[i, w], 0.001);  // Minimum depth to avoid division by zero
                
                // Compute friction velocity (u_star) using Manning's roughness
                double u_star = Math.Sqrt(g * roughness * roughness * u_mag * u_mag / Math.Pow(R, 1.0 / 3.0));
                
                // Base bed/bank shear stress
                tau[i, w] = rho * u_star * u_star;
            }
        }
        
        // Add bank collision stress
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
        
        for (int i = 0; i < numCrossSections; i++)
        {
            for (int w = 0; w < widthResolution; w++)
            {
                // Compute excess shear stress for banks
                double tau_excess = Math.Max(tau[i, w] - bankCriticalShear, 0.0);
                
                // Erosion rate only applies to BANK cells
                if (cellType[i, w] != RiverCellType.FLUID)
                {
                    // Bank erosion rate scales the base transport coefficient
                    bank_erosion[i, w] = bankErosionRate * transportCoefficient * Math.Pow(tau_excess, 1.5);
                }
                else
                {
                    bank_erosion[i, w] = 0.0;
                }
            }
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
        double max_change_rate = 0.02;
        
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
                double max_change = max_change_rate * dt;
                h_change = Math.Max(-max_change, Math.Min(max_change, h_change));
                
                h_new[i, w] = h[i, w] + h_change;
                
                // Final safety clip
                h_new[i, w] = Math.Max(-10.0, Math.Min(10.0, h_new[i, w]));
            }
        }
        
        // Apply boundary conditions: maintain bank elevations and prevent edge falling
        ApplyBedElevationBoundaryConditions(h_new);
        
        // Accumulate bank erosion for migration tracking
        // Only accumulate at bank edge cells (BANK cells adjacent to FLUID cells)
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
                        // Accumulate erosion (integrate over time)
                        // Bank erosion rate is in units of length/time, so multiply by dt to get cumulative erosion
                        cumulativeBankErosion[i, w] += bank_erosion_rate[i, w] * dt;
                    }
                }
            }
        }
        
        // Process bank migration: convert FLUID cells to BANK when cumulative erosion exceeds threshold
        ProcessBankMigration();
        
        // Apply spatial smoothing to reduce numerical noise
        // Conservative smoothing to prevent mesh peaking while preserving physical behavior
        h_new = ApplySpatialSmoothing(h_new, smoothingFactor: 0.05);
        
        // Store current erosion rate for visualization
        current_dh_dt = dh_dt;
        
        return (dh_dt, h_new);
    }
    
    /// <summary>
    /// Processes bank migration by converting FLUID cells to BANK cells when cumulative erosion exceeds threshold.
    /// This simulates banks eroding and the river channel narrowing over time.
    /// The bank edge moves inward as adjacent FLUID cells are converted to BANK.
    /// </summary>
    private void ProcessBankMigration()
    {
        // Process each cross-section
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
            
            // Check if we should migrate the left bank inward
            if (leftBankEdge >= 0 && leftBankEdge < widthResolution - 1)
            {
                // Check cumulative erosion at the bank edge
                if (cumulativeBankErosion[i, leftBankEdge] >= bankMigrationThreshold)
                {
                    // Find the adjacent FLUID cell (moving inward from left)
                    int nextFluidW = leftBankEdge + 1;
                    if (nextFluidW < widthResolution && cellType[i, nextFluidW] == RiverCellType.FLUID)
                    {
                        // Convert FLUID to BANK: bank migrates inward
                        cellType[i, nextFluidW] = RiverCellType.BANK;
                        waterDepth[i, nextFluidW] = 0.0;
                        u[i, nextFluidW] = 0.0;
                        v[i, nextFluidW] = 0.0;
                        
                        // Transfer cumulative erosion to new bank position (carry over excess)
                        double excessErosion = cumulativeBankErosion[i, leftBankEdge] - bankMigrationThreshold;
                        cumulativeBankErosion[i, nextFluidW] = excessErosion;
                        cumulativeBankErosion[i, leftBankEdge] = 0.0;
                    }
                }
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
            
            // Check if we should migrate the right bank inward
            if (rightBankEdge > 0 && rightBankEdge < widthResolution)
            {
                // Check cumulative erosion at the bank edge
                if (cumulativeBankErosion[i, rightBankEdge] >= bankMigrationThreshold)
                {
                    // Find the adjacent FLUID cell (moving inward from right)
                    int nextFluidW = rightBankEdge - 1;
                    if (nextFluidW >= 0 && cellType[i, nextFluidW] == RiverCellType.FLUID)
                    {
                        // Convert FLUID to BANK: bank migrates inward
                        cellType[i, nextFluidW] = RiverCellType.BANK;
                        waterDepth[i, nextFluidW] = 0.0;
                        u[i, nextFluidW] = 0.0;
                        v[i, nextFluidW] = 0.0;
                        
                        // Transfer cumulative erosion to new bank position (carry over excess)
                        double excessErosion = cumulativeBankErosion[i, rightBankEdge] - bankMigrationThreshold;
                        cumulativeBankErosion[i, nextFluidW] = excessErosion;
                        cumulativeBankErosion[i, rightBankEdge] = 0.0;
                    }
                }
            }
        }
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
}


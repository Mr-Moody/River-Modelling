using UnityEngine;
using System;

/// <summary>
/// Physics solver for coupled Shallow Water and Exner equations (river flow and sediment transport).
/// This class handles the core finite difference calculations.
/// NOTE: The loops used for finite difference operations are not optimized (not using Burst/Jobs) 
/// and will be slow for large grids.
/// </summary>
public class PhysicsSolver
{
    // --- Physics Parameters ---
    public readonly int gridWidth;
    public readonly int gridHeight;
    public readonly float cellSize;
    public readonly double nu;              // Kinematic viscosity (m^2/s)
    public readonly double rho;             // Water density (kg/m^3)
    public readonly double g;               // Gravitational acceleration (m/s^2)
    public readonly double sedimentDensity; // Sediment density (kg/m^3)
    public readonly double porosity;        // Bed porosity
    public readonly double criticalShear;   // Critical shear stress for sediment motion (Pa)
    public readonly double transportCoefficient;
    public readonly int bankWidth;         // Width of bank regions (in cells)
    public readonly double bankErosionRate;
    public readonly double bankCriticalShear;
    public readonly int terrainWidth;

    // --- Grid Dimensions ---
    public readonly int nx; // nx = gridWidth + 1
    public readonly int ny; // ny = gridHeight + 1

    // --- Flow Fields (Data Arrays) ---
    public double[,] u;              // x-velocity
    public double[,] v;              // y-velocity
    public double[,] p;              // Pressure
    public double[,] h;              // Bed elevation (Z-coordinate of the river bed)
    public double[,] waterDepth;     // Water depth (height of water column)
    public int[,] cellType;          // Cell type mask (FLUID, BANK, TERRAIN)

    public PhysicsSolver(int gridWidth, int gridHeight, float cellSize,
                         double nu = 1e-6, double rho = 1000.0, double g = 9.81,
                         double sedimentDensity = 2650.0, double porosity = 0.4,
                         double criticalShear = 0.05, double transportCoefficient = 0.1,
                         int bankWidth = 2, double bankErosionRate = 0.3, double bankCriticalShear = 0.15,
                         int terrainWidth = 3)
    {
        this.gridWidth = gridWidth;
        this.gridHeight = gridHeight;
        this.cellSize = cellSize;
        this.nu = nu;
        this.rho = rho;
        this.g = g;
        this.sedimentDensity = sedimentDensity;
        this.porosity = porosity;
        this.criticalShear = criticalShear;
        this.transportCoefficient = transportCoefficient;
        this.bankWidth = bankWidth;
        this.bankErosionRate = bankErosionRate;
        this.bankCriticalShear = bankCriticalShear;
        this.terrainWidth = terrainWidth;

        this.nx = gridWidth + 1;
        this.ny = gridHeight + 1;

        u = new double[nx, ny];
        v = new double[nx, ny];
        p = new double[nx, ny];
        h = new double[nx, ny];
        waterDepth = new double[nx, ny];
        cellType = new int[nx, ny];

        InitialiseCellTypes();
        InitialiseFields();
    }

    // --- Initialization Methods ---

    private void InitialiseCellTypes()
    {
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                if (j < terrainWidth || j >= ny - terrainWidth)
                {
                    cellType[i, j] = RiverCellType.TERRAIN;
                }
                else if (j < terrainWidth + bankWidth || j >= ny - terrainWidth - bankWidth)
                {
                    cellType[i, j] = RiverCellType.BANK;
                }
                else
                {
                    cellType[i, j] = RiverCellType.FLUID;
                }
            }
        }
    }

    private void InitialiseFields()
    {
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                if (cellType[i, j] == RiverCellType.FLUID)
                {
                    waterDepth[i, j] = 0.5;
                    u[i, j] = 0.1;
                    v[i, j] = 0.0;
                }
                else
                {
                    waterDepth[i, j] = 0.0;
                    u[i, j] = 0.0;
                    v[i, j] = 0.0;
                }
                h[i, j] = 0.0;
                p[i, j] = 0.0;
            }
        }
    }

    public void UpdateBedElevation(double[,] bedElevations)
    {
        if (bedElevations.GetLength(0) != nx || bedElevations.GetLength(1) != ny)
        {
            throw new ArgumentException($"Bed elevation shape ({bedElevations.GetLength(0)}, {bedElevations.GetLength(1)}) does not match grid dimensions ({nx}, {ny})");
        }
        this.h = bedElevations;
    }

    // --- Core Solver Methods ---

    public void NavierStokesStep(float dt)
    {
        double[,] uNew = (double[,])u.Clone();
        double[,] vNew = (double[,])v.Clone();
        double dx = cellSize;
        double dy = cellSize;

        // 1. Calculate water surface elevation for pressure gradient
        double[,] waterSurface = new double[nx, ny];
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                waterSurface[i, j] = h[i, j] + waterDepth[i, j];
            }
        }

        // 2. Solve velocity equations (Euler integration for interior cells)
        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                // Advection Terms (Upwind)
                double u_adv_x = u[i, j] > 0 ? u[i, j] * (u[i, j] - u[i - 1, j]) / dx : u[i, j] * (u[i + 1, j] - u[i, j]) / dx;
                double u_adv_y = v[i, j] > 0 ? v[i, j] * (u[i, j] - u[i, j - 1]) / dy : v[i, j] * (u[i, j + 1] - u[i, j]) / dy;
                double v_adv_x = u[i, j] > 0 ? u[i, j] * (v[i, j] - v[i - 1, j]) / dx : u[i, j] * (v[i + 1, j] - v[i, j]) / dx;
                double v_adv_y = v[i, j] > 0 ? v[i, j] * (v[i, j] - v[i, j - 1]) / dy : v[i, j] * (v[i, j + 1] - v[i, j]) / dy;

                // Pressure Gradient (Water Surface Slope, Central Diff)
                double p_grad_x = -g * (waterSurface[i + 1, j] - waterSurface[i - 1, j]) / (2 * dx);
                double p_grad_y = -g * (waterSurface[i, j + 1] - waterSurface[i, j - 1]) / (2 * dy);

                // Viscous Diffusion (Laplacian, Central Diff)
                double u_laplacian = ((u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / (dx * dx)) +
                                     ((u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / (dy * dy));
                double v_laplacian = ((v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / (dx * dx)) +
                                     ((v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) / (dy * dy));

                // Bed Slope Effect
                double h_slope_x = -g * waterDepth[i, j] * (h[i + 1, j] - h[i - 1, j]) / (2 * dx);
                double h_slope_y = -g * waterDepth[i, j] * (h[i, j + 1] - h[i, j - 1]) / (2 * dy);

                // Update Velocities
                uNew[i, j] = u[i, j] + dt * (-u_adv_x - u_adv_y + p_grad_x + nu * u_laplacian + h_slope_x);
                vNew[i, j] = v[i, j] + dt * (-v_adv_x - v_adv_y + p_grad_y + nu * v_laplacian + h_slope_y);
            }
        }

        // 3. Apply Velocity Boundary Conditions (Retain current boundary values)
        for (int j = 0; j < ny; j++)
        {
            uNew[0, j] = u[0, j];
            uNew[nx - 1, j] = u[nx - 1, j];
        }
        for (int i = 0; i < nx; i++)
        {
            vNew[i, 0] = v[i, 0];
            vNew[i, ny - 1] = v[i, ny - 1];
        }

        // 4. Update Water Depth (Continuity Equation)
        waterDepth = SolveContinuityEquation(waterDepth, uNew, vNew, dt);

        // 5. Final assignment
        u = uNew;
        v = vNew;
    }

    public double[,] SolveContinuityEquation(double[,] h_w, double[,] u, double[,] v, float dt)
    {
        double[,] h_w_new = (double[,])h_w.Clone();
        double dx = cellSize;
        double dy = cellSize;
        double equilibriumDepth = 0.5;
        double maxVariation = 0.01;
        double relaxationTime = 0.01;
        double minDepth = 0.01;

        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                double div_u = (u[i + 1, j] - u[i - 1, j]) / (2 * dx) +
                               (v[i, j + 1] - v[i, j - 1]) / (2 * dy);

                double relaxationTerm = -(h_w[i, j] - equilibriumDepth) / relaxationTime;
                double flowAdjustment = -div_u * 0.01;
                flowAdjustment = Math.Max(-maxVariation, Math.Min(maxVariation, flowAdjustment));

                double dh_w_dt = 0.95 * relaxationTerm + 0.05 * flowAdjustment;

                double maxChangeRate = 0.005;
                double maxChange = maxChangeRate * equilibriumDepth;
                dh_w_dt = Math.Max(-maxChange / dt, Math.Min(maxChange / dt, dh_w_dt));

                h_w_new[i, j] = h_w[i, j] + dh_w_dt * dt;
            }
        }

        // Apply boundary conditions and constraints
        // Inflow boundary (x=0)
        double inflowDepth = 0.5;
        for (int j = 1; j < ny - 1; j++) h_w_new[0, j] = inflowDepth;
        // Outflow boundary (x=nx-1)
        for (int j = 1; j < ny - 1; j++) h_w_new[nx - 1, j] = h_w_new[nx - 2, j];
        // Lateral boundaries (y=0, y=ny-1)
        for (int i = 1; i < nx - 1; i++) h_w_new[i, 0] = h_w_new[i, 1];
        for (int i = 1; i < nx - 1; i++) h_w_new[i, ny - 1] = h_w_new[i, ny - 2];

        // Corners: Simplification using averages
        h_w_new[0, 0] = (h_w_new[0, 1] + h_w_new[1, 0]) / 2;
        h_w_new[0, ny - 1] = (h_w_new[0, ny - 2] + h_w_new[1, ny - 1]) / 2;
        h_w_new[nx - 1, 0] = (h_w_new[nx - 1, 1] + h_w_new[nx - 2, 0]) / 2;
        h_w_new[nx - 1, ny - 1] = (h_w_new[nx - 1, ny - 2] + h_w_new[nx - 2, ny - 1]) / 2;


        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                if (cellType[i, j] != RiverCellType.FLUID)
                {
                    h_w_new[i, j] = 0.0;
                }
                else
                {
                    h_w_new[i, j] = Math.Max(h_w_new[i, j], minDepth);
                    h_w_new[i, j] = Math.Min(h_w_new[i, j], 2.0); // Max depth clip
                }
            }
        }

        return h_w_new;
    }

    // --- Sediment and Erosion Methods ---

    public double[,] ComputeShearStress()
    {
        double[,] tau = new double[nx, ny];
        double n = 0.03;
        double n_bank = 0.05;

        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                double u_mag = Math.Sqrt(u[i, j] * u[i, j] + v[i, j] * v[i, j]);
                double roughness = (cellType[i, j] == RiverCellType.FLUID) ? n : n_bank;
                double R = waterDepth[i, j];

                if (R < 0.001) R = 0.001;

                // Compute friction velocity (u_star) using Manning's roughness
                double u_star = Math.Sqrt(g * roughness * roughness * u_mag * u_mag / Math.Pow(R, 1.0 / 3.0));

                tau[i, j] = rho * u_star * u_star;
            }
        }

        // Add bank collision stress
        tau = AddBankCollisionStress(tau);

        return tau;
    }

    private double[,] AddBankCollisionStress(double[,] tau_base)
    {
        double[,] tau = (double[,])tau_base.Clone();
        double collisionFactor = 1.5;

        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                if (cellType[i, j] == RiverCellType.BANK)
                {
                    // Check left neighbor (j-1) for fluid flowing right (+v)
                    if (j > 0 && cellType[i, j - 1] == RiverCellType.FLUID && v[i, j - 1] > 0)
                    {
                        double v_normal = Math.Abs(v[i, j - 1]);
                        double collisionStress = rho * collisionFactor * v_normal * v_normal;
                        tau[i, j] += collisionStress;
                    }
                    // Check right neighbor (j+1) for fluid flowing left (-v)
                    if (j < ny - 1 && cellType[i, j + 1] == RiverCellType.FLUID && v[i, j + 1] < 0)
                    {
                        double v_normal = Math.Abs(v[i, j + 1]);
                        double collisionStress = rho * collisionFactor * v_normal * v_normal;
                        tau[i, j] += collisionStress;
                    }
                }
            }
        }
        return tau;
    }

    public double[,] SedimentFlux(double[,] tau)
    {
        double[,] qs_magnitude = new double[nx, ny];
        double dx = cellSize;
        double dy = cellSize;

        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                // Compute excess shear stress
                double tau_excess = Math.Max(tau[i, j] - criticalShear, 0.0);

                // Base sediment flux (Meyer-Peter and Müller type)
                qs_magnitude[i, j] = transportCoefficient * Math.Pow(tau_excess, 1.5);
            }
        }

        // Apply bed slope effect (uses central difference for interior, forward/backward for boundaries)
        double[,] h_slope_x = new double[nx, ny];
        double[,] h_slope_y = new double[nx, ny];

        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                // X-slope
                if (i > 0 && i < nx - 1) h_slope_x[i, j] = (h[i + 1, j] - h[i - 1, j]) / (2 * dx);
                else if (i == 0) h_slope_x[i, j] = (h[1, j] - h[0, j]) / dx;
                else h_slope_x[i, j] = (h[nx - 1, j] - h[nx - 2, j]) / dx;

                // Y-slope
                if (j > 0 && j < ny - 1) h_slope_y[i, j] = (h[i, j + 1] - h[i, j - 1]) / (2 * dy);
                else if (j == 0) h_slope_y[i, j] = (h[i, 1] - h[i, 0]) / dy;
                else h_slope_y[i, j] = (h[i, ny - 1] - h[i, ny - 2]) / dy;

                double slope_mag = Math.Sqrt(h_slope_x[i, j] * h_slope_x[i, j] + h_slope_y[i, j] * h_slope_y[i, j]);

                // Slope factor: reduces transport on steep slopes
                double max_slope = 0.3;
                double slope_factor = 1.0 / (1.0 + slope_mag / max_slope);

                qs_magnitude[i, j] *= slope_factor;
            }
        }
        return qs_magnitude;
    }

    public (double[,] qs_x, double[,] qs_y) ComputeSedimentFluxVector(double[,] tau)
    {
        double[,] qs_magnitude = SedimentFlux(tau);
        double[,] qs_x = new double[nx, ny];
        double[,] qs_y = new double[nx, ny];
        double dx = cellSize;
        double dy = cellSize;

        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                double u_mag = Math.Sqrt(u[i, j] * u[i, j] + v[i, j] * v[i, j]);

                // Flow Direction Vector (Normalized)
                double u_norm = u_mag > 1e-10 ? u[i, j] / u_mag : 0.0;
                double v_norm = u_mag > 1e-10 ? v[i, j] / u_mag : 0.0;

                // Slope Direction Vector (Negative Gradient = Downhill)
                // Use the same finite difference scheme as in SedimentFlux
                double h_slope_x_raw;
                double h_slope_y_raw;

                // X-slope (Central/Forward/Backward)
                if (i > 0 && i < nx - 1) h_slope_x_raw = (h[i + 1, j] - h[i - 1, j]) / (2 * dx);
                else if (i == 0) h_slope_x_raw = (h[1, j] - h[0, j]) / dx;
                else h_slope_x_raw = (h[nx - 1, j] - h[nx - 2, j]) / dx;

                // Y-slope
                if (j > 0 && j < ny - 1) h_slope_y_raw = (h[i, j + 1] - h[i, j - 1]) / (2 * dy);
                else if (j == 0) h_slope_y_raw = (h[i, 1] - h[i, 0]) / dy;
                else h_slope_y_raw = (h[i, ny - 1] - h[i, ny - 2]) / dy;

                // Downhill direction = negative gradient
                double h_slope_x = -h_slope_x_raw;
                double h_slope_y = -h_slope_y_raw;

                double slope_mag = Math.Sqrt(h_slope_x * h_slope_x + h_slope_y * h_slope_y);
                double slope_x_norm = slope_mag > 1e-10 ? h_slope_x / slope_mag : 0.0;
                double slope_y_norm = slope_mag > 1e-10 ? h_slope_y / slope_mag : 0.0;

                // Weighted Average of Flow (70%) and Slope (30%)
                double slopeFactor = 0.3;
                double flowFactor = 1.0 - slopeFactor;

                double u_combined = flowFactor * u_norm + slopeFactor * slope_x_norm;
                double v_combined = flowFactor * v_norm + slopeFactor * slope_y_norm;

                // Normalize Combined Direction
                double combined_mag = Math.Sqrt(u_combined * u_combined + v_combined * v_combined);
                double u_final_norm = combined_mag > 1e-10 ? u_combined / combined_mag : 0.0;
                double v_final_norm = combined_mag > 1e-10 ? v_combined / combined_mag : 0.0;

                // Compute final flux vector
                qs_x[i, j] = qs_magnitude[i, j] * u_final_norm;
                qs_y[i, j] = qs_magnitude[i, j] * v_final_norm;
            }
        }

        return (qs_x, qs_y);
    }

    public double[,] ComputeBankErosion(double[,] tau)
    {
        double[,] bank_erosion = new double[nx, ny];

        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                // Compute excess shear stress for banks
                double tau_excess = Math.Max(tau[i, j] - bankCriticalShear, 0.0);

                // Erosion rate only applies to BANK or TERRAIN cells
                if (cellType[i, j] != RiverCellType.FLUID)
                {
                    bank_erosion[i, j] = transportCoefficient * Math.Pow(tau_excess, 1.5);
                }
                else
                {
                    bank_erosion[i, j] = 0.0;
                }
            }
        }
        return bank_erosion;
    }

    public void ConvertTerrainToBank(double[,] h_new, double[,] tau)
    {
        double min_shear_for_conversion = 0.1;

        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                if (cellType[i, j] == RiverCellType.TERRAIN)
                {
                    bool adjacentToFluid = false;

                    // Check neighbors for fluid cells
                    for (int di = -1; di <= 1; di++)
                    {
                        for (int dj = -1; dj <= 1; dj++)
                        {
                            if (di == 0 && dj == 0) continue;
                            int ni = i + di;
                            int nj = j + dj;

                            if (ni >= 0 && ni < nx && nj >= 0 && nj < ny)
                            {
                                if (cellType[ni, nj] == RiverCellType.FLUID)
                                {
                                    adjacentToFluid = true;
                                    break;
                                }
                            }
                        }
                        if (adjacentToFluid) break;
                    }

                    // Conversion criteria
                    if (adjacentToFluid && tau[i, j] > min_shear_for_conversion)
                    {
                        cellType[i, j] = RiverCellType.BANK;
                    }
                    else if (waterDepth[i, j] > 0.01) // Water has flowed into terrain
                    {
                        cellType[i, j] = RiverCellType.BANK;
                    }
                }
            }
        }
    }

    public (double[,] dh_dt, double[,] h_new) ExnerEquation(double[,] qs_x, double[,] qs_y, float dt)
    {
        double[,] h_new = (double[,])h.Clone();
        double dx = cellSize;
        double dy = cellSize;

        // 1. Apply flux boundary conditions (Zero Flux at all boundaries)
        double[,] qs_x_bc = (double[,])qs_x.Clone();
        double[,] qs_y_bc = (double[,])qs_y.Clone();

        for (int j = 0; j < ny; j++) { qs_x_bc[0, j] = 0.0; qs_x_bc[nx - 1, j] = 0.0; }
        for (int i = 0; i < nx; i++) { qs_y_bc[i, 0] = 0.0; qs_y_bc[i, ny - 1] = 0.0; }

        // 2. Compute Divergence (div_qs)
        double[,] div_qs = new double[nx, ny];

        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                double dqsx_dx, dqsy_dy;

                // X-Divergence (Uses appropriate difference based on boundary)
                if (i > 0 && i < nx - 1) dqsx_dx = (qs_x_bc[i + 1, j] - qs_x_bc[i - 1, j]) / (2 * dx); // Central
                else if (i == 0) dqsx_dx = (qs_x_bc[1, j] - qs_x_bc[0, j]) / dx; // Forward (since qs_x[0] is zero flux)
                else dqsx_dx = (qs_x_bc[nx - 1, j] - qs_x_bc[nx - 2, j]) / dx; // Backward (since qs_x[nx-1] is zero flux)

                // Y-Divergence
                if (j > 0 && j < ny - 1) dqsy_dy = (qs_y_bc[i, j + 1] - qs_y_bc[i, j - 1]) / (2 * dy); // Central
                else if (j == 0) dqsy_dy = (qs_y_bc[i, 1] - qs_y_bc[i, 0]) / dy;
                else dqsy_dy = (qs_y_bc[i, ny - 1] - qs_y_bc[i, ny - 2]) / dy;

                div_qs[i, j] = dqsx_dx + dqsy_dy;
            }
        }

        // Corners are not explicitly handled in the Python code boundary loops; they fall out as approximations. 
        // We set them to zero flux explicitly as in the original code's final assignment:
        div_qs[0, 0] = div_qs[0, ny - 1] = div_qs[nx - 1, 0] = div_qs[nx - 1, ny - 1] = 0.0;

        // 3. Exner Equation: dh/dt = -1/(1-porosity) * div(qs) + bank_erosion
        double[,] dh_dt = new double[nx, ny];
        double porosityFactor = 1.0 / (1.0 - porosity);
        double max_change_rate = 0.02;

        double[,] tau = ComputeShearStress();
        double[,] bank_erosion_rate = ComputeBankErosion(tau);

        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                dh_dt[i, j] = -porosityFactor * div_qs[i, j];

                // Add bank erosion rate (only where applicable)
                if (cellType[i, j] != RiverCellType.FLUID)
                {
                    dh_dt[i, j] -= porosityFactor * bank_erosion_rate[i, j];
                }

                // 4. Update Bed Elevation and Apply Stability Constraint
                double h_change = dh_dt[i, j] * dt;

                double max_change = max_change_rate * dt;
                h_change = Math.Max(-max_change, Math.Min(max_change, h_change));

                h_new[i, j] = h[i, j] + h_change;

                // Final safety clip
                h_new[i, j] = Math.Max(-10.0, Math.Min(10.0, h_new[i, j]));
            }
        }

        // 5. Convert TERRAIN to BANK based on erosion/water flow
        ConvertTerrainToBank(h_new, tau);

        return (dh_dt, h_new);
    }
}
"""
Physics module for Navier-Stokes and Exner coupled equations.
Implements shallow water equations for river flow with sediment transport.
"""

import numpy as np

# Cell type constants
FLUID = 0
BANK_LEFT = 1
BANK_RIGHT = 2


class PhysicsSolver:
    """
    Physics solver for coupled Navier-Stokes and Exner equations.
    """
    
    def __init__(self, grid_width, grid_height, cell_size, 
                 nu=1e-6, rho=1000.0, g=9.81, 
                 sediment_density=2650.0, porosity=0.4,
                 critical_shear=0.05, transport_coefficient=0.1,
                 bank_width=2, bank_erosion_rate=0.3, bank_critical_shear=0.15):
        """
        Initialize physics parameters.
        
        Args:
            grid_width: Number of cells in x direction
            grid_height: Number of cells in y direction
            cell_size: Size of each cell (meters)
            nu: Kinematic viscosity (m^2/s)
            rho: Water density (kg/m^3)
            g: Gravitational acceleration (m/s^2)
            sediment_density: Sediment density (kg/m^3)
            porosity: Bed porosity (dimensionless)
            critical_shear: Critical shear stress for sediment motion (Pa)
            transport_coefficient: Sediment transport coefficient
            bank_width: Width of bank regions on each side (in cells)
            bank_erosion_rate: Bank erosion rate relative to bed (0-1)
            bank_critical_shear: Critical shear stress for bank erosion (Pa)
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.nu = nu  # Kinematic viscosity
        self.rho = rho  # Water density
        self.g = g  # Gravity
        self.sediment_density = sediment_density
        self.porosity = porosity
        self.critical_shear = critical_shear
        self.transport_coefficient = transport_coefficient
        self.bank_width = bank_width
        self.bank_erosion_rate = bank_erosion_rate
        self.bank_critical_shear = bank_critical_shear
        
        # Grid dimensions (vertices)
        self.nx = grid_width + 1
        self.ny = grid_height + 1
        
        # Initialize flow fields
        self.u = np.zeros((self.nx, self.ny))  # x-velocity
        self.v = np.zeros((self.nx, self.ny))  # y-velocity
        self.p = np.zeros((self.nx, self.ny))  # Pressure
        self.h = np.zeros((self.nx, self.ny))  # Bed elevation
        self.water_depth = np.zeros((self.nx, self.ny))  # Water depth
        
        # Cell type mask: FLUID=0, BANK_LEFT=1, BANK_RIGHT=2
        self.cell_type = np.zeros((self.nx, self.ny), dtype=int)
        self._initialize_cell_types()
        
        # Initialize with some initial conditions
        self._initialize_fields()
        
    def _initialize_cell_types(self):
        """Initialize cell type mask for banks."""
        # Mark left bank (first bank_width columns)
        self.cell_type[:, :self.bank_width] = BANK_LEFT
        # Mark right bank (last bank_width columns)
        self.cell_type[:, -self.bank_width:] = BANK_RIGHT
        # Rest are FLUID cells
        
    def _initialize_fields(self):
        """Initialize flow fields with default conditions."""
        # Initialize bed elevation (h) - will be updated from mesh
        # Initialize water depth (assuming constant initial depth)
        self.water_depth.fill(0.5)  # 0.5m initial water depth
        
        # Initialize velocity fields (small initial flow)
        self.u.fill(0.1)  # 0.1 m/s initial flow in x direction
        self.v.fill(0.0)  # No initial flow in y direction
        
        # Set velocity to zero in bank cells
        self.u[self.cell_type != FLUID] = 0.0
        self.v[self.cell_type != FLUID] = 0.0
        
    def update_bed_elevation(self, bed_elevations):
        """
        Update bed elevation from mesh points.
        
        Args:
            bed_elevations: Array of bed elevations (z-coordinates) from mesh points
        """
        if bed_elevations.shape == (self.nx, self.ny):
            self.h = bed_elevations.copy()
        elif bed_elevations.shape == (self.nx * self.ny,):
            # Reshape from flattened array
            self.h = bed_elevations.reshape(self.nx, self.ny)
        else:
            raise ValueError(f"Bed elevation shape {bed_elevations.shape} does not match grid dimensions")
    
    def navier_stokes_step(self, u, v, p, h, dt):
        """
        Solve one step of Navier-Stokes equations using shallow water approximation.
        
        Args:
            u: x-velocity field (nx, ny)
            v: y-velocity field (nx, ny)
            p: pressure field (nx, ny)
            h: bed elevation (nx, ny)
            dt: time step
            
        Returns:
            u_new, v_new, p_new: Updated velocity and pressure fields
        """
        nx, ny = u.shape
        dx = self.cell_size
        dy = self.cell_size
        
        # Calculate water depth (assuming water surface at constant elevation)
        # For simplicity, use constant water depth or calculate from h
        water_surface = h + self.water_depth
        
        # Compute gradients using finite differences
        # For u-velocity equation
        u_new = u.copy()
        v_new = v.copy()
        
        # Vectorized computation for better performance
        # Create masks for upwind differencing
        u_pos_x = u[1:-1, 1:-1] > 0
        u_neg_x = ~u_pos_x
        v_pos_y = v[1:-1, 1:-1] > 0
        v_neg_y = ~v_pos_y
        
        # U-velocity equation
        # Advection (upwind differencing)
        u_adv_x = np.where(u_pos_x,
                          u[1:-1, 1:-1] * (u[1:-1, 1:-1] - u[:-2, 1:-1]) / dx,
                          u[1:-1, 1:-1] * (u[2:, 1:-1] - u[1:-1, 1:-1]) / dx)
        u_adv_y = np.where(v_pos_y,
                          v[1:-1, 1:-1] * (u[1:-1, 1:-1] - u[1:-1, :-2]) / dy,
                          v[1:-1, 1:-1] * (u[1:-1, 2:] - u[1:-1, 1:-1]) / dy)
        
        # Pressure gradient (from water surface)
        p_grad_x = -self.g * (water_surface[2:, 1:-1] - water_surface[:-2, 1:-1]) / (2 * dx)
        
        # Viscous diffusion
        u_laplacian = (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx**2) + \
                     (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy**2)
        
        # Bed slope effect (drives flow downhill)
        h_slope_x = -self.g * self.water_depth[1:-1, 1:-1] * (h[2:, 1:-1] - h[:-2, 1:-1]) / (2 * dx)
        
        # Update u (interior points only)
        u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (u_adv_x + u_adv_y + p_grad_x + self.nu * u_laplacian + h_slope_x)
        
        # V-velocity equation
        # Advection
        v_adv_x = np.where(u_pos_x,
                          u[1:-1, 1:-1] * (v[1:-1, 1:-1] - v[:-2, 1:-1]) / dx,
                          u[1:-1, 1:-1] * (v[2:, 1:-1] - v[1:-1, 1:-1]) / dx)
        v_adv_y = np.where(v_pos_y,
                          v[1:-1, 1:-1] * (v[1:-1, 1:-1] - v[1:-1, :-2]) / dy,
                          v[1:-1, 1:-1] * (v[1:-1, 2:] - v[1:-1, 1:-1]) / dy)
        
        # Pressure gradient
        p_grad_y = -self.g * (water_surface[1:-1, 2:] - water_surface[1:-1, :-2]) / (2 * dy)
        
        # Viscous diffusion
        v_laplacian = (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) / (dx**2) + \
                     (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2]) / (dy**2)
        
        # Bed slope effect
        h_slope_y = -self.g * self.water_depth[1:-1, 1:-1] * (h[1:-1, 2:] - h[1:-1, :-2]) / (2 * dy)
        
        # Update v
        v_new[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * (v_adv_x + v_adv_y + p_grad_y + self.nu * v_laplacian + h_slope_y)
        
        # Apply boundary conditions (will be updated later, but set boundaries here)
        # Boundaries are handled in apply_boundary_conditions, but initialize here
        u_new[0, :] = u[0, :]  # Will be set by boundary conditions
        u_new[-1, :] = u[-1, :]
        v_new[:, 0] = v[:, 0]
        v_new[:, -1] = v[:, -1]
        
        # Validate and clean NaN/Inf values
        u_new = np.nan_to_num(u_new, nan=0.0, posinf=1e3, neginf=-1e3)
        v_new = np.nan_to_num(v_new, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # Clip velocity to reasonable bounds
        max_velocity = 10.0  # Maximum velocity (m/s)
        u_new = np.clip(u_new, -max_velocity, max_velocity)
        v_new = np.clip(v_new, -max_velocity, max_velocity)
        
        # Update pressure using continuity equation (simplified)
        # For incompressible flow: div(u) = 0
        # Here we use a simple pressure correction
        p_new = p.copy()
        p_new = np.nan_to_num(p_new, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Update stored fields
        self.u = u_new
        self.v = v_new
        self.p = p_new
        
        return u_new, v_new, p_new
    
    def compute_shear_stress(self, u, v, h):
        """
        Compute bed and bank shear stress from velocity field.
        Includes enhanced stress at bank walls due to water-wall collisions.
        
        Args:
            u: x-velocity field
            v: y-velocity field
            h: bed elevation (for depth calculation)
            
        Returns:
            tau: Bed/bank shear stress magnitude (nx, ny)
        """
        # Manning's roughness coefficient (typical for rivers)
        n = 0.03  # Roughness coefficient for bed
        n_bank = 0.05  # Higher roughness for banks
        
        # Compute velocity magnitude
        u_mag = np.sqrt(u**2 + v**2)
        
        # Compute water depth from bed elevation
        # Assume constant water surface elevation (or use stored value)
        if hasattr(self, 'water_surface_elevation'):
            water_surface = self.water_surface_elevation
        else:
            # Calculate from current bed elevation
            water_surface = np.max(h) + 0.5  # Reference water surface
        water_depth = np.maximum(water_surface - h, 0.01)  # Minimum depth
        
        # Use different roughness for banks vs bed
        roughness = np.where(self.cell_type == FLUID, n, n_bank)
        
        # Bed shear stress using Manning's equation and friction factor
        # tau = rho * u*^2, where u* is friction velocity
        # Using Manning-Strickler relation: u* = (g * n^2 * U^2 / R^(1/3))^(1/2)
        # Simplified: tau = rho * g * n^2 * U^2 / R^(1/3)
        
        # Hydraulic radius (approximate as water depth for wide channels)
        R = np.maximum(water_depth, 0.01)  # Minimum depth to avoid division by zero
        
        # Friction velocity
        u_star = np.sqrt(self.g * roughness**2 * u_mag**2 / (R**(1/3)))
        
        # Base bed/bank shear stress
        tau = self.rho * u_star**2
        
        # Enhanced stress at bank walls due to water-wall collisions
        # Water hitting bank walls creates additional pressure and shear
        tau = self._add_bank_collision_stress(tau, u, v, h)
        
        return tau
    
    def _add_bank_collision_stress(self, tau_base, u, v, h):
        """
        Add additional shear stress at bank walls due to water-wall collisions.
        Models the impact of water hitting vertical bank walls.
        
        Args:
            tau_base: Base shear stress from flow
            u: x-velocity field
            v: y-velocity field
            h: bed elevation
            
        Returns:
            tau: Enhanced shear stress including collision effects
        """
        tau = tau_base.copy()
        nx, ny = tau.shape
        
        # Collision enhancement factor (how much extra stress from impacts)
        collision_factor = 1.5
        
        # Check for water flow toward banks (collision detection)
        for i in range(nx):
            for j in range(ny):
                cell_type = self.cell_type[i, j]
                
                # Left bank: water flowing from right (positive v) creates collision
                if cell_type == BANK_LEFT and j < ny - 1:
                    # Check if water is flowing toward this bank
                    if self.cell_type[i, j+1] == FLUID and v[i, j+1] < 0:
                        # Water velocity component normal to bank
                        v_normal = abs(v[i, j+1])
                        # Additional stress from collision (proportional to velocity squared)
                        collision_stress = self.rho * collision_factor * v_normal**2
                        tau[i, j] += collision_stress
                
                # Right bank: water flowing from left (negative v) creates collision
                elif cell_type == BANK_RIGHT and j > 0:
                    # Check if water is flowing toward this bank
                    if self.cell_type[i, j-1] == FLUID and v[i, j-1] > 0:
                        # Water velocity component normal to bank
                        v_normal = abs(v[i, j-1])
                        # Additional stress from collision
                        collision_stress = self.rho * collision_factor * v_normal**2
                        tau[i, j] += collision_stress
        
        return tau
    
    def sediment_flux(self, tau, h=None):
        """
        Compute sediment transport flux from bed shear stress.
        Uses Meyer-Peter and Müller type formulation with bed slope effects.
        
        Args:
            tau: Bed shear stress (nx, ny)
            h: Bed elevation (optional, for slope effects)
            
        Returns:
            qs: Sediment flux magnitude (nx, ny)
        """
        # Excess shear stress
        tau_excess = np.maximum(tau - self.critical_shear, 0.0)
        
        # Base sediment flux (Meyer-Peter and Müller type)
        # qs = K * (tau_excess)^(3/2)
        qs_magnitude = self.transport_coefficient * (tau_excess ** 1.5)
        
        # Apply bed slope effect (reduces transport uphill, increases downhill)
        # This helps prevent unbounded growth of peaks
        if h is not None:
            nx, ny = h.shape
            dx = self.cell_size
            dy = self.cell_size
            
            # Compute bed slope magnitude
            # Use central differences for interior, forward/backward at boundaries
            h_slope_x = np.zeros_like(h)
            h_slope_y = np.zeros_like(h)
            
            # X-direction slope
            h_slope_x[1:-1, :] = (h[2:, :] - h[:-2, :]) / (2 * dx)
            h_slope_x[0, :] = (h[1, :] - h[0, :]) / dx  # Forward difference at boundary
            h_slope_x[-1, :] = (h[-1, :] - h[-2, :]) / dx  # Backward difference at boundary
            
            # Y-direction slope
            h_slope_y[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / (2 * dy)
            h_slope_y[:, 0] = (h[:, 1] - h[:, 0]) / dy
            h_slope_y[:, -1] = (h[:, -1] - h[:, -2]) / dy
            
            # Slope magnitude
            slope_mag = np.sqrt(h_slope_x**2 + h_slope_y**2)
            
            # Slope factor: reduces transport on steep slopes (prevents instability)
            # Maximum slope factor (steep slopes transport less)
            max_slope = 0.5  # ~30 degrees
            slope_factor = 1.0 / (1.0 + slope_mag / max_slope)
            
            # Apply slope factor to reduce transport on steep slopes
            qs_magnitude = qs_magnitude * slope_factor
        
        return qs_magnitude
    
    def compute_sediment_flux_vector(self, tau, u, v, h=None):
        """
        Compute sediment flux vector (direction based on velocity and bed slope).
        
        Args:
            tau: Bed shear stress magnitude
            u: x-velocity
            v: y-velocity
            h: Bed elevation (for slope effects)
            
        Returns:
            qs_x, qs_y: Sediment flux components
        """
        qs_magnitude = self.sediment_flux(tau, h)
        
        # Velocity magnitude
        u_mag = np.sqrt(u**2 + v**2)
        
        # Normalize velocity to get direction
        u_normalized = np.where(u_mag > 1e-10, u / u_mag, 0.0)
        v_normalized = np.where(u_mag > 1e-10, v / u_mag, 0.0)
        
        # Add bed slope effect to flux direction (sediment moves downhill)
        if h is not None:
            nx, ny = h.shape
            dx = self.cell_size
            dy = self.cell_size
            
            # Compute bed slope
            h_slope_x = np.zeros_like(h)
            h_slope_y = np.zeros_like(h)
            
            # X-direction slope (negative gradient = downhill direction)
            h_slope_x[1:-1, :] = -(h[2:, :] - h[:-2, :]) / (2 * dx)
            h_slope_x[0, :] = -(h[1, :] - h[0, :]) / dx
            h_slope_x[-1, :] = -(h[-1, :] - h[-2, :]) / dx
            
            # Y-direction slope
            h_slope_y[:, 1:-1] = -(h[:, 2:] - h[:, :-2]) / (2 * dy)
            h_slope_y[:, 0] = -(h[:, 1] - h[:, 0]) / dy
            h_slope_y[:, -1] = -(h[:, -1] - h[:, -2]) / dy
            
            # Combine flow direction with slope direction
            # Slope contribution factor (how much slope affects transport)
            slope_factor = 0.3  # 30% contribution from slope
            flow_factor = 1.0 - slope_factor
            
            # Normalize slope vector
            slope_mag = np.sqrt(h_slope_x**2 + h_slope_y**2)
            slope_x_norm = np.where(slope_mag > 1e-10, h_slope_x / slope_mag, 0.0)
            slope_y_norm = np.where(slope_mag > 1e-10, h_slope_y / slope_mag, 0.0)
            
            # Combined direction (weighted average of flow and slope)
            u_normalized = flow_factor * u_normalized + slope_factor * slope_x_norm
            v_normalized = flow_factor * v_normalized + slope_factor * slope_y_norm
            
            # Renormalize
            combined_mag = np.sqrt(u_normalized**2 + v_normalized**2)
            u_normalized = np.where(combined_mag > 1e-10, u_normalized / combined_mag, 0.0)
            v_normalized = np.where(combined_mag > 1e-10, v_normalized / combined_mag, 0.0)
        
        # Sediment flux in combined direction
        qs_x = qs_magnitude * u_normalized
        qs_y = qs_magnitude * v_normalized
        
        return qs_x, qs_y
    
    def compute_bank_erosion(self, tau, u, v, h):
        """
        Compute bank erosion rate based on shear stress and water-wall collisions.
        Banks erode differently than bed - typically slower and require higher stress.
        
        Args:
            tau: Shear stress field
            u: x-velocity field
            v: y-velocity field
            h: Current elevation (bed + banks)
            
        Returns:
            bank_erosion_rate: Rate of bank erosion (nx, ny)
        """
        # Bank erosion uses higher critical shear stress
        tau_excess = np.maximum(tau - self.bank_critical_shear, 0.0)
        
        # Bank erosion rate (slower than bed erosion)
        # Erosion proportional to excess stress, but reduced by bank_erosion_rate factor
        bank_erosion = self.bank_erosion_rate * self.transport_coefficient * (tau_excess ** 1.5)
        
        # Only apply to bank cells
        bank_erosion = np.where(self.cell_type != FLUID, bank_erosion, 0.0)
        
        return bank_erosion
    
    def exner_equation(self, qs_x, qs_y, h, dt, tau=None, u=None, v=None):
        """
        Solve Exner equation for bed and bank evolution with zero-flux boundary conditions.
        dh/dt = -1/(1-porosity) * div(qs) + bank_erosion
        
        Uses finite differences with zero-flux boundary conditions to conserve sediment.
        Sediment flux is set to zero at boundaries to prevent creation/destruction.
        Includes separate bank erosion term.
        
        Args:
            qs_x: Sediment flux in x direction
            qs_y: Sediment flux in y direction
            h: Current bed elevation
            dt: Time step
            tau: Shear stress (optional, for bank erosion)
            u: x-velocity (optional, for bank erosion)
            v: y-velocity (optional, for bank erosion)
            
        Returns:
            dh_dt: Rate of change of bed elevation
            h_new: Updated bed elevation
        """
        nx, ny = h.shape
        dx = self.cell_size
        dy = self.cell_size
        
        # Apply zero-flux boundary conditions: set flux to zero at boundaries
        # This ensures no sediment crosses boundaries (closed system)
        qs_x_bc = qs_x.copy()
        qs_y_bc = qs_y.copy()
        
        # Set flux to zero at boundaries
        qs_x_bc[0, :] = 0.0   # Left boundary
        qs_x_bc[-1, :] = 0.0  # Right boundary
        qs_y_bc[:, 0] = 0.0   # Bottom boundary
        qs_y_bc[:, -1] = 0.0  # Top boundary
        
        # Compute divergence using finite differences
        # Use central differences for interior, one-sided at boundaries
        div_qs = np.zeros_like(h)
        
        # Interior points: central differences
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                dqsx_dx = (qs_x_bc[i+1, j] - qs_x_bc[i-1, j]) / (2 * dx)
                dqsy_dy = (qs_y_bc[i, j+1] - qs_y_bc[i, j-1]) / (2 * dy)
                div_qs[i, j] = dqsx_dx + dqsy_dy
        
        # Left boundary (i=0): forward difference in x
        for j in range(1, ny-1):
            dqsx_dx = (qs_x_bc[1, j] - qs_x_bc[0, j]) / dx  # qs_x_bc[0, j] = 0
            dqsy_dy = (qs_y_bc[0, j+1] - qs_y_bc[0, j-1]) / (2 * dy)
            div_qs[0, j] = dqsx_dx + dqsy_dy
        
        # Right boundary (i=nx-1): backward difference in x
        for j in range(1, ny-1):
            dqsx_dx = (qs_x_bc[-1, j] - qs_x_bc[-2, j]) / dx  # qs_x_bc[-1, j] = 0
            dqsy_dy = (qs_y_bc[-1, j+1] - qs_y_bc[-1, j-1]) / (2 * dy)
            div_qs[-1, j] = dqsx_dx + dqsy_dy
        
        # Bottom boundary (j=0): forward difference in y
        for i in range(1, nx-1):
            dqsx_dx = (qs_x_bc[i+1, 0] - qs_x_bc[i-1, 0]) / (2 * dx)
            dqsy_dy = (qs_y_bc[i, 1] - qs_y_bc[i, 0]) / dy  # qs_y_bc[i, 0] = 0
            div_qs[i, 0] = dqsx_dx + dqsy_dy
        
        # Top boundary (j=ny-1): backward difference in y
        for i in range(1, nx-1):
            dqsx_dx = (qs_x_bc[i+1, -1] - qs_x_bc[i-1, -1]) / (2 * dx)
            dqsy_dy = (qs_y_bc[i, -1] - qs_y_bc[i, -2]) / dy  # qs_y_bc[i, -1] = 0
            div_qs[i, -1] = dqsx_dx + dqsy_dy
        
        # Corners: zero flux (set to zero)
        div_qs[0, 0] = 0.0
        div_qs[0, -1] = 0.0
        div_qs[-1, 0] = 0.0
        div_qs[-1, -1] = 0.0
        
        # Exner equation: dh/dt = -1/(1-porosity) * div(qs)
        dh_dt = -1.0 / (1.0 - self.porosity) * div_qs
        
        # Add bank erosion term if provided
        if tau is not None and u is not None and v is not None:
            bank_erosion = self.compute_bank_erosion(tau, u, v, h)
            # Bank erosion reduces elevation (erosion = negative change)
            # Convert erosion rate to elevation change rate
            dh_dt_bank = -bank_erosion / (1.0 - self.porosity)
            # Only apply to bank cells
            dh_dt = np.where(self.cell_type != FLUID, 
                           dh_dt + dh_dt_bank, 
                           dh_dt)
        
        # Validate and clean NaN/Inf values
        dh_dt = np.nan_to_num(dh_dt, nan=0.0, posinf=0.05, neginf=-0.05)
        
        # Update bed elevation
        h_new = h + dh_dt * dt
        
        # Apply stability constraint (limit change rate to prevent numerical instability)
        max_change_rate = 0.02  # Reduced further to prevent instability
        max_change = max_change_rate * dt
        h_change = h_new - h
        h_change = np.clip(h_change, -max_change, max_change)
        h_new = h + h_change
        
        # Validate and clean bed elevation
        h_new = np.nan_to_num(h_new, nan=h, posinf=np.max(h) + 0.05, neginf=np.min(h) - 0.05)
        
        # Soft bounds check - warn if approaching limits but don't clip aggressively
        if np.any(h_new > 5.0) or np.any(h_new < -5.0):
            # Clip only extreme values to prevent numerical issues
            h_new = np.clip(h_new, -10.0, 10.0)
        
        return dh_dt, h_new
    
    def apply_boundary_conditions(self, u, v, h):
        """
        Apply boundary conditions to velocity fields based on bed profile.
        Uses static physical grid but sets velocity boundary conditions according to bed profile h.
        
        Args:
            u: x-velocity field
            v: y-velocity field
            h: bed elevation
            
        Returns:
            u, v: Updated velocity fields with boundary conditions
        """
        nx, ny = u.shape
        
        # Inflow boundary (left side, x=0) - prescribed inflow
        inflow_velocity = 0.5  # m/s
        u[0, 1:-1] = inflow_velocity  # Interior points on left boundary
        v[0, :] = 0.0  # No flow across left boundary
        
        # Outflow boundary (right side, x=nx-1) - zero gradient (Neumann condition)
        u[-1, 1:-1] = u[-2, 1:-1]  # Extrapolate from interior
        v[-1, :] = v[-2, :]
        
        # Top and bottom boundaries (no-slip walls for side boundaries)
        u[:, 0] = 0.0  # Bottom wall
        v[:, 0] = 0.0
        u[:, -1] = 0.0  # Top wall
        v[:, -1] = 0.0
        
        # Bank boundaries: no-slip walls with collision effects
        # Left bank: no flow into bank, reflect velocity
        for i in range(nx):
            for j in range(self.bank_width):
                if self.cell_type[i, j] == BANK_LEFT:
                    u[i, j] = 0.0
                    v[i, j] = 0.0
                    # If adjacent fluid cell flows toward bank, create pressure
                    if j < ny - 1 and self.cell_type[i, j+1] == FLUID:
                        # Reflect velocity component (water bounces off bank)
                        if v[i, j+1] < 0:  # Flowing toward bank
                            v[i, j+1] *= -0.3  # Partial reflection (energy loss)
        
        # Right bank: no flow into bank, reflect velocity
        for i in range(nx):
            for j in range(ny - self.bank_width, ny):
                if self.cell_type[i, j] == BANK_RIGHT:
                    u[i, j] = 0.0
                    v[i, j] = 0.0
                    # If adjacent fluid cell flows toward bank, create pressure
                    if j > 0 and self.cell_type[i, j-1] == FLUID:
                        # Reflect velocity component (water bounces off bank)
                        if v[i, j-1] > 0:  # Flowing toward bank
                            v[i, j-1] *= -0.3  # Partial reflection (energy loss)
        
        # Corner points (average of adjacent boundaries)
        u[0, 0] = 0.0
        u[0, -1] = 0.0
        u[-1, 0] = 0.0
        u[-1, -1] = 0.0
        
        # Update water depth based on bed elevation (assuming constant water surface)
        water_surface_elevation = np.max(h) + 0.5  # Reference water surface
        water_depth_actual = np.maximum(water_surface_elevation - h, 0.01)  # Minimum depth
        
        # Update stored water depth and water surface elevation
        self.water_depth = water_depth_actual
        self.water_surface_elevation = water_surface_elevation
        
        return u, v
    
    def solve_coupled_step(self, h, dt):
        """
        Solve one coupled step of Navier-Stokes and Exner equations.
        
        Args:
            h: Current bed elevation
            dt: Time step
            
        Returns:
            u, v, p: Updated velocity and pressure fields
            h_new: Updated bed elevation
            tau: Bed shear stress
            qs: Sediment flux magnitude
        """
        # Update bed elevation
        self.update_bed_elevation(h)
        
        # Navier-Stokes step
        u, v, p = self.navier_stokes_step(self.u, self.v, self.p, self.h, dt)
        
        # Compute bed shear stress
        tau = self.compute_shear_stress(u, v, self.h)
        
        # Sediment transport
        qs_x, qs_y = self.compute_sediment_flux_vector(tau, u, v)
        qs_magnitude = self.sediment_flux(tau)
        
        # Exner equation (with bank erosion)
        dh_dt, h_new = self.exner_equation(qs_x, qs_y, self.h, dt, tau=tau, u=u, v=v)
        
        # Update boundary conditions based on new bed profile
        u, v = self.apply_boundary_conditions(u, v, h_new)
        
        # Update stored fields
        self.u = u
        self.v = v
        self.h = h_new
        
        return u, v, p, h_new, tau, qs_magnitude


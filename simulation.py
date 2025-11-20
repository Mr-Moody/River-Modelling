import numpy as np
import pickle
from datetime import datetime

from Objects.MeshGrid import MeshGrid
from physics import PhysicsSolver

class Simulation():
    def __init__(self, width:int=20, height:int=20, cell_size:float=0.5, dt:float=0.01,
                 nu=1e-6, rho=1000.0, g=9.81, 
                 sediment_density=2650.0, porosity=0.4,
                 critical_shear=0.05, transport_coefficient=0.1,
                 bank_width=2, bank_erosion_rate=0.3, bank_critical_shear=0.15,
                 bank_height=1.5):
        self.dt = dt  # Delta Time
        self.time = 0.0 # Length of simulation in seconds
        self.frames = []
        self.bank_width = bank_width
        self.bank_height = bank_height  # Height of banks above bed
        
        # Create unified mesh for entire river geometry (left bank + bed + right bank)
        # This is the brown mesh representing the river channel carved into the ground
        self.grid = MeshGrid(width=width, height=height, cell_size=cell_size)
        self.grid.generateMesh()
        self._initialize_unified_mesh()
        
        # Create fluid surface mesh (blue) - represents water surface
        # This spans only the fluid region (between banks)
        fluid_height = height - 2 * bank_width if bank_width > 0 else height
        self.fluid_surface = MeshGrid(width=width, height=fluid_height, cell_size=cell_size)
        self.fluid_surface.generateMesh()
        self._initialize_fluid_surface()
        
        # Initialize physics solver
        self.physics = PhysicsSolver(
            grid_width=width,
            grid_height=height,
            cell_size=cell_size,
            nu=nu,
            rho=rho,
            g=g,
            sediment_density=sediment_density,
            porosity=porosity,
            critical_shear=critical_shear,
            transport_coefficient=transport_coefficient,
            bank_width=bank_width,
            bank_erosion_rate=bank_erosion_rate,
            bank_critical_shear=bank_critical_shear
        )
        
        # Initialize bed elevation from mesh points
        self._update_physics_from_mesh()
    
    def _initialize_unified_mesh(self):
        """Initialize unified mesh with banks elevated above bed region."""
        nx = self.grid.width + 1
        ny = self.grid.height + 1
        
        for i, point in enumerate(self.grid.points):
            # Get grid indices
            grid_i = i // ny
            grid_j = i % ny
            
            # Get current position
            x = point.position[0]
            y = point.position[1]
            z_base = point.position[2]  # Base elevation from mesh generation
            
            # Determine if this point is in bank region or bed region
            if grid_j < self.bank_width:
                # Left bank region: elevate above bed
                # Get bed elevation at interface (column bank_width) for smooth connection
                interface_idx = grid_i * ny + self.bank_width
                if interface_idx < len(self.grid.points):
                    bed_elev_at_interface = self.grid.points[interface_idx].position[2]
                else:
                    bed_elev_at_interface = z_base
                # Bank elevation: smoothly transition from bed + bank_height at interface
                # to higher elevation at outer edge
                distance_from_interface = self.bank_width - grid_j
                z = bed_elev_at_interface + self.bank_height * (1.0 - distance_from_interface / self.bank_width)
            elif grid_j >= (ny - 1 - self.bank_width):
                # Right bank region: elevate above bed
                # Get bed elevation at interface (column height - bank_width) for smooth connection
                interface_col = ny - 1 - self.bank_width
                interface_idx = grid_i * ny + interface_col
                if interface_idx < len(self.grid.points):
                    bed_elev_at_interface = self.grid.points[interface_idx].position[2]
                else:
                    bed_elev_at_interface = z_base
                # Bank elevation: smoothly transition from bed + bank_height at interface
                distance_from_interface = grid_j - interface_col
                z = bed_elev_at_interface + self.bank_height * (1.0 - distance_from_interface / self.bank_width)
            else:
                # Bed (fluid) region: keep base elevation
                z = z_base
            
            point.position = np.array([x, y, z])
    
    def _initialize_fluid_surface(self):
        """Initialize fluid surface mesh at water level."""
        nx = self.fluid_surface.width + 1
        ny = self.fluid_surface.height + 1
        
        for i, point in enumerate(self.fluid_surface.points):
            # Get grid indices
            grid_i = i // ny
            grid_j = i % ny
            
            # Set position
            x = grid_i * self.fluid_surface.cell_size
            # Y position: shift to align with fluid region (starts at bank_width)
            y = (grid_j + self.bank_width) * self.fluid_surface.cell_size
            
            # Z elevation: will be set from water depth in physics
            # Initial: use average bed elevation + initial water depth
            z = 0.5  # Initial water surface elevation (will be updated from physics)
            
            point.position = np.array([x, y, z])
            point.velocity = np.array([0.0, 0.0, 0.0])
    
    def _update_physics_from_mesh(self):
        """Update physics solver with current elevations from unified mesh."""
        # Extract all elevations (z-coordinates) from unified mesh
        # This includes banks and bed as one continuous surface
        all_elevations = np.array([point.position[2] for point in self.grid.points])
        
        # Reshape to match physics grid dimensions
        nx = self.physics.nx
        ny = self.physics.ny
        h_full = all_elevations.reshape(nx, ny)
        
        self.physics.update_bed_elevation(h_full)
    
    def _update_mesh_from_physics(self, h, u, v):
        """Update unified mesh with new elevations and velocity fields."""
        from physics import FLUID, BANK_LEFT, BANK_RIGHT
        
        nx = self.physics.nx
        ny = self.physics.ny
        
        # Validate inputs
        h = np.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0)
        u = np.nan_to_num(u, nan=0.0, posinf=1.0, neginf=-1.0)
        v = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Update all mesh points from physics grid
        h_flat = h.flatten()
        
        for i, point in enumerate(self.grid.points):
            if i < len(h_flat):
                elev = float(h_flat[i])
                if not np.isfinite(elev):
                    elev = point.position[2]
                point.position[2] = elev
                
                # Get grid indices
                grid_i = i // ny
                grid_j = i % ny
                grid_i = min(max(grid_i, 0), nx - 1)
                grid_j = min(max(grid_j, 0), ny - 1)
                
                # Update velocity based on cell type
                cell_type = self.physics.cell_type[grid_i, grid_j]
                if cell_type == FLUID:
                    # Fluid region: use computed velocity
                    vel_u = float(u[grid_i, grid_j])
                    vel_v = float(v[grid_i, grid_j])
                    if not np.isfinite(vel_u):
                        vel_u = 0.0
                    if not np.isfinite(vel_v):
                        vel_v = 0.0
                    point.velocity[0] = vel_u
                    point.velocity[1] = vel_v
                    point.velocity[2] = 0.0
                else:
                    # Bank region: no velocity (static walls)
                    point.velocity = np.array([0.0, 0.0, 0.0])
        
        # Update fluid surface mesh
        self._update_fluid_surface(h, u, v)
    
    def _update_fluid_surface(self, h, u, v):
        """Update fluid surface mesh with water surface elevation."""
        from physics import FLUID
        
        nx = self.physics.nx
        ny = self.physics.ny
        
        # Get water depth from physics
        water_depth = self.physics.water_depth
        
        # Extract fluid region (between banks)
        fluid_h = h[:, self.bank_width:-self.bank_width] if self.bank_width > 0 else h
        fluid_depth = water_depth[:, self.bank_width:-self.bank_width] if self.bank_width > 0 else water_depth
        
        # Water surface elevation = bed elevation + water depth
        water_surface = fluid_h + fluid_depth
        water_surface_flat = water_surface.flatten()
        
        # Update fluid surface mesh points
        for i, point in enumerate(self.fluid_surface.points):
            if i < len(water_surface_flat):
                surface_elev = float(water_surface_flat[i])
                if not np.isfinite(surface_elev):
                    surface_elev = point.position[2]
                point.position[2] = surface_elev
                
                # Map velocity from physics grid (fluid region only)
                grid_i = i // (ny - 2 * self.bank_width)
                grid_j = i % (ny - 2 * self.bank_width) + self.bank_width
                grid_i = min(max(grid_i, 0), nx - 1)
                grid_j = min(max(grid_j, 0), ny - 1)
                
                vel_u = float(u[grid_i, grid_j])
                vel_v = float(v[grid_i, grid_j])
                if not np.isfinite(vel_u):
                    vel_u = 0.0
                if not np.isfinite(vel_v):
                    vel_v = 0.0
                point.velocity[0] = vel_u
                point.velocity[1] = vel_v
                point.velocity[2] = 0.0

    def saveFrame(self):
        """
        Save current state as a frame with timestamp.
        Includes:
        - Bed/bank mesh (brown): full river geometry
        - Fluid surface mesh (blue): water surface
        """
        self.grid.updateVerticesFromPoints()
        self.fluid_surface.updateVerticesFromPoints()
        
        # Validate and collect vertices from bed/bank mesh (brown)
        bed_vertices = np.nan_to_num(self.grid.vertices.copy(), nan=0.0, posinf=1e3, neginf=-1e3)
        bed_velocities = np.array([point.velocity for point in self.grid.points])
        bed_velocities = np.nan_to_num(bed_velocities, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # Validate and collect vertices from fluid surface mesh (blue)
        fluid_vertices = np.nan_to_num(self.fluid_surface.vertices.copy(), nan=0.0, posinf=1e3, neginf=-1e3)
        fluid_velocities = np.array([point.velocity for point in self.fluid_surface.points])
        fluid_velocities = np.nan_to_num(fluid_velocities, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # Save frame with both meshes
        frame = {
            "timestamp": self.time,
            # Bed/bank mesh (brown) - full river geometry
            "bed_vertices": bed_vertices,
            "bed_velocities": bed_velocities,
            # Fluid surface mesh (blue) - water surface
            "fluid_vertices": fluid_vertices,
            "fluid_velocities": fluid_velocities,
            # Legacy format for backward compatibility
            "vertices": bed_vertices,
            "velocities": bed_velocities
        }

        self.frames.append(frame)
    

    def step(self):
        """
        Perform one simulation step using coupled Navier-Stokes and Exner equations.
        """
        # Get current bed elevation from mesh
        self._update_physics_from_mesh()
        h = self.physics.h.copy()
        
        # Navier-Stokes step
        u, v, p = self.physics.navier_stokes_step(
            self.physics.u, 
            self.physics.v, 
            self.physics.p, 
            h, 
            self.dt
        )
        
        # Compute bed shear stress from velocity
        tau = self.physics.compute_shear_stress(u, v, h)
        
        # Sediment transport and Exner equation
        # Pass bed elevation to include slope effects
        qs_x, qs_y = self.physics.compute_sediment_flux_vector(tau, u, v, h)
        qs = self.physics.sediment_flux(tau, h)
        
        # Solve Exner equation for bed and bank evolution (with bank erosion)
        dh_dt, h_new = self.physics.exner_equation(qs_x, qs_y, h, self.dt, tau=tau, u=u, v=v)
        
        # Update boundary conditions based on new bed profile
        u, v = self.physics.apply_boundary_conditions(u, v, h_new)
        
        # Update mesh points with new bed elevation and velocities
        self._update_mesh_from_physics(h_new, u, v)
        
        # Update stored physics fields
        self.physics.u = u
        self.physics.v = v
        self.physics.p = p
        self.physics.h = h_new
        
        self.time += self.dt
        
        # Update vertices for next step
        self.grid.updateVerticesFromPoints()
        self.fluid_surface.updateVerticesFromPoints()
    

    def run(self, duration:float=10.0, save_interval:float=0.1, output_file:str="simulation_frames.pkl"):
        """
        Run simulation and save frames.
        
        Args:
            duration: Total simulation time in seconds.
            save_interval: Time between saved frames in seconds.
            output_file: Path to output file.
        """
        num_steps = int(duration / self.dt)
        save_every = int(save_interval / self.dt)
        
        print(f"Running simulation for {duration} seconds...")
        print(f"Saving every {save_interval} seconds ({save_every} steps)")
        
        # Save initial frame
        self.saveFrame()
        
        for step in range(num_steps):
            self.step()
            
            # Save frame at intervals
            if step % save_every == 0:
                self.saveFrame()
                if step % (save_every * 10) == 0:
                    print(f"Time: {self.time:.2f}s, Frame: {len(self.frames)}")
        
        # Save final frame
        if len(self.frames) == 0 or self.frames[-1]['timestamp'] != self.time:
            self.saveFrame()
        
        # Save to file
        self.saveToFile(output_file)
        print(f"\nSimulation complete. Saved {len(self.frames)} frames to {output_file}")
    

    def saveToFile(self, filename:str="simulation_frames.pkl"):
        """
        Save all frames to a pickle file.
        """

        with open(filename, "wb") as f:
            pickle.dump({
                "frames": self.frames,
                "metadata": {
                    "width": self.grid.width,
                    "height": self.grid.height,
                    "cell_size": self.grid.cell_size,
                    "dt": self.dt,
                    "num_frames": len(self.frames),
                    "duration": self.time,
                    "bank_width": self.bank_width,
                    "bank_height": self.bank_height,
                    "created": datetime.now().isoformat()
                }
            }, f)
    

    @staticmethod
    def loadFromFile(filename:str="simulation_frames.pkl"):
        """
        Load simulation frames from file.
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)

        return data


if __name__ == "__main__":
    sim = Simulation(width=20, height=20, cell_size=1, dt=0.01, nu=1e-6, rho=1000.0, g=9.81, sediment_density=2650.0, porosity=0.4, critical_shear=0.05, transport_coefficient=0.1)
    sim.run(duration=20.0, save_interval=0.1, output_file="simulation_frames.pkl")


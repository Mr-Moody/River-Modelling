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
                 bank_height=1, terrain_width=3):

        self.dt = dt  # Delta Time
        self.time = 0.0 # Length of simulation in seconds
        self.frames = []
        self.bank_width = bank_width
        self.bank_height = bank_height  # Height of banks above bed
        self.terrain_width = terrain_width
        
        # Create mesh for entire geometry left bank, bed, right bank
        self.grid = MeshGrid(width=width, height=height, cell_size=cell_size)
        self.grid.generateMesh()
        self._initialise_unified_mesh()
        
        # Create fluid surface mesh between banks
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
            bank_critical_shear=bank_critical_shear,
            terrain_width=terrain_width
        )
        
        # Initialise bed elevation from mesh points
        self._update_physics_from_mesh()
    
    def _initialise_unified_mesh(self):
        """
        Initialise unified mesh with terrain, banks, and bed regions.
        Layout: [TERRAIN | BANK | FLUID | BANK | TERRAIN]
        Terrain is highest, then banks, then bed (lowest).
        """
        from physics import TERRAIN, BANK, FLUID
        
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
            
            # Determine cell type from physics solver
            if hasattr(self, 'physics'):
                cell_type = self.physics.cell_type[grid_i, grid_j]
            else:
                # Fallback: determine from position
                if grid_j < self.terrain_width:
                    cell_type = TERRAIN
                elif grid_j < self.terrain_width + self.bank_width:
                    cell_type = BANK
                elif grid_j >= ny - (self.terrain_width + self.bank_width):
                    cell_type = TERRAIN
                elif grid_j >= ny - self.terrain_width:
                    cell_type = BANK
                else:
                    cell_type = FLUID
            
            # Set elevation based on cell type
            if cell_type == TERRAIN:
                # Terrain is highest - above banks
                terrain_height = self.bank_height + 0.5  # Terrain 0.5m above banks
                z = z_base + terrain_height
            elif cell_type == BANK:
                # Bank: transition from bed to bank height
                # Determine which side (left or right) based on grid index
                if grid_j < ny // 2:
                    # Left bank
                    interface_col = self.terrain_width + self.bank_width
                    if grid_j < interface_col:
                        distance_from_interface = interface_col - grid_j
                        z = z_base + self.bank_height * (distance_from_interface / self.bank_width)
                    else:
                        z = z_base
                else:
                    # Right bank
                    interface_col = ny - 1 - (self.terrain_width + self.bank_width)
                    if grid_j > interface_col:
                        distance_from_interface = grid_j - interface_col
                        z = z_base + self.bank_height * (distance_from_interface / self.bank_width)
                    else:
                        z = z_base
            else:  # FLUID
                # Keep base elevation at bed level
                z = z_base
            
            point.position = np.array([x, y, z])
    
    def _initialize_fluid_surface(self):
        """
        Initialise fluid surface mesh at water level.
        """
        nx = self.fluid_surface.width + 1
        ny = self.fluid_surface.height + 1
        
        for i, point in enumerate(self.fluid_surface.points):
            grid_i = i // ny
            grid_j = i % ny
            
            # Set position
            x = grid_i * self.fluid_surface.cell_size
            y = (grid_j + self.bank_width) * self.fluid_surface.cell_size
            z = 0.5  # Initial water surface elevation (will be updated from physics)
            
            point.position = np.array([x, y, z])
            point.velocity = np.array([0.0, 0.0, 0.0])
    
    def _update_physics_from_mesh(self):
        """
        Update physics solver with current elevations from unified mesh.
        """
        # Extract all elevations from unified mesh
        all_elevations = np.array([point.position[2] for point in self.grid.points])
        
        # Reshape to match physics grid dimensions
        nx = self.physics.nx
        ny = self.physics.ny
        h_full = all_elevations.reshape(nx, ny)
        
        self.physics.update_bed_elevation(h_full)
    
    def _update_mesh_from_physics(self, h, u, v):
        """
        Update unified mesh with new elevations and velocity fields.
        """
        from physics import FLUID
        
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
                    # Bank regions have no velocity as static
                    point.velocity = np.array([0.0, 0.0, 0.0])
        
        # Update fluid surface mesh
        self._update_fluid_surface(h, u, v)
    
    def _update_fluid_surface(self, h, u, v):
        """
        Update fluid surface mesh with water surface elevation.
        Snaps water vertices to terrain surface when water would cut into terrain.
        """
        from physics import FLUID, TERRAIN
        
        nx = self.physics.nx
        ny = self.physics.ny
        
        water_depth = self.physics.water_depth
        
        # Extract fluid region between banks
        fluid_h = h[:, self.bank_width:-self.bank_width] if self.bank_width > 0 else h
        fluid_depth = water_depth[:, self.bank_width:-self.bank_width] if self.bank_width > 0 else water_depth
        
        # Water surface elevation = bed elevation + water depth
        water_surface = fluid_h + fluid_depth
        water_surface_flat = water_surface.flatten()
        
        # Build elevation map from unified mesh (all cells, including terrain)
        # This allows us to check terrain elevation at any (x, y) position
        unified_elevations = np.zeros((nx, ny))
        for idx, grid_point in enumerate(self.grid.points):
            grid_i = idx // ny
            grid_j = idx % ny
            if 0 <= grid_i < nx and 0 <= grid_j < ny:
                unified_elevations[grid_i, grid_j] = grid_point.position[2]
        
        # Update fluid surface mesh points
        for i, point in enumerate(self.fluid_surface.points):
            if i < len(water_surface_flat):
                surface_elev = float(water_surface_flat[i])

                if not np.isfinite(surface_elev):
                    surface_elev = point.position[2]
                
                # Get the (x, y) position of this fluid surface point
                x = point.position[0]
                y = point.position[1]
                
                # Find corresponding grid indices in unified mesh
                grid_i_unified = int(round(x / self.grid.cell_size))
                grid_j_unified = int(round(y / self.grid.cell_size))
                
                # Clamp to valid range
                grid_i_unified = max(0, min(grid_i_unified, nx - 1))
                grid_j_unified = max(0, min(grid_j_unified, ny - 1))
                
                # Get elevation from unified mesh at this location
                unified_elev = unified_elevations[grid_i_unified, grid_j_unified]
                cell_type_at_location = self.physics.cell_type[grid_i_unified, grid_j_unified]
                
                # Check for terrain at current location and adjacent cells
                # Find maximum terrain elevation in current cell and 8-connected neighbors
                max_terrain_elev = -np.inf
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = grid_i_unified + di, grid_j_unified + dj
                        if 0 <= ni < nx and 0 <= nj < ny:
                            if self.physics.cell_type[ni, nj] == TERRAIN:
                                neighbor_elev = unified_elevations[ni, nj]
                                if neighbor_elev > max_terrain_elev:
                                    max_terrain_elev = neighbor_elev
                
                # If terrain is found and its elevation is higher than water surface,
                # snap water to terrain surface at the same elevation height
                if max_terrain_elev > -np.inf and max_terrain_elev > surface_elev:
                    surface_elev = max_terrain_elev

                point.position[2] = surface_elev
                
                # Map velocity from physics grid (fluid region only)
                fluid_height = ny - 2 * self.bank_width if self.bank_width > 0 else ny
                grid_i = i // fluid_height
                grid_j = i % fluid_height + self.bank_width
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
        - Ground mesh (brown/green): terrain, banks, and bed combined
        - Fluid surface mesh (blue): water surface
        """
        
        self.grid.updateVerticesFromPoints()
        self.fluid_surface.updateVerticesFromPoints()
        
        # Get ground vertices (unified terrain/bank/bed)
        ground_vertices = np.nan_to_num(self.grid.vertices.copy(), nan=0.0, posinf=1e3, neginf=-1e3)
        ground_velocities = np.array([point.velocity for point in self.grid.points])
        ground_velocities = np.nan_to_num(ground_velocities, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # Validate vertices from fluid surface mesh
        fluid_vertices = np.nan_to_num(self.fluid_surface.vertices.copy(), nan=0.0, posinf=1e3, neginf=-1e3)
        fluid_velocities = np.array([point.velocity for point in self.fluid_surface.points])
        fluid_velocities = np.nan_to_num(fluid_velocities, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # Save frame with unified ground mesh and fluid mesh
        frame = {
            "timestamp": self.time,
            "ground_vertices": ground_vertices,
            "ground_velocities": ground_velocities,
            "fluid_vertices": fluid_vertices,
            "fluid_velocities": fluid_velocities,
            # Keep legacy fields empty or mapped for backward compatibility if needed
            "vertices": ground_vertices,
            "velocities": ground_velocities
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
        
        # Sediment transport and slope effects
        qs_x, qs_y = self.physics.compute_sediment_flux_vector(tau, u, v, h)
        qs = self.physics.sediment_flux(tau, h)
        
        # Solve Exner equation for bed and bank evolution with bank erosion
        dh_dt, h_new = self.physics.exner_equation(qs_x, qs_y, h, self.dt, tau=tau, u=u, v=v)
        
        # Update boundary conditions based on new bed profile
        u, v = self.physics.apply_boundary_conditions(u, v, h_new)
        
        # Update mesh points with new bed elevation and velocities
        self._update_mesh_from_physics(h_new, u, v)
        
        # Update physics variables
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
                    "terrain_width": self.terrain_width,
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
    sim = Simulation(width=15, height=15, cell_size=1, dt=0.01, nu=1e-6, rho=1000.0, g=9.81, sediment_density=2650.0, porosity=0.4, critical_shear=0.05, transport_coefficient=0.1, bank_width=1, bank_erosion_rate=0.3, bank_critical_shear=0.15, bank_height=1, terrain_width=2)
    sim.run(duration=20.0, save_interval=0.1, output_file="simulation_frames.pkl")


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional

from simulation import Simulation

# Try to import PyVista for GPU-accelerated rendering
try:
    import pyvista as pv  # type: ignore
    PYVISTA_AVAILABLE = True
    # PyVista uses VTK which automatically leverages GPU acceleration through OpenGL
    # Set preferred backend for better GPU support
    pv.set_plot_theme("document")  # Use document theme for better performance
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None  # type: ignore

def generate_triangles_indices(width, height):
    """Generate triangle indices for a mesh grid."""
    num_vertices_x = width + 1
    num_vertices_z = height + 1
    triangles_indices = []
    
    for i in range(width):
        for j in range(height):
            bottom_left = i * num_vertices_z + j
            bottom_right = i * num_vertices_z + (j + 1)
            top_left = (i + 1) * num_vertices_z + j
            top_right = (i + 1) * num_vertices_z + (j + 1)
            
            triangles_indices.append([bottom_left, top_left, bottom_right])
            triangles_indices.append([bottom_right, top_left, top_right])
    
    return triangles_indices

def visualiseSimulation(filename:str="simulation_frames.pkl", frame_index:int=0, animate:bool=False, use_gpu:bool=True):
    """
    Visualise simulation frame. If animate is True, animate through all frames otherwise display single frame.
    
    Args:
        filename: Path to simulation frames file
        frame_index: Index of frame to display (if not animating)
        animate: Whether to animate through all frames
        use_gpu: Whether to use GPU-accelerated rendering (PyVista). Falls back to matplotlib if PyVista unavailable.
    """

    data = Simulation.loadFromFile(filename)
    frames = data["frames"]
    metadata = data["metadata"]
    
    print(f"Loaded {len(frames)} frames")
    print(f"Duration: {metadata['duration']:.2f}s")
    print(f"Grid: {metadata['width']}x{metadata['height']}, cell_size={metadata['cell_size']}")
    print(f"Delta Time: {metadata['dt']:.2f}s")
    print(f"Number of frames: {metadata['num_frames']}")
    
    width = metadata["width"]
    height = metadata["height"]
    
    # Generate triangle indices for bed
    bed_triangles = generate_triangles_indices(width, height)
    
    # Generate triangle indices for banks if they exist
    bank_width = metadata.get("bank_width", 0)
    bank_triangles = None
    if bank_width > 0:
        bank_triangles = generate_triangles_indices(width, bank_width)
    
    triangles_data = {
        "bed": bed_triangles,
        "bank": bank_triangles
    }
    
    # Use GPU-accelerated PyVista if available and requested
    if use_gpu and PYVISTA_AVAILABLE:
        if animate:
            animateFramesGPU(frames, triangles_data, metadata)
        else:
            frame = frames[frame_index]
            plotFrameGPU(frame, triangles_data, metadata, frame_index)
    else:
        if not use_gpu:
            print("Using CPU-based matplotlib rendering")
        elif not PYVISTA_AVAILABLE:
            print("PyVista not available, falling back to matplotlib. Install with: pip install pyvista")
        if animate:
            animateFrames(frames, triangles_data, metadata)
        else:
            frame = frames[frame_index]
            plotFrame(frame, triangles_data, metadata, frame_index)


def createMeshFromVertices(vertices:np.ndarray, triangles_indices:list, 
                          facecolor="lightblue", edgecolor="blue", alpha=0.6) -> Poly3DCollection:
    """
    Helper function to create a Poly3DCollection from vertices and triangle indices.
    Validates vertices and handles NaN/Inf values.
    """
    # Clean vertices
    vertices = np.nan_to_num(vertices, nan=0.0, posinf=1e3, neginf=-1e3)
    
    triangles = []

    for idx in triangles_indices:
        # Validate indices
        if any(i >= len(vertices) or i < 0 for i in idx):
            continue
        v0 = vertices[idx[0]]
        v1 = vertices[idx[1]]
        v2 = vertices[idx[2]]
        
        # Validate vertices are finite
        if not all(np.isfinite(v0)) or not all(np.isfinite(v1)) or not all(np.isfinite(v2)):
            continue

        triangles.append([v0, v1, v2])
    
    if len(triangles) == 0:
        # Return empty collection if no valid triangles
        return Poly3DCollection([], alpha=alpha, facecolor=facecolor, edgecolor=edgecolor, linewidths=0.3)
    
    return Poly3DCollection(triangles, alpha=alpha, facecolor=facecolor, edgecolor=edgecolor, linewidths=0.3)


def setupAxes(ax, vertices:np.ndarray):
    """
    Helper function to set up axis limits and labels.
    Handles NaN/Inf values gracefully.
    """
    # Remove NaN/Inf values for range calculation
    valid_vertices = vertices[np.isfinite(vertices).all(axis=1)]
    
    if len(valid_vertices) == 0:
        # If no valid vertices, use default range
        x_range = [0, 1]
        y_range = [0, 1]
        z_range = [0, 1]
    else:
        x_range = [valid_vertices[:, 0].min(), valid_vertices[:, 0].max()]
        y_range = [valid_vertices[:, 1].min(), valid_vertices[:, 1].max()]
        z_range = [valid_vertices[:, 2].min(), valid_vertices[:, 2].max()]
        
        # Make range non zero
        if x_range[1] == x_range[0]:
            x_range[1] += 1.0
        if y_range[1] == y_range[0]:
            y_range[1] += 1.0
        if z_range[1] == z_range[0]:
            z_range[1] += 0.1
    
    # Use default ranges if invalid
    if not all(np.isfinite(x_range + y_range + z_range)):
        x_range = [0, 1]
        y_range = [0, 1]
        z_range = [0, 1]
    
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_zlim(z_range[0], z_range[1])
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def plotFrame(frame:dict, triangles_data:dict, metadata:dict, frame_index:int, ax=None, total_frames:Optional[int]=None):
    """
    Plot a single frame on the figure, including bed and banks.
    
    Args:
        frame: Frame data dictionary
        triangles_data: Dictionary with "bed" and "bank" triangle indices
        metadata: Simulation metadata
        frame_index: Index of the frame
        ax: Optional existing axes to plot on (for animation)
        total_frames: Total number of frames (for animation title)
    
    """
    # Collect all vertices for axis setup
    all_vertices_list = []
    
    # Get unified mesh vertices (includes banks + bed as one continuous surface)
    if "bed_vertices" in frame and frame["bed_vertices"] is not None:
        bed_vertices = frame["bed_vertices"]
    elif "vertices" in frame and frame["vertices"] is not None:
        bed_vertices = frame["vertices"]
    else:
        print(f"Warning: Frame {frame_index} has no valid vertices")
        return
    
    bed_vertices = np.nan_to_num(bed_vertices, nan=0.0, posinf=1e3, neginf=-1e3)
    all_vertices_list.append(bed_vertices)
    
    # Get fluid surface vertices if they exist
    fluid_vertices = None
    if "fluid_vertices" in frame and frame["fluid_vertices"] is not None:
        fluid_vertices = np.nan_to_num(frame["fluid_vertices"], nan=0.0, posinf=1e3, neginf=-1e3)
        all_vertices_list.append(fluid_vertices)
    
    # Combine all vertices for axis limits
    all_vertices = np.concatenate(all_vertices_list, axis=0) if all_vertices_list else bed_vertices
    timestamp = frame["timestamp"]
    
    created_new_figure = False
    
    # Create new figure and axes if not provided
    if ax is None:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        setupAxes(ax, all_vertices)

        ax.set_title(f"River Simulation Frame {frame_index} | Time: {timestamp:.2f}s")

        created_new_figure = True
    else:
        # Clear existing mesh for animation
        while ax.collections:
            ax.collections[0].remove()

        # Update title for animation
        if total_frames is not None:
            ax.set_title(f"River Simulation Frame {frame_index}/{total_frames-1} | Time: {timestamp:.2f}s")
        else:
            ax.set_title(f"River Simulation Frame {frame_index} | Time: {timestamp:.2f}s")
    
    # Get parameters
    bank_width = metadata.get("bank_width", 0)
    terrain_width = metadata.get("terrain_width", 0)
    height = metadata["height"]
    width = metadata["width"]
    
    # Plot terrain mesh (green) - surrounding terrain
    if "terrain_vertices" in frame and frame["terrain_vertices"] is not None and len(frame["terrain_vertices"]) > 0:
        terrain_vertices = np.nan_to_num(frame["terrain_vertices"], nan=0.0, posinf=1e3, neginf=-1e3)
        all_vertices_list.append(terrain_vertices)
        
        # Generate triangles for terrain mesh (full grid, filtered to terrain cells)
        # Use the full bed triangles but filter to only include terrain vertices
        terrain_triangles = []
        if "terrain_indices" in frame:
            terrain_indices_set = set(frame["terrain_indices"])
            bed_triangles = triangles_data.get("bed", [])
            for triangle in bed_triangles:
                # Check if all three vertices of triangle are terrain
                if all(idx in terrain_indices_set for idx in triangle):
                    # Remap indices to terrain vertex array
                    remapped_triangle = [frame["terrain_indices"].index(idx) for idx in triangle]
                    terrain_triangles.append(remapped_triangle)
        
        if len(terrain_triangles) > 0:
            terrain_mesh = createMeshFromVertices(terrain_vertices, terrain_triangles,
                                                 facecolor="green", edgecolor="darkgreen", alpha=0.7)
            ax.add_collection3d(terrain_mesh)
    
    # Plot bed/bank mesh (brown) - full river geometry
    bed_triangles = triangles_data.get("bed", [])
    # Filter out triangles that are entirely terrain
    if "terrain_indices" in frame and "bed_indices" in frame:
        terrain_indices_set = set(frame["terrain_indices"])
        bed_indices_set = set(frame["bed_indices"])
        bed_bank_triangles = []
        for triangle in bed_triangles:
            # Include triangle if at least one vertex is bed/bank (not all terrain)
            if not all(idx in terrain_indices_set for idx in triangle):
                # Remap indices to bed/bank vertex array
                try:
                    remapped_triangle = [frame["bed_indices"].index(idx) for idx in triangle if idx in bed_indices_set]
                    if len(remapped_triangle) == 3:
                        bed_bank_triangles.append(remapped_triangle)
                except ValueError:
                    pass
        bed_triangles = bed_bank_triangles if bed_bank_triangles else bed_triangles
    
    bed_mesh = createMeshFromVertices(bed_vertices, bed_triangles,
                                      facecolor="saddlebrown", edgecolor="brown", alpha=0.8)
    ax.add_collection3d(bed_mesh)
    
    # Plot fluid surface mesh (blue) - water surface connecting between banks
    if fluid_vertices is not None:
        # Generate triangles for fluid surface mesh
        fluid_height = height - 2 * bank_width if bank_width > 0 else height
        num_vertices_y_fluid = fluid_height + 1
        fluid_triangles = []
        
        for i in range(width):
            for j in range(fluid_height):
                bottom_left = i * num_vertices_y_fluid + j
                bottom_right = i * num_vertices_y_fluid + (j + 1)
                top_left = (i + 1) * num_vertices_y_fluid + j
                top_right = (i + 1) * num_vertices_y_fluid + (j + 1)
                
                fluid_triangles.append([bottom_left, top_left, bottom_right])
                fluid_triangles.append([bottom_right, top_left, top_right])
        
        # Render fluid surface (blue)
        fluid_mesh = createMeshFromVertices(fluid_vertices, fluid_triangles,
                                           facecolor="lightblue", edgecolor="blue", alpha=0.6)
        ax.add_collection3d(fluid_mesh)
    
    # Show or update figure
    if created_new_figure:
        plt.show()
    else:
        plt.draw()


def animateFrames(frames:list, triangles_data:dict, metadata:dict):
    """
    Animate through all frames.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    # Collect all vertices, filtering out invalid frames
    valid_frames = [f for f in frames if ("vertices" in f or "bed_vertices" in f) and 
                    (f.get("vertices") is not None or f.get("bed_vertices") is not None)]
    if len(valid_frames) == 0:
        print("No valid frames to animate!")
        return
    
    # Clean vertices before concatenation
    all_vertices_list = []
    for frame in valid_frames:
        # Get bed/bank mesh vertices (brown)
        if "bed_vertices" in frame:
            vertices = frame["bed_vertices"]
        else:
            vertices = frame["vertices"]
        # Clean NaN/Inf values
        vertices_clean = np.nan_to_num(vertices, nan=0.0, posinf=1e3, neginf=-1e3)
        all_vertices_list.append(vertices_clean)
        
        # Add fluid surface vertices (blue) if they exist
        if "fluid_vertices" in frame and frame["fluid_vertices"] is not None:
            fluid_vertices = np.nan_to_num(frame["fluid_vertices"], nan=0.0, posinf=1e3, neginf=-1e3)
            all_vertices_list.append(fluid_vertices)
    
    if len(all_vertices_list) > 0:
        all_vertices = np.concatenate(all_vertices_list, axis=0)
        setupAxes(ax, all_vertices)
    else:
        setupAxes(ax, np.array([[0, 0, 0], [1, 1, 1]]))

    total_frames = len(frames)

    for frame_idx, frame in enumerate(frames):
        if ("vertices" not in frame and "bed_vertices" not in frame) or \
           (frame.get("vertices") is None and frame.get("bed_vertices") is None):
            continue
        plotFrame(frame, triangles_data, metadata, frame_idx, ax=ax, total_frames=total_frames)

        # Pause for Delta Time
        plt.pause(metadata["dt"])
    
    # Keeps showing final frame in figure
    plt.show()


# ============================================================================
# GPU-ACCELERATED RENDERING FUNCTIONS (PyVista)
# ============================================================================

def setCameraAboveRiver(plotter, vertices: np.ndarray):
    """
    Set camera position to look down at the river from above.
    
    Args:
        plotter: PyVista plotter object
        vertices: Array of vertices to calculate bounds from
    """
    if len(vertices) == 0:
        return
    
    # Calculate bounds
    valid_vertices = vertices[np.isfinite(vertices).all(axis=1)]
    if len(valid_vertices) == 0:
        return
    
    # Get center of the river
    center_x = (valid_vertices[:, 0].min() + valid_vertices[:, 0].max()) / 2
    center_y = (valid_vertices[:, 1].min() + valid_vertices[:, 1].max()) / 2
    center_z = (valid_vertices[:, 2].min() + valid_vertices[:, 2].max()) / 2
    
    # Calculate dimensions
    width = valid_vertices[:, 0].max() - valid_vertices[:, 0].min()
    height = valid_vertices[:, 1].max() - valid_vertices[:, 1].min()
    depth = valid_vertices[:, 2].max() - valid_vertices[:, 2].min()
    
    # Set camera position above the river
    # Height should be enough to see the entire river, with some margin
    max_dimension = max(width, height, depth)
    camera_height = max_dimension * 1.5  # 1.5x the largest dimension
    
    # Camera position: above the center, looking down
    camera_position = [center_x, center_y, center_z + camera_height]
    focal_point = [center_x, center_y, center_z]  # Look at the center
    view_up = [0, 1, 0]  # Y-axis is up
    
    plotter.camera_position = [camera_position, focal_point, view_up]  # type: ignore


def createPyVistaMesh(vertices: np.ndarray, triangles_indices: list):
    """
    Create a PyVista PolyData mesh from vertices and triangle indices.
    This uses GPU acceleration through VTK/OpenGL.
    """
    # Clean vertices
    vertices = np.nan_to_num(vertices, nan=0.0, posinf=1e3, neginf=-1e3)
    
    # Filter out invalid triangles
    valid_triangles = []
    for idx in triangles_indices:
        if all(0 <= i < len(vertices) for i in idx):
            v0 = vertices[idx[0]]
            v1 = vertices[idx[1]]
            v2 = vertices[idx[2]]
            if all(np.isfinite(v0)) and all(np.isfinite(v1)) and all(np.isfinite(v2)):
                valid_triangles.append(idx)
    
    if len(valid_triangles) == 0:
        # Return empty mesh
        return pv.PolyData()
    
    # Convert to numpy array format for PyVista
    # PyVista expects faces as [n, i0, i1, i2, ...] where n is number of points
    faces = []
    for triangle in valid_triangles:
        faces.append(3)  # Number of points in this face
        faces.extend(triangle)
    
    faces_array = np.array(faces, dtype=np.int32)
    
    # Create PyVista mesh
    if pv is None:
        raise ImportError("PyVista is not available")
    mesh = pv.PolyData(vertices, faces_array)  # type: ignore
    return mesh


def plotFrameGPU(frame: dict, triangles_data: dict, metadata: dict, frame_index: int, 
                 plotter: Optional[object] = None, total_frames: Optional[int] = None,
                 actors: Optional[dict] = None):
    """
    Plot a single frame using GPU-accelerated PyVista rendering.
    
    Args:
        frame: Frame data dictionary
        triangles_data: Dictionary with "bed" and "bank" triangle indices
        metadata: Simulation metadata
        frame_index: Index of the frame
        plotter: Optional existing PyVista plotter (for animation)
        total_frames: Total number of frames (for animation title)
        actors: Dictionary to store actor references for smooth updates (for animation)
    """
    if not PYVISTA_AVAILABLE or pv is None:
        raise ImportError("PyVista is not available. Install with: pip install pyvista")
    
    # Get bed vertices
    if "bed_vertices" in frame and frame["bed_vertices"] is not None:
        bed_vertices = frame["bed_vertices"]
    elif "vertices" in frame and frame["vertices"] is not None:
        bed_vertices = frame["vertices"]
    else:
        print(f"Warning: Frame {frame_index} has no valid vertices")
        return None
    
    bed_vertices = np.nan_to_num(bed_vertices, nan=0.0, posinf=1e3, neginf=-1e3)
    
    # Get fluid surface vertices if they exist
    fluid_vertices = None
    if "fluid_vertices" in frame and frame["fluid_vertices"] is not None:
        fluid_vertices = np.nan_to_num(frame["fluid_vertices"], nan=0.0, posinf=1e3, neginf=-1e3)
    
    # Get terrain vertices if they exist
    terrain_vertices = None
    if "terrain_vertices" in frame and frame["terrain_vertices"] is not None and len(frame["terrain_vertices"]) > 0:
        terrain_vertices = np.nan_to_num(frame["terrain_vertices"], nan=0.0, posinf=1e3, neginf=-1e3)
    
    timestamp = frame["timestamp"]
    
    # Create or use existing plotter
    created_new_plotter = False
    if plotter is None:
        plotter = pv.Plotter(window_size=[1400, 1000])  # type: ignore
        created_new_plotter = True
        # Set up initial camera and bounds
        all_vertices_list = [bed_vertices]
        if fluid_vertices is not None:
            all_vertices_list.append(fluid_vertices)
        if terrain_vertices is not None:
            all_vertices_list.append(terrain_vertices)
        all_vertices = np.concatenate(all_vertices_list, axis=0)
        valid_vertices = all_vertices[np.isfinite(all_vertices).all(axis=1)]
        if len(valid_vertices) > 0:
            setCameraAboveRiver(plotter, all_vertices)  # Set camera above river
    
    # Update title text - only update if it's the first frame or every 30 frames to reduce flashing
    # Removing/re-adding text causes flashing, so we minimize it
    update_title = (actors is None or "title" not in actors or frame_index % 30 == 0)
    if update_title:
        if actors is not None and "title" in actors:
            plotter.remove_actor(actors["title"])  # type: ignore
        
        # Add title
        if total_frames is not None:
            title = f"River Simulation Frame {frame_index}/{total_frames-1} | Time: {timestamp:.2f}s"
        else:
            title = f"River Simulation Frame {frame_index} | Time: {timestamp:.2f}s"
        title_actor = plotter.add_text(title, font_size=12)  # type: ignore
        if actors is not None:
            actors["title"] = title_actor
    
    # Get parameters
    bank_width = metadata.get("bank_width", 0)
    height = metadata["height"]
    width = metadata["width"]
    
    # Update or create terrain mesh
    if terrain_vertices is not None:
        terrain_triangles = []
        if "terrain_indices" in frame:
            terrain_indices_set = set(frame["terrain_indices"])
            bed_triangles = triangles_data.get("bed", [])
            for triangle in bed_triangles:
                if all(idx in terrain_indices_set for idx in triangle):
                    remapped_triangle = [frame["terrain_indices"].index(idx) for idx in triangle]
                    terrain_triangles.append(remapped_triangle)
        
        if len(terrain_triangles) > 0:
            terrain_mesh = createPyVistaMesh(terrain_vertices, terrain_triangles)
            if terrain_mesh.n_points > 0:  # type: ignore
                if actors is not None and "terrain" in actors and actors["terrain"] is not None:
                    # Update existing mesh in place to avoid flashing
                    try:
                        # Get the underlying PolyData from the actor
                        actor_mesh = actors["terrain"].mapper.dataset  # type: ignore
                        # Update points if count matches
                        if actor_mesh.n_points == terrain_mesh.n_points:  # type: ignore
                            # Use VTK's direct point update method
                            vtk_points = actor_mesh.GetPoints()  # type: ignore
                            if vtk_points is not None:
                                # Update points directly in VTK
                                vtk_points.SetData(terrain_mesh.GetPoints().GetData())  # type: ignore
                                actor_mesh.GetPoints().Modified()  # type: ignore
                                actor_mesh.Modified()  # type: ignore
                                # Force mapper to update
                                actors["terrain"].mapper.Update()  # type: ignore
                            else:
                                # Fallback to array update
                                actor_mesh.points = terrain_mesh.points  # type: ignore
                                actor_mesh.modified()  # type: ignore
                                actors["terrain"].mapper.update()  # type: ignore
                        else:
                            # Point count changed, need to replace (this causes flashing but is necessary)
                            plotter.remove_actor(actors["terrain"])  # type: ignore
                            terrain_actor = plotter.add_mesh(terrain_mesh, color="green", opacity=0.7, show_edges=False, smooth_shading=True)  # type: ignore
                            actors["terrain"] = terrain_actor
                    except Exception as e:
                        # Fallback: remove and re-add (this will cause flashing)
                        plotter.remove_actor(actors["terrain"])  # type: ignore
                        terrain_actor = plotter.add_mesh(terrain_mesh, color="green", opacity=0.7, show_edges=False, smooth_shading=True)  # type: ignore
                        actors["terrain"] = terrain_actor
                else:
                    # First frame - add new mesh (disable edges to reduce texture usage)
                    terrain_actor = plotter.add_mesh(terrain_mesh, color="green", opacity=0.7, show_edges=False, smooth_shading=True)  # type: ignore
                    if actors is not None:
                        actors["terrain"] = terrain_actor
    
    # Update or create bed/bank mesh
    bed_triangles = triangles_data.get("bed", [])
    if "terrain_indices" in frame and "bed_indices" in frame:
        terrain_indices_set = set(frame["terrain_indices"])
        bed_indices_set = set(frame["bed_indices"])
        bed_bank_triangles = []
        for triangle in bed_triangles:
            if not all(idx in terrain_indices_set for idx in triangle):
                try:
                    remapped_triangle = [frame["bed_indices"].index(idx) for idx in triangle if idx in bed_indices_set]
                    if len(remapped_triangle) == 3:
                        bed_bank_triangles.append(remapped_triangle)
                except ValueError:
                    pass
        bed_triangles = bed_bank_triangles if bed_bank_triangles else bed_triangles
    
    bed_mesh = createPyVistaMesh(bed_vertices, bed_triangles)
    if bed_mesh.n_points > 0:  # type: ignore
        if actors is not None and "bed" in actors and actors["bed"] is not None:
            # Update existing mesh in place to avoid flashing
            try:
                # Get the underlying PolyData from the actor
                actor_mesh = actors["bed"].mapper.dataset  # type: ignore
                # Update points if count matches
                if actor_mesh.n_points == bed_mesh.n_points:  # type: ignore
                    # Use VTK's direct point update method
                    vtk_points = actor_mesh.GetPoints()  # type: ignore
                    if vtk_points is not None:
                        # Update points directly in VTK
                        vtk_points.SetData(bed_mesh.GetPoints().GetData())  # type: ignore
                        actor_mesh.GetPoints().Modified()  # type: ignore
                        actor_mesh.Modified()  # type: ignore
                        # Force mapper to update
                        actors["bed"].mapper.Update()  # type: ignore
                    else:
                        # Fallback to array update
                        actor_mesh.points = bed_mesh.points  # type: ignore
                        actor_mesh.modified()  # type: ignore
                        actors["bed"].mapper.update()  # type: ignore
                else:
                    # Point count changed, need to replace (this causes flashing but is necessary)
                    plotter.remove_actor(actors["bed"])  # type: ignore
                    bed_actor = plotter.add_mesh(bed_mesh, color="saddlebrown", opacity=0.8, show_edges=False, smooth_shading=True)  # type: ignore
                    actors["bed"] = bed_actor
            except Exception as e:
                # Fallback: remove and re-add (this will cause flashing)
                plotter.remove_actor(actors["bed"])  # type: ignore
                bed_actor = plotter.add_mesh(bed_mesh, color="saddlebrown", opacity=0.8, show_edges=False, smooth_shading=True)  # type: ignore
                actors["bed"] = bed_actor
        else:
            # First frame - add new mesh (disable edges to reduce texture usage)
            bed_actor = plotter.add_mesh(bed_mesh, color="saddlebrown", opacity=0.8, show_edges=False, smooth_shading=True)  # type: ignore
            if actors is not None:
                actors["bed"] = bed_actor
    
    # Update or create fluid surface mesh
    if fluid_vertices is not None:
        fluid_height = height - 2 * bank_width if bank_width > 0 else height
        num_vertices_y_fluid = fluid_height + 1
        fluid_triangles = []
        
        for i in range(width):
            for j in range(fluid_height):
                bottom_left = i * num_vertices_y_fluid + j
                bottom_right = i * num_vertices_y_fluid + (j + 1)
                top_left = (i + 1) * num_vertices_y_fluid + j
                top_right = (i + 1) * num_vertices_y_fluid + (j + 1)
                
                fluid_triangles.append([bottom_left, top_left, bottom_right])
                fluid_triangles.append([bottom_right, top_left, top_right])
        
        fluid_mesh = createPyVistaMesh(fluid_vertices, fluid_triangles)
        if fluid_mesh.n_points > 0:  # type: ignore
            if actors is not None and "fluid" in actors and actors["fluid"] is not None:
                # Update existing mesh in place to avoid flashing
                try:
                    # Get the underlying PolyData from the actor
                    actor_mesh = actors["fluid"].mapper.dataset  # type: ignore
                    # Update points if count matches
                    if actor_mesh.n_points == fluid_mesh.n_points:  # type: ignore
                        # Use VTK's direct point update method
                        vtk_points = actor_mesh.GetPoints()  # type: ignore
                        if vtk_points is not None:
                            # Update points directly in VTK
                            vtk_points.SetData(fluid_mesh.GetPoints().GetData())  # type: ignore
                            actor_mesh.GetPoints().Modified()  # type: ignore
                            actor_mesh.Modified()  # type: ignore
                            # Force mapper to update
                            actors["fluid"].mapper.Update()  # type: ignore
                        else:
                            # Fallback to array update
                            actor_mesh.points = fluid_mesh.points  # type: ignore
                            actor_mesh.modified()  # type: ignore
                            actors["fluid"].mapper.update()  # type: ignore
                    else:
                        # Point count changed, need to replace (this causes flashing but is necessary)
                        plotter.remove_actor(actors["fluid"])  # type: ignore
                        fluid_actor = plotter.add_mesh(fluid_mesh, color="lightblue", opacity=0.6, show_edges=False, smooth_shading=True)  # type: ignore
                        actors["fluid"] = fluid_actor
                except Exception as e:
                    # Fallback: remove and re-add (this will cause flashing)
                    plotter.remove_actor(actors["fluid"])  # type: ignore
                    fluid_actor = plotter.add_mesh(fluid_mesh, color="lightblue", opacity=0.6, show_edges=False, smooth_shading=True)  # type: ignore
                    actors["fluid"] = fluid_actor
            else:
                # First frame - add new mesh (disable edges to reduce texture usage)
                fluid_actor = plotter.add_mesh(fluid_mesh, color="lightblue", opacity=0.6, show_edges=False, smooth_shading=True)  # type: ignore
                if actors is not None:
                    actors["fluid"] = fluid_actor
    
    # Show or update plotter
    if created_new_plotter:
        plotter.show()  # type: ignore
    # Don't render here - let the animation loop handle it for smoother updates
    
    return plotter


def animateFramesGPU(frames: list, triangles_data: dict, metadata: dict):
    """
    Animate through all frames using GPU-accelerated PyVista rendering.
    Uses in-place mesh updates to prevent screen flashing.
    """
    if not PYVISTA_AVAILABLE or pv is None:
        raise ImportError("PyVista is not available. Install with: pip install pyvista")
    
    # Create plotter with better settings for smooth animation
    plotter = pv.Plotter(window_size=[1400, 1000], off_screen=False)  # type: ignore
    
    # Enable double buffering to reduce flashing
    try:
        plotter.renderer.GetRenderWindow().SetDoubleBuffer(1)  # type: ignore
    except Exception:
        pass
    
    # Enable anti-aliasing for smoother rendering (if supported)
    try:
        plotter.enable_anti_aliasing('ssaa')  # type: ignore
    except Exception:
        # Anti-aliasing not available, continue without it
        pass
    
    # Collect all vertices for bounds calculation
    valid_frames = [f for f in frames if ("vertices" in f or "bed_vertices" in f) and 
                    (f.get("vertices") is not None or f.get("bed_vertices") is not None)]
    if len(valid_frames) == 0:
        print("No valid frames to animate!")
        return
    
    all_vertices_list = []
    for frame in valid_frames:
        if "bed_vertices" in frame:
            vertices = frame["bed_vertices"]
        else:
            vertices = frame["vertices"]
        vertices_clean = np.nan_to_num(vertices, nan=0.0, posinf=1e3, neginf=-1e3)
        all_vertices_list.append(vertices_clean)
        
        if "fluid_vertices" in frame and frame["fluid_vertices"] is not None:
            fluid_vertices = np.nan_to_num(frame["fluid_vertices"], nan=0.0, posinf=1e3, neginf=-1e3)
            all_vertices_list.append(fluid_vertices)
    
    if len(all_vertices_list) > 0:
        all_vertices = np.concatenate(all_vertices_list, axis=0)
        valid_vertices = all_vertices[np.isfinite(all_vertices).all(axis=1)]
        if len(valid_vertices) > 0:
            setCameraAboveRiver(plotter, all_vertices)  # Set camera above river
    
    total_frames = len(frames)
    
    # Dictionary to store actor references for smooth updates
    actors = {}
    
    # Show plotter window first (non-blocking)
    plotter.show(interactive_update=True, auto_close=False)  # type: ignore
    
    # Render first frame
    if len(frames) > 0:
        first_frame = frames[0]
        if (("vertices" in first_frame or "bed_vertices" in first_frame) and 
            (first_frame.get("vertices") is not None or first_frame.get("bed_vertices") is not None)):
            plotFrameGPU(first_frame, triangles_data, metadata, 0, 
                        plotter=plotter, total_frames=total_frames, actors=actors)
            plotter.render()  # type: ignore
    
    for frame_idx, frame in enumerate(frames):
        if ("vertices" not in frame and "bed_vertices" not in frame) or \
           (frame.get("vertices") is None and frame.get("bed_vertices") is None):
            continue
        
        # Skip first frame since we already rendered it
        if frame_idx == 0:
            continue
        
        # Disable rendering during updates to prevent flashing
        try:
            plotter.renderer.GetRenderWindow().SetSwapBuffers(0)  # type: ignore
        except Exception:
            pass
        
        # Plot frame (updates meshes in place to avoid flashing)
        plotFrameGPU(frame, triangles_data, metadata, frame_idx, 
                    plotter=plotter, total_frames=total_frames, actors=actors)
        
        # Re-enable rendering and render the updated scene
        try:
            plotter.renderer.GetRenderWindow().SetSwapBuffers(1)  # type: ignore
        except Exception:
            pass
        
        # Render the updated scene - use render() for consistent rendering
        # This ensures all updates are complete before displaying
        plotter.render()  # type: ignore
        
        # Small delay to ensure rendering completes
        import time
        time.sleep(0.01)  # Small delay for rendering
        
        # Pause for Delta Time
        time.sleep(metadata["dt"])
    
    # Keep window open
    plotter.show()  # type: ignore


if __name__ == "__main__":    
    visualiseSimulation(filename="simulation_frames.pkl", animate=True, frame_index=0, use_gpu=False)

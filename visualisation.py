import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional

from simulation import Simulation

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

def visualiseSimulation(filename:str="simulation_frames.pkl", frame_index:int=0, animate:bool=False):
    """
    Visualise simulation frame. If animate is True, animate through all frames otherwise display single frame.
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

if __name__ == "__main__":    
    visualiseSimulation(filename="simulation_frames.pkl", animate=True, frame_index=0)

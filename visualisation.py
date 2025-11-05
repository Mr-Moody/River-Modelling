import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional

from simulation import Simulation

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
    
    if animate:
        animateFrames(frames, triangles_indices, metadata)

    else:
        frame = frames[frame_index]
        plotFrame(frame, triangles_indices, metadata, frame_index)


def createMeshFromVertices(vertices:np.ndarray, triangles_indices:list) -> Poly3DCollection:
    """
    Helper function to create a Poly3DCollection from vertices and triangle indices.
    """
    triangles = []

    for idx in triangles_indices:
        v0 = vertices[idx[0]]
        v1 = vertices[idx[1]]
        v2 = vertices[idx[2]]

        triangles.append([v0, v1, v2])
    
    return Poly3DCollection(triangles, alpha=0.6, facecolor="lightblue", edgecolor="blue", linewidths=0.3)


def setupAxes(ax, vertices:np.ndarray):
    """
    Helper function to set up axis limits and labels.
    """
    x_range = [vertices[:, 0].min(), vertices[:, 0].max()]
    y_range = [vertices[:, 1].min(), vertices[:, 1].max()]
    z_range = [vertices[:, 2].min(), vertices[:, 2].max()]
    
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_zlim(z_range[0], z_range[1])
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def plotFrame(frame:dict, triangles_indices:list, metadata:dict, frame_index:int, ax=None, total_frames:Optional[int]=None):
    """
    Plot a single frame on the figure.
    
    Args:
        frame: Frame data dictionary
        triangles_indices: List of triangle vertex indices
        metadata: Simulation metadata
        frame_index: Index of the frame
        ax: Optional existing axes to plot on (for animation)
        total_frames: Total number of frames (for animation title)
    
    """

    vertices = frame["vertices"]
    timestamp = frame["timestamp"]
    
    created_new_figure = False
    
    # Create new figure and axes if not provided
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        setupAxes(ax, vertices)

        ax.set_title(f"Simulation Frame {frame_index} | Time: {timestamp:.2f}s")

        created_new_figure = True
    else:
        # Clear existing mesh for animation
        while ax.collections:
            ax.collections[0].remove()

        # Update title for animation
        if total_frames is not None:
            ax.set_title(f"Simulation Frame {frame_index}/{total_frames-1} | Time: {timestamp:.2f}s")
        else:
            ax.set_title(f"Simulation Frame {frame_index} | Time: {timestamp:.2f}s")
    
    # Show new mesh
    mesh_collection = createMeshFromVertices(vertices, triangles_indices)
    ax.add_collection3d(mesh_collection)
    
    # Show or update figure
    if created_new_figure:
        plt.show()
    else:
        plt.draw()


def animateFrames(frames:list, triangles_indices:list, metadata:dict):
    """
    Animate through all frames.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    all_vertices = np.concatenate([frame["vertices"] for frame in frames], axis=0)
    setupAxes(ax, all_vertices)

    total_frames = len(frames)

    for frame_idx, frame in enumerate(frames):
        plotFrame(frame, triangles_indices, metadata, frame_idx, ax=ax, total_frames=total_frames)

        # Pause for Delta Time
        plt.pause(metadata["dt"])
    
    # Keeps showing final frame in figure
    plt.show()

if __name__ == "__main__":    
    visualiseSimulation(filename="simulation_frames.pkl", animate=True, frame_index=0)

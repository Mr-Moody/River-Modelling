"""
Diagnostic tool to check sediment conservation and identify issues.
"""

import numpy as np
import pickle
from simulation import Simulation

def check_sediment_conservation(filename="simulation_frames.pkl"):
    """
    Check if sediment is conserved in the simulation.
    """
    data = Simulation.loadFromFile(filename)
    frames = data["frames"]
    metadata = data["metadata"]
    
    width = metadata["width"]
    height = metadata["height"]
    cell_size = metadata["cell_size"]
    area = cell_size * cell_size  # Area of each cell
    
    print("Checking sediment conservation...")
    print(f"Grid: {width}x{height}, cell_size={cell_size}m")
    print(f"Cell area: {area} m²\n")
    
    # Calculate total sediment volume (bed elevation * area) for each frame
    total_sediment = []
    
    for i, frame in enumerate(frames):
        vertices = frame["vertices"]
        # Bed elevation is z-coordinate
        bed_elevations = vertices[:, 2]
        
        # Total sediment volume = sum of bed elevations * cell area
        # For a grid with (width+1) x (height+1) vertices, we have width x height cells
        # For simplicity, use average elevation per cell
        total_volume = np.sum(bed_elevations) * area
        total_sediment.append(total_volume)
        
        if i == 0:
            initial_volume = total_volume
        elif i % 20 == 0:
            change = total_volume - initial_volume
            change_pct = (change / abs(initial_volume)) * 100 if initial_volume != 0 else 0
            print(f"Frame {i}: Total volume = {total_volume:.4f} m³, "
                  f"Change = {change:+.4f} m³ ({change_pct:+.2f}%)")
    
    print(f"\nInitial volume: {total_sediment[0]:.4f} m³")
    print(f"Final volume: {total_sediment[-1]:.4f} m³")
    print(f"Total change: {total_sediment[-1] - total_sediment[0]:+.4f} m³")
    print(f"Change percentage: {((total_sediment[-1] - total_sediment[0]) / abs(total_sediment[0])) * 100:+.2f}%")
    
    # Check for conservation (should be ~0 change)
    if abs(total_sediment[-1] - total_sediment[0]) > 0.01:
        print("\n⚠️  WARNING: Sediment is NOT conserved!")
        print("   This indicates sediment is being created or destroyed.")
    else:
        print("\n✓ Sediment appears to be conserved.")
    
    # Check bed elevation statistics
    print("\nBed elevation statistics:")
    final_vertices = frames[-1]["vertices"]
    final_elevations = final_vertices[:, 2]
    print(f"  Min: {final_elevations.min():.4f} m")
    print(f"  Max: {final_elevations.max():.4f} m")
    print(f"  Mean: {final_elevations.mean():.4f} m")
    print(f"  Std: {final_elevations.std():.4f} m")
    print(f"  Range: {final_elevations.max() - final_elevations.min():.4f} m")
    
    # Check if values are hitting the bounds
    if final_elevations.max() >= 9.9 or final_elevations.min() <= -9.9:
        print("\n⚠️  WARNING: Bed elevations are hitting the clipping bounds!")
        print("   This can cause artificial accumulation.")

if __name__ == "__main__":
    check_sediment_conservation()


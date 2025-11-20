"""
Test script for coupled Navier-Stokes and Exner equations.
This demonstrates a simple 1D channel test case as suggested.
"""

import numpy as np
from simulation import Simulation

def test_1d_channel():
    """
    Test with a simple 1D channel (small grid, longer in x-direction).
    """
    print("Running 1D channel test...")
    
    # Create a narrow channel (wide in x, narrow in y)
    sim = Simulation(
        width=20,   # x-direction (flow direction)
        height=5,   # y-direction (cross-channel)
        cell_size=0.5,  # 0.5m cells
        dt=0.001,   # Small time step for stability
        nu=1e-6,    # Kinematic viscosity
        rho=1000.0, # Water density
        g=9.81,     # Gravity
        sediment_density=2650.0,
        porosity=0.4,
        critical_shear=0.05,
        transport_coefficient=0.1
    )
    
    # Initialize with a simple bed profile (slight slope downstream)
    for i, point in enumerate(sim.grid.points):
        x = point.position[0]
        # Create a gentle downstream slope
        point.position[2] = -0.01 * x  # 1cm drop per 10m
    
    # Update physics with initial bed elevation
    sim._update_physics_from_mesh()
    
    print(f"Initial bed elevation range: {sim.physics.h.min():.4f} to {sim.physics.h.max():.4f} m")
    print(f"Initial velocity range: u={sim.physics.u.min():.4f} to {sim.physics.u.max():.4f} m/s")
    
    # Run simulation for a short duration
    sim.run(duration=1.0, save_interval=0.1, output_file="test_simulation.pkl")
    
    print(f"Final bed elevation range: {sim.physics.h.min():.4f} to {sim.physics.h.max():.4f} m")
    print(f"Final velocity range: u={sim.physics.u.min():.4f} to {sim.physics.u.max():.4f} m/s")
    print("Test completed successfully!")


if __name__ == "__main__":
    test_1d_channel()


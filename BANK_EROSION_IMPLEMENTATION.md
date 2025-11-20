# Bank Erosion Implementation Summary

## Overview
This implementation adds river banks with erosion mechanics and water-wall collision modeling to the river simulation.

## Key Features

### 1. Bank Cell Types
- **FLUID**: River channel cells where water flows
- **BANK_LEFT**: Left bank cells (static walls)
- **BANK_RIGHT**: Right bank cells (static walls)

### 2. Bank Erosion Mechanics
- Banks erode at a different rate than the bed (controlled by `bank_erosion_rate`)
- Banks require higher shear stress to erode (`bank_critical_shear`)
- Erosion is proportional to excess shear stress above the critical threshold
- Models the impact of water hitting bank walls

### 3. Water-Wall Collisions
- **No-slip boundary conditions**: Velocity set to zero at bank walls
- **Collision detection**: Detects when water flows toward banks
- **Velocity reflection**: Water partially bounces off banks (30% reflection with energy loss)
- **Enhanced shear stress**: Additional stress at bank walls due to collisions

### 4. Visualization
- **Bed mesh**: Light blue, represents the river channel
- **Bank meshes**: Brown, represent the left and right banks
- All meshes update over time showing erosion

## Usage

### Creating a Simulation with Banks

```python
from simulation import Simulation

# Create simulation with banks
sim = Simulation(
    width=20, 
    height=20, 
    cell_size=1, 
    dt=0.01,
    bank_width=2,              # Width of banks (in cells)
    bank_erosion_rate=0.3,     # Bank erodes 30% as fast as bed
    bank_critical_shear=0.15,  # Banks need 3x more stress to erode
    bank_height=1.5            # Initial height of banks above bed
)

# Run simulation
sim.run(duration=20.0, save_interval=0.1)
```

### Visualizing Results

```python
from visualisation import visualiseSimulation

# Animate the simulation
visualiseSimulation(
    filename="simulation_frames.pkl", 
    animate=True
)
```

## Physics Details

### Bank Erosion Equation
```
bank_erosion = bank_erosion_rate * transport_coefficient * (tau_excess^1.5)
```
where `tau_excess = max(tau - bank_critical_shear, 0)`

### Collision Stress
When water flows toward a bank wall, additional shear stress is added:
```
collision_stress = rho * collision_factor * v_normal^2
```
where `v_normal` is the velocity component normal to the bank.

### Boundary Conditions
- **Bank cells**: u = 0, v = 0 (no-slip walls)
- **Fluid cells adjacent to banks**: Velocity reflected with 30% coefficient
- **Bank-fluid interface**: Enhanced shear stress from collisions

## Parameters

### PhysicsSolver Parameters
- `bank_width`: Number of cells on each side that are banks (default: 2)
- `bank_erosion_rate`: Relative erosion rate of banks vs bed (0-1, default: 0.3)
- `bank_critical_shear`: Critical shear stress for bank erosion (Pa, default: 0.15)

### Simulation Parameters
- `bank_height`: Initial height of banks above bed (meters, default: 1.5)

## File Structure

- `physics.py`: Enhanced with bank cell types, erosion, and collision mechanics
- `simulation.py`: Manages separate meshes for bed and banks
- `visualisation.py`: Renders bed and banks with different colors

## Notes

- Banks start at a fixed height above the bed
- Bank erosion is slower than bed erosion (more resistant material)
- Water-wall collisions create additional erosion at bank surfaces
- All meshes evolve over time showing the changing river morphology


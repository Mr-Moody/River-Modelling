# Coupled Navier-Stokes and Exner Equations Implementation

## Overview

This implementation solves the coupled Navier-Stokes and Exner equations for river flow with sediment transport and bed evolution. The velocity field influences sediment movement, and the changing bed elevation alters the flow domain.

## Implementation Structure

### Files

1. **`physics.py`** - Contains the `RiverPhysics` class with all physics solvers
2. **`simulation.py`** - Updated `Simulation` class that uses the physics module
3. **`test_coupled_simulation.py`** - Test script for 1D channel case

### Key Components

#### 1. Navier-Stokes Solver (`navier_stokes_step`)

Solves the shallow water equations (depth-averaged Navier-Stokes) for fluid flow:
- **Advection terms**: Uses upwind differencing for numerical stability
- **Pressure gradients**: Derived from water surface elevation
- **Viscous diffusion**: Laplacian term with kinematic viscosity
- **Bed slope effects**: Gravity-driven flow down bed slopes

#### 2. Bed Shear Stress (`compute_shear_stress`)

Computes bed shear stress from velocity field using:
- Manning's roughness coefficient
- Friction velocity derived from flow velocity and water depth
- `τ = ρ * u*²` where `u*` is the friction velocity

#### 3. Sediment Transport (`sediment_flux`, `compute_sediment_flux_vector`)

- Uses Meyer-Peter and Müller type formulation
- `qs = K * (τ - τ_critical)^(3/2)`
- Sediment flux direction follows flow direction

#### 4. Exner Equation (`exner_equation`)

Solves bed evolution equation:
- `dh/dt = -1/(1-porosity) * div(qs)`
- Uses `np.gradient` for flux divergence calculation (as requested)
- Includes stability constraints to prevent numerical instability

#### 5. Boundary Conditions (`apply_boundary_conditions`)

- **Inflow (left)**: Prescribed velocity (0.5 m/s)
- **Outflow (right)**: Zero gradient (Neumann condition)
- **Side walls (top/bottom)**: No-slip walls
- Updates velocity fields based on bed profile `h`

## Simulation Step

The `step()` function in `Simulation` class follows the exact pattern you requested:

```python
# Get current bed elevation from mesh
h = self.physics.h.copy()

# Navier-Stokes step
u, v, p = self.physics.navier_stokes_step(u, v, p, h, dt)

# Compute bed shear stress from velocity
tau = self.physics.compute_shear_stress(u, v, h)

# Sediment transport and Exner equation
qs_x, qs_y = self.physics.compute_sediment_flux_vector(tau, u, v)
qs = self.physics.sediment_flux(tau)

# Solve Exner equation for bed evolution
dh_dt, h_new = self.physics.exner_equation(qs_x, qs_y, h, dt)

# Update boundary conditions based on new bed profile
u, v = self.physics.apply_boundary_conditions(u, v, h_new)

# Update mesh points with new bed elevation and velocities
self._update_mesh_from_physics(h_new, u, v)
```

## Key Features

1. **Coupled System**: Velocity affects sediment transport, and bed evolution affects flow
2. **Static Physical Grid**: Grid geometry remains constant, but velocity boundary conditions adjust based on bed profile
3. **Stability**: Includes stability constraints and upwind differencing
4. **Flexible Parameters**: All physical parameters are configurable

## Physical Parameters

Default values (can be adjusted in `Simulation.__init__`):

- `nu = 1e-6` m²/s (kinematic viscosity)
- `rho = 1000.0` kg/m³ (water density)
- `g = 9.81` m/s² (gravity)
- `sediment_density = 2650.0` kg/m³
- `porosity = 0.4` (bed porosity)
- `critical_shear = 0.05` Pa (critical shear stress for sediment motion)
- `transport_coefficient = 0.1` (sediment transport coefficient)

## Usage

### Basic Usage

```python
from simulation import Simulation

# Create simulation
sim = Simulation(
    width=20,
    height=20,
    cell_size=1.0,
    dt=0.01
)

# Run simulation
sim.run(duration=10.0, save_interval=0.1)
```

### 1D Channel Test

```python
from test_coupled_simulation import test_1d_channel

# Run 1D channel test case
test_1d_channel()
```

## Numerical Method

- **Time Integration**: Explicit Euler method
- **Spatial Discretization**: Finite differences
- **Advection**: Upwind differencing for stability
- **Gradients**: Central differences for interior points, `np.gradient` for Exner equation

## Stability Considerations

1. **CFL Condition**: Time step should satisfy `dt < dx / max_velocity`
2. **Bed Evolution**: Maximum bed change rate is limited to prevent instability
3. **Water Depth**: Minimum depth enforced to avoid division by zero

## Future Improvements

1. **ODE Solver**: Can integrate with `scipy.integrate.odeint` for higher-order time integration
2. **Adaptive Time Stepping**: Adjust time step based on CFL condition
3. **Better Pressure Solver**: Implement proper pressure projection method for incompressible flow
4. **Advanced Sediment Transport**: Add more sophisticated sediment transport formulas (e.g., van Rijn, Engelund-Hansen)

## Notes

- The implementation uses a shallow water approximation (depth-averaged equations)
- Bed elevation is represented by the z-coordinate of mesh points
- Velocity fields (u, v) are horizontal components only
- The grid remains static; only bed elevation and velocities evolve


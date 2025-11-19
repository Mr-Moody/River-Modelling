import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Domain size and resolution
Lx, Ly, Lz = 10.0, 5.0, 2.0  # length, width, depth in meters
nx, ny, nz = 50, 25, 10      # grid points
dx, dy, dz = Lx/(nx-1), Ly/(ny-1), Lz/(nz-1)

# Time stepping
dt = 0.01                   # time step
nt = 100                    # number of iterations

# Initialize velocity fields (u, v, w), pressure p, sediment concentration c, bed elevation zb
u = np.zeros((nx, ny, nz))
v = np.zeros((nx, ny, nz))
w = np.zeros((nx, ny, nz))
p = np.zeros((nx, ny, nz))
c = np.zeros((nx, ny, nz))  # sediment concentration
zb = np.zeros((nx, ny))     # bed elevation (2D since surface)

# Constants (need updating)
nu = 0 # Kinematic viscosity
rho = 0 # Fluid density
Fx = 0 # Body forces (including gravity)
lambda_p = 0 # Bed porosity

# Initial and boundary conditions (simplified)
u[:, :, :] = 1.0            # initial uniform flow in x-direction
c[:, :, :] = 0.01           # initial sediment concentration

# Poisson solver: Using the Pressure Poisson Equation, the water pressure can be updated
def pressure_poisson(p, div_u, dx, dy, dz, rho, dt, iter_max=50): 
    pn = np.empty_like(p)
    for it in range(iter_max):
        pn[:] = p[:]
        p[1:-1,1:-1,1:-1] = (
            (pn[2:,1:-1,1:-1] + pn[:-2,1:-1,1:-1]) / dx**2 +
            (pn[1:-1,2:,1:-1] + pn[1:-1,:-2,1:-1]) / dy**2 +
            (pn[1:-1,1:-1,2:] + pn[1:-1,1:-1,:-2]) / dz**2
            - (rho / dt) * div_u[1:-1,1:-1,1:-1]
        ) / (2/dx**2 + 2/dy**2 + 2/dz**2)

        # Boundary conditions for p (Neumann zero-gradient)
        p[0,:,:] = p[1,:,:]
        p[-1,:,:] = p[-2,:,:]
        p[:,0,:] = p[:,1,:]
        p[:,-1,:] = p[:,-2,:]
        p[:,:,0] = p[:,:,1]
        p[:,:,-1] = p[:,:,-2]

    return p

# Corrects the calculated velocity to enforce divergence-free field (https://books.physics.oregonstate.edu/GSF/divfree.html)
def correct_velocity(u, v, w, p, dx, dy, dz, rho, dt):
    u[1:-1,:,:] -= dt / rho * (p[2:,:,:] - p[1:-1,:,:]) / dx
    v[:,1:-1,:] -= dt / rho * (p[:,2:,:] - p[:,1:-1,:]) / dy
    w[:,:,1:-1] -= dt / rho * (p[:,:,2:] - p[:,:,1:-1]) / dz
    return u, v, w

# STEP 1: Update the continuity and momentum
def update_flow(u, v, w, p, dx, dy, dz, dt, rho, nu, Fx, iter_poisson=50):

    # Calculate advection terms (nonlinear transport) for velocity fields
    du_dx = (u[1:, :, :] - u[:-1, :, :]) / dx
    du_dy = (u[:, 1:, :] - u[:, :-1, :]) / dy
    du_dz = (u[:, :, 1:] - u[:, :, :-1]) / dz

    # Calculate diffusion terms (viscous terms) for velocity fields
    d2u_dx2 = (u[2:, :, :] - 2*u[1:-1, :, :] + u[:-2, :, :]) / dx**2
    d2u_dy2 = (u[:, 2:, :] - 2*u[:, 1:-1, :] + u[:, :-2, :]) / dy**2
    d2u_dz2 = (u[:, :, 2:] - 2*u[:, :, 1:-1] + u[:, :, :-2]) / dz**2

    # Update intermediate u velocity ignoring pressure gradient and pressure correction
    # This advances the momentum equation in time, including effects calculated previously
    u_star = u.copy()
    u_star[1:-1, 1:-1, 1:-1] += dt * (
        - u[1:-1, 1:-1, 1:-1] * du_dx[1:-1, 1:-1, 1:-1]
        - v[1:-1, 1:-1, 1:-1] * du_dy[1:-1, 1:-1, 1:-1]
        - w[1:-1, 1:-1, 1:-1] * du_dz[1:-1, 1:-1, 1:-1]
        + nu * (d2u_dx2 + d2u_dy2 + d2u_dz2)
        + Fx  # body forces
    )

    # Step 2: Calculate divergence of intermediate velocity
    # Measures how much the velocity field violates the continuity equation
    div_u_star = (
        (u_star[1:, :, :] - u_star[:-1, :, :]) / dx +
        (v[ :, 1:, :] - v[:, :-1, :]) / dy +
        (w[ :, :, 1:] - w[:, :, :-1]) / dz
    )

    # Step 3: Solve Poisson equation for pressure (to enforce incompressibility)
    # Uses divergence of the intermediate velocity as the source
    # Ensures final velocity field will be divergence-free (satisfies continuity)
    p = pressure_poisson(p, div_u_star, dx, dy, dz, rho, dt, iter_max=iter_poisson)

    # Step 4: Correct velocities with updated pressure
    # Subtracting gradient of updated pressure, scaling by density and time step
    # Enforces incompressibility, ensure final velocity field satisfies the continuity equation
    u_new, v_new, w_new = correct_velocity(u_star, v, w, p, dx, dy, dz, rho, dt)

    return u_new, v_new, w_new, p

# STEP 2: Updates the bed sediment level
# Calculates spatial gradient of sediment concenctration (c) in the x, y, z directions
def update_sediment(c, u, v, w):
    # Simple advection-diffusion for sediment (simplified)
    # Diffusion and vertical transport not included here for brevity
    dc_dx = (c[1:, :, :] - c[:-1, :, :]) / dx
    dc_dy = (c[:, 1:, :] - c[:, :-1, :]) / dy
    dc_dz = (c[:, :, 1:] - c[:, :, :-1]) / dz

    # Update concentration 
    c[1:-1, :, :] += dt * (- u[1:-1, :, :] * dc_dx[1:-1, :, :])
    c[:, 1:-1, :] += dt * (- v[:, 1:-1, :] * dc_dy[:, 1:-1, :])
    c[:, :, 1:-1] += dt * (- w[:, :, 1:-1] * dc_dz[:, :, 1:-1])

    return c

def update_bed(zb, qbx, qby, dx, dy, dt, lambda_p):
    # Compute spatial derivatives (divergence) of sediment fluxes
    dqbx_dx = (qbx[1:, :] - qbx[:-1, :]) / dx  # shape (nx-1, ny)
    dqby_dy = (qby[:, 1:] - qby[:, :-1]) / dy  # shape (nx, ny-1)

    # Create arrays with same shape as zb
    div_qb = np.zeros_like(zb)

    # Assign divergence components to interior points (average to align in grid)
    div_qb[:-1, :] += dqbx_dx
    div_qb[:, :-1] += dqby_dy

    # Update bed elevation according to Exner equation (forward Euler)
    zb -= (dt / (1 - lambda_p)) * div_qb

    return zb

# Calculating bedload transport
# Extract bed layer sediment concentration and velocity
c_bed = c[:, :, 0]      # sediment concentration at bed
u_bed = u[:, :, 0]      # velocity x-component at bed
v_bed = v[:, :, 0]      # velocity y-component at bed

alpha = 1.0  # empirical parameter
qbx = alpha * c_bed * u_bed
qby = alpha * c_bed * v_bed

for t in range(nt):
    u, v, w = update_flow(u, v, w, p)
    c = update_sediment(c, u, v, w)
    zb = update_bed(zb, qbx, qby, dx, dy, dz, c)
    if t % 10 == 0:
        print(f'Time step {t}, max velocity: {np.max(u):.3f}, max sediment concentration: {np.max(c):.3f}')

# Visualization: plot bed elevation surface
X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, zb.T, cmap='viridis')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Bed Elevation (m)')
plt.show()

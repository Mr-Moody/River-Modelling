import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Domain size and resolution
Lx, Ly, Lz = 10.0, 5.0, 2.0
nx, ny, nz = 50, 25, 10
dx, dy, dz = Lx/(nx-1), Ly/(ny-1), Lz/(nz-1)

# Time stepping
dt = 0.001      # reduced time step for stability!
nt = 1000

# Grids for bed manipulation (make sure shape is (nx, ny) to match zb)
X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), indexing='ij')

# Initialize main fields (velocity, pressure, sediment, bed elevation)
u = np.zeros((nx, ny, nz))
v = np.zeros((nx, ny, nz))
w = np.zeros((nx, ny, nz))
p = np.zeros((nx, ny, nz))
c = np.zeros((nx, ny, nz))
zb = np.zeros((nx, ny))

# Physical constants
nu = 1e-6
rho = 1000
Fx = 0.1
lambda_p = 0.35

# Initial velocity and sediment
u[:, :, :] = 1.0
c[:, :, :] = 0.01

# Bed perturbation
zb += 0.05 * np.exp(-((X-5)**2 + (Y-2.5)**2))
zb += 0.05 * (X - Lx/2) / Lx

# Extra sediment at upstream boundary
c[0,:,:] = 0.2

# Turbulence model
k = np.full((nx, ny, nz), 0.01)
epsilon = np.full((nx, ny, nz), 0.01)
C_mu = 0.09

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
        p[0,:,:] = p[1,:,:]
        p[-1,:,:] = p[-2,:,:]
        p[:,0,:] = p[:,1,:]
        p[:,-1,:] = p[:,-2,:]
        p[:,:,0] = p[:,:,1]
        p[:,:,-1] = p[:,:,-2]
    return p

def correct_velocity(u, v, w, p, dx, dy, dz, rho, dt):
    u[1:-1,:,:] -= dt / rho * (p[2:,:,:] - p[1:-1,:,:]) / dx
    v[:,1:-1,:] -= dt / rho * (p[:,2:,:] - p[:,1:-1,:]) / dy
    w[:,:,1:-1] -= dt / rho * (p[:,:,2:] - p[:,:,1:-1]) / dz
    return u, v, w

def update_flow(u, v, w, p, dx, dy, dz, dt, rho, nu_eff, Fx, iter_poisson=50):
    du_dx = (u[2:,1:-1,1:-1] - u[:-2,1:-1,1:-1]) / (2*dx)
    du_dy = (u[1:-1,2:,1:-1] - u[1:-1,:-2,1:-1]) / (2*dy)
    du_dz = (u[1:-1,1:-1,2:] - u[1:-1,1:-1,:-2]) / (2*dz)
    d2u_dx2 = (u[2:,1:-1,1:-1] - 2*u[1:-1,1:-1,1:-1] + u[:-2,1:-1,1:-1]) / dx**2
    d2u_dy2 = (u[1:-1,2:,1:-1] - 2*u[1:-1,1:-1,1:-1] + u[1:-1,:-2,1:-1]) / dy**2
    d2u_dz2 = (u[1:-1,1:-1,2:] - 2*u[1:-1,1:-1,1:-1] + u[1:-1,1:-1,:-2]) / dz**2
    u_star = u.copy()
    u_star[1:-1, 1:-1, 1:-1] += dt * (
        - u[1:-1, 1:-1, 1:-1] * du_dx
        - v[1:-1, 1:-1, 1:-1] * du_dy
        - w[1:-1, 1:-1, 1:-1] * du_dz
        + nu_eff[1:-1, 1:-1, 1:-1] * (d2u_dx2 + d2u_dy2 + d2u_dz2)
        + Fx
    )
    div_u_star = (
        (u_star[2:,1:-1,1:-1] - u_star[:-2,1:-1,1:-1])/(2*dx) +
        (v[1:-1,2:,1:-1] - v[1:-1,:-2,1:-1])/(2*dy) +
        (w[1:-1,1:-1,2:] - w[1:-1,1:-1,:-2])/(2*dz)
    )
    div_u_grid = np.zeros_like(u)
    div_u_grid[1:-1, 1:-1, 1:-1] = div_u_star
    p = pressure_poisson(p, div_u_grid, dx, dy, dz, rho, dt, iter_max=iter_poisson)
    u_new = u_star.copy()
    u_new[1:-1, 1:-1, 1:-1] -= dt / rho * (p[2:,1:-1,1:-1] - p[:-2,1:-1,1:-1]) / (2*dx)
    v_new = v.copy()
    v_new[1:-1, 1:-1, 1:-1] -= dt / rho * (p[1:-1,2:,1:-1] - p[1:-1,:-2,1:-1]) / (2*dy)
    w_new = w.copy()
    w_new[1:-1, 1:-1, 1:-1] -= dt / rho * (p[1:-1,1:-1,2:] - p[1:-1,1:-1,:-2]) / (2*dz)
    return u_new, v_new, w_new, p

def update_turbulence(u, v, w, k, epsilon, dx, dy, dz, dt):
    sigma_k, sigma_epsilon, C1, C2 = 1.0, 1.3, 1.44, 1.92
    du_dx = np.gradient(u, dx, axis=0)
    dv_dy = np.gradient(v, dy, axis=1)
    dw_dz = np.gradient(w, dz, axis=2)
    Pk = k * (np.abs(du_dx) + np.abs(dv_dy) + np.abs(dw_dz))
    dk_dt = Pk - epsilon
    k_new = k + dt * dk_dt
    epsilon_new = epsilon + dt * (
        (C1 * Pk * epsilon / (k + 1e-12)) - (C2 * epsilon**2 / (k + 1e-12))
    )
    k_new = np.maximum(k_new, 1e-8)
    epsilon_new = np.maximum(epsilon_new, 1e-8)
    return k_new, epsilon_new

def eddy_viscosity(k, epsilon, C_mu=0.09):
    return C_mu * k**2 / (epsilon + 1e-12)

def update_sediment(c, u, v, w):
    dc_dx = (c[2:,1:-1,1:-1] - c[:-2,1:-1,1:-1]) / (2*dx)
    dc_dy = (c[1:-1,2:,1:-1] - c[1:-1,:-2,1:-1]) / (2*dy)
    dc_dz = (c[1:-1,1:-1,2:] - c[1:-1,1:-1,:-2]) / (2*dz)
    c[1:-1,1:-1,1:-1] += dt * (
        - u[1:-1,1:-1,1:-1] * dc_dx
        - v[1:-1,1:-1,1:-1] * dc_dy
        - w[1:-1,1:-1,1:-1] * dc_dz
    )
    return c

def update_bed(zb, qbx, qby, dx, dy, dt, lambda_p):
    div_qb = np.zeros_like(zb)
    div_qb[1:-1,1:-1] = (
          (qbx[2:,1:-1] - qbx[:-2,1:-1]) / (2*dx)
        + (qby[1:-1,2:] - qby[1:-1,:-2]) / (2*dy)
    )
    zb_new = zb.copy()
    zb_new[1:-1,1:-1] -= (dt / (1 - lambda_p)) * div_qb[1:-1,1:-1]
    return zb_new

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, Y, zb, cmap='viridis')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Bed Elevation (m)')

for t in range(nt):
    k, epsilon = update_turbulence(u, v, w, k, epsilon, dx, dy, dz, dt)
    nu_t = eddy_viscosity(k, epsilon, C_mu)
    nu_eff = nu + nu_t
    u, v, w, p = update_flow(u, v, w, p, dx, dy, dz, dt, rho, nu_eff, Fx)
    c_bed = c[:, :, 0]
    u_bed = u[:, :, 0]
    v_bed = v[:, :, 0]

    alpha = 0.01   # stability: much smaller than before
    qbx = alpha * c_bed * u_bed
    qby = alpha * c_bed * v_bed

    # ENFORCE ZERO-FLUX AT BOUNDARIES
    qbx[0,:] = 0
    qbx[-1,:] = 0
    qby[:,0] = 0
    qby[:,-1] = 0

    c = update_sediment(c, u, v, w)
    zb = update_bed(zb, qbx, qby, dx, dy, dt, lambda_p)
    zb = np.clip(zb, -1, 1)  # optional: clamp for extra debugging

    if t % 50 == 0 or t == nt-1:
        ax.clear()
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Bed Elevation (m)')
        surface = ax.plot_surface(X, Y, zb, cmap='viridis')
        plt.title(f'Bed Elevation, step={t}\nmin={zb.min():.3e}, max={zb.max():.3e}')
        plt.pause(0.05)
        # Print min/max to debug possible blow-up early
        print(f"Step {t}: zb min {zb.min():.3e}, max {zb.max():.3e}")

plt.ioff()
plt.show()

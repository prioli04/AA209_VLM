# VLM - Time Stepping Wake
# Rectangular Wing
# No sweep
# No twist
# No dihedral
# Symmetric airfoil

import cProfile
import VLM

b = 2.5 # m
AR = 5.0 # -
S = b**2 / AR # m^2
V_inf = 12.0 # m/s
wake_dt = 0.5 * (b / AR) / V_inf

params = VLM.Parameters(
    V_inf = V_inf, 
    alfa_deg = 5.0, 
    rho = 1.225, 
    S = S, 
    AR = AR, 
    b = b, 

    n_wake_deform = 5, 
    wake_steps = 20, 
    wake_dt = wake_dt, 
    wake_dx = 0.3 * wake_dt * V_inf,

    CL_tol = 1e-4,
    CD_tol = 1e-5
)

nx, ny = 3, 20
panels = VLM.Panels(params, nx, ny, plot=False)
solver = VLM.Solver(panels, params)
results = solver.solve()
# cProfile.run("solver.solve()")
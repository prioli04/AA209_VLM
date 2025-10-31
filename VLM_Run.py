# VLM - Time Stepping Wake
# Rectangular Wing
# No sweep
# No twist
# No dihedral
# Symmetric airfoil

import cProfile
import VLM

params = VLM.Parameters(
    V_inf = 12.0, 
    alfa_deg = 5.0, 
    rho = 1.225, 
    AR = 5.0, 
    b = 2.5, 

    n_wake_deform = 5, 
    wake_steps = 20, 
    wake_dt_fact = 0.5, 
    wake_dx_fact = 0.3,

    CL_tol = 1e-4,
    CD_tol = 1e-5
)

nx, ny = 20, 30
panels = VLM.Panels(params, nx, ny, plot=False)
solver = VLM.Solver(panels, params)
results = solver.solve()
# cProfile.run("solver.solve()")
# VLM - Time Stepping Wake
# No twist
# No dihedral
# Symmetric airfoil

import cProfile
import VLM

sections = [
    VLM.Section(fy_pos=0.0, fc=1.0, x_offset=0.0),
    VLM.Section(fy_pos=0.5, fc=0.8, x_offset=0.0),
    VLM.Section(fy_pos=1.0, fc=0.5, x_offset=0.1)
]

params = VLM.Parameters(
    V_inf = 12.0, 
    alfa_deg = 15.0, 
    rho = 1.225, 
    AR = 5.0, 
    b = 2.5, 

    n_wake_deform = 5, 
    wake_dt_fact = 0.5, 
    wake_dx_fact = 0.3,

    CL_tol = 1e-1,
    CD_tol = 1e-1,

    sym=True,
    ground=False
)

nx, ny = 20, 30
panels = VLM.Panels(params, sections, nx, ny, plot=True)
solver = VLM.Solver(panels, params)
results = solver.solve()

input("Press any key to exit.")
# cProfile.run("solver.solve()")
# VLM - Time Stepping Wake
# Rectangular Wing
# No sweep
# No twist
# No dihedral
# Symmetric airfoil

import cProfile
import VLM

b = 2.5 # m
AR = 4.0 # -
S = b**2 / AR # m^2

rho = 1.225 # kg/m^3
V_inf = 12.0 # m/s
alfa = 5.0 # °

nx, ny = 4, 13
wake_steps = 5
wake_dt = 10 * (b / AR) / V_inf
wake_dx = 0.3 * wake_dt * V_inf

panels = VLM.Panels(b, AR, nx, ny, wake_dx, wake_steps)
params = VLM.Parameters(V_inf, alfa, rho, S, AR, wake_steps, wake_dt, wake_dx)
solver = VLM.Solver(panels, params)
results = solver.solve()
# cProfile.run("solver.solve()")

# solver.print_results()
panels.plot_model()
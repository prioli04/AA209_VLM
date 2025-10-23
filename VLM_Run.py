# VLM - Fixed Wake
# Rectangular Wing
# No sweep
# No twist
# No dihedral
# Symmetric airfoil

import VLM

b = 2.5 # m
AR = 5.0 # -
S = b**2 / AR # m^2

rho = 1.225 # kg/m^3
V_inf = 12.0 # m/s
alfa = 5.0 # °

params = VLM.Parameters(V_inf, alfa, rho, S, AR)

nx, ny = 3, 4
mesh = VLM.Mesh(b, AR, nx, ny)
mesh.plot_mesh()

solver = VLM.Solver(mesh, params)
results = solver.solve()
solver.print_results()
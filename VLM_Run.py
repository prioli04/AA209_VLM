# VLM - Time Stepping Wake
# No dihedral

import cProfile
import VLM
from VLM.WingPatch import DiscretizationType

import numpy as np
import matplotlib.pyplot as plt

sections = [
    VLM.Section(fy_pos=0.0, fc=1.0, x_offset=0.0, twist_deg=0.0, airfoil_path_str="naca0012.dat", xfoil_path_str="NACA0012_XFOIL.OUT"),
    # VLM.Section(fy_pos=0.5, fc=1.0 ,x_offset=0.0, twist_deg=0.0, airfoil_path_str="naca0012.dat", xfoil_path_str="NACA0012_XFOIL.OUT"),
    VLM.Section(fy_pos=1.0, fc=1.0, x_offset=0.0, twist_deg=0.0, airfoil_path_str="naca0012.dat", xfoil_path_str="NACA0012_XFOIL.OUT")
]

patches = [
    VLM.WingPatch(8, 40, DiscretizationType.COSINE, DiscretizationType.MINUS_SINE),
    # VLM.WingPatch(8, 10, DiscretizationType.COSINE, DiscretizationType.MINUS_SINE)
]

wing_geom = VLM.WingGeometry(sections, patches, b=2.5, AR=10.0)

params = VLM.Parameters(
    V_inf = 12.0, 
    alfa_deg = 18, 
    beta_deg = 0.0,
    rho = 1.225, 
    AR = wing_geom.AR, 
    b = wing_geom.b, 
    MAC = wing_geom.MAC,

    wake_fixed = True,
    sym = True,
    ground = False,
    decambering=True,
    Z = 0.3
)

alfas = np.linspace(0.0, 25.0, 26)
panels = VLM.Panels(wing_geom, params, plot=True)
solver = VLM.Solver(panels, params)
CL_pot = np.zeros_like(alfas)
CL_visc = np.zeros_like(alfas)
delta_decamber = np.zeros(panels._wing_panels._ny)

for i in range(len(alfas)):
    params.decambering = False
    params.alfa_deg = alfas[i]
    solver = VLM.Solver(panels, params)
    results, _ = solver.solve()
    CL_pot[i] = results.CL

    params.decambering = True
    solver = VLM.Solver(panels, params)
    results, delta_decamber = solver.solve(delta_decamber)
    CL_visc[i] = results.CL

plt.figure()
plt.plot(alfas, CL_pot, label="Potential")
plt.plot(panels._wing_panels._airfoils[0]._alfa_visc, panels._wing_panels._airfoils[0]._Cl_visc, label="Foil")
plt.plot(alfas, CL_visc, label="Decambered")
plt.xlim([0, 30])
plt.ylim([0, 2])
plt.legend()
plt.show()

solver.solve()
input("Press any key to exit.")
# cProfile.run("solver.solve()")
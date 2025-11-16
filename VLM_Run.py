# VLM - Time Stepping Wake
# No dihedral

import cProfile
import VLM
from VLM.WingPatch import DiscretizationType

import numpy as np
import matplotlib.pyplot as plt

sections = [
    VLM.Section(fy_pos=0.0, fc=1.0, x_offset=0.0, twist_deg=0.0, airfoil_path_str="naca0012.dat", xfoil_path_str="NACA0012_XFOIL.OUT"),
    VLM.Section(fy_pos=0.5, fc=1.0 ,x_offset=0.0, twist_deg=0.0, airfoil_path_str="naca0012.dat"),
    VLM.Section(fy_pos=1.0, fc=1.0, x_offset=0.0, twist_deg=0.0, airfoil_path_str="naca0012.dat")
]

patches = [
    VLM.WingPatch(12, 24, DiscretizationType.UNIFORM, DiscretizationType.UNIFORM),
    VLM.WingPatch(12, 24, DiscretizationType.UNIFORM, DiscretizationType.UNIFORM)
]

wing_geom = VLM.WingGeometry(sections, patches, b=2.5, AR=5.0)

params = VLM.Parameters(
    V_inf = 12.0, 
    alfa_deg = 5.0, 
    beta_deg = 0.0,
    rho = 1.225, 
    AR = wing_geom.AR, 
    b = wing_geom.b, 
    MAC = wing_geom.MAC,

    wake_fixed = True,
    sym = False,
    ground = True,
    Z = 0.3
)

# panels = VLM.Panels(wing_geom, params, plot=False)
# solver = VLM.Solver(panels, params)
# results = solver.solve()

alfas = np.linspace(0.0, 20.0, 21)
Cl = np.zeros_like(alfas)
Cm = np.zeros_like(alfas)
decamb = VLM.Decambering(sections[0], 20, params)

for (i, alfa) in enumerate(alfas):
    Cl[i], Cm[i] = decamb.solve(alfa)
    
plt.figure()
plt.plot(decamb._alfa_visc, decamb._Cl_visc, label="XFOIL")
plt.plot(alfas, Cl, label="Decambering")
plt.legend()
plt.show()

input("Press any key to exit.")
# cProfile.run("solver.solve()")
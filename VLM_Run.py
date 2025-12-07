# VLM - Time Stepping Wake + Decambering

import VLM
from VLM.WingPatch import DiscretizationType

import numpy as np
import matplotlib.pyplot as plt

sections = [
    VLM.Section(fy_pos=0.0, fc=1.0, x_offset=0.0, z_offset=0.0, twist_deg=0.0, airfoil_path_str="Assets/naca0012.dat", xfoil_path_str="Assets/NACA0012_XFOIL.OUT"),
    VLM.Section(fy_pos=1.0, fc=1.0, x_offset=0.0, z_offset=0.0, twist_deg=0.0, airfoil_path_str="Assets/naca0012.dat", xfoil_path_str="Assets/NACA0012_XFOIL.OUT")
]

patches = [
    VLM.WingPatch(3, 10, DiscretizationType.COSINE, DiscretizationType.MINUS_SINE),
]

wing_geom = VLM.WingGeometry(sections, patches, b=1.0, AR=8.0)

params = VLM.Parameters(
    alfa_deg = 20.0, 
    AR = wing_geom.AR, 
    MAC = wing_geom.MAC,
    r_ref=np.array([0.125, 0.0, 0.0]),
    wake_fixed=True,
    decambering=True
)

alfas = np.linspace(0.0, 25.0, 26)
panels = VLM.Panels(wing_geom, params, plot=True)
solver = VLM.Solver(panels, params)
CL_pot = np.zeros_like(alfas)
CL_visc = np.zeros_like(alfas)
CD_pot = np.zeros_like(alfas)
CD_visc = np.zeros_like(alfas)
CD_p = np.zeros_like(alfas)
delta_decamber = np.zeros(panels._wing_panels._ny)

for i in range(len(alfas)):
    params.decambering = False
    params.alfa_deg = alfas[i]
    solver = VLM.Solver(panels, params, delta_decamber)
    results, _ = solver.solve()
    CL_pot[i] = results.coefs_3D.CL
    CD_pot[i] = results.coefs_3D.CD

    params.decambering = True
    solver = VLM.Solver(panels, params)
    results, delta_decamber = solver.solve()
    CL_visc[i] = results.coefs_3D.CL
    CD_visc[i] = results.coefs_3D.CD
    CD_p[i] = results.CDp

plt.figure()
plt.plot(alfas, CL_pot, label="Potential")
plt.plot(panels._wing_panels._airfoils[0]._alfa_visc, panels._wing_panels._airfoils[0]._Cl_visc, label="Foil")
plt.plot(alfas, CL_visc, label="Decambered")
plt.xlim([0, 30])
plt.ylim([0, 2])
plt.legend()
plt.show(block=False)

plt.figure()
plt.plot(CL_pot, CD_pot, label="Potential")
plt.plot(panels._wing_panels._airfoils[0]._Cl_visc, panels._wing_panels._airfoils[0]._Cd_visc, label="Foil")
plt.plot(CL_visc, CD_visc, label="Decambered")
plt.xlim([0, 2])
plt.legend()
plt.show(block=False)

plt.figure()
plt.plot(alfas, CD_visc, label="CD_tot")
plt.plot(alfas, CD_visc - CD_p, label="CDi")
plt.plot(alfas, CD_p, label="CDp")
plt.xlim([0, 30])
# plt.ylim([0, 2])
plt.legend()
plt.show()

solver.solve(plot=False)
input("Press any key to exit.")
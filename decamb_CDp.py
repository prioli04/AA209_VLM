# Parasite drag effect with decambering model
import VLM
from VLM.WingPatch import DiscretizationType

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Rectangular wing with NACA0012 airfoil
sections = [
    VLM.Section(fy_pos=0.0, fc=1.0, x_offset=0.0, z_offset=0.0, twist_deg=0.0, airfoil_path_str="Assets/naca0012.dat", xfoil_path_str="Assets/NACA0012_XFOIL.out"),
    VLM.Section(fy_pos=1.0, fc=1.0, x_offset=0.0, z_offset=0.0, twist_deg=0.0, airfoil_path_str="Assets/naca0012.dat", xfoil_path_str="Assets/NACA0012_XFOIL.out")
]

# Describe wing discretization
patch = [VLM.WingPatch(16, 32, DiscretizationType.COSINE, DiscretizationType.MINUS_SINE)]

# Make wing geometry
wing_geom = VLM.WingGeometry(sections, patch, b=1.0, AR=10.0)

alfas = np.linspace(0.0, 20.0, 20)
CL_decamb, CD_decamb, CDp_decamb = [], [], []
CL, CD = [], []

for alfa in alfas:
    # Parameters
    params = VLM.Parameters(alfa_deg = alfa, AR = wing_geom.AR, MAC = wing_geom.MAC, r_ref = np.array([0.0, 0.0, 0.0]))
    params_decamb = VLM.Parameters(decambering=True, alfa_deg = alfa, AR = wing_geom.AR, MAC = wing_geom.MAC, r_ref = np.array([0.0, 0.0, 0.0]))

    # Discretize geometry
    panels = VLM.Panels(wing_geom, params)
    panels_decamb = VLM.Panels(wing_geom, params_decamb)

    # Initiate solvers and solve VLM
    solver = VLM.Solver(panels, params)
    result, _ = solver.solve()

    solver_decamb = VLM.Solver(panels_decamb, params_decamb)
    result_decamb, _ = solver_decamb.solve()

    CL.append(result.coefs_3D.CL)
    CD.append(result.coefs_3D.CD)

    CL_decamb.append(result_decamb.coefs_3D.CL)
    CD_decamb.append(result_decamb.coefs_3D.CD)
    CDp_decamb.append(result_decamb.CDp)

# Create plots
fig, axs = plt.subplots(1, 2)
ax_polar: Axes = axs[0]
ax_drag_comp: Axes = axs[1]
fig.subplots_adjust(wspace=0.4)

# Drag polar plot
ax_polar.set_title("Drag polar (AR = 10.0)")
ax_polar.set_xlabel("CL [-]")
ax_polar.set_ylabel("CD [-]")
ax_polar.grid()

ax_polar.plot(CL_decamb, CD_decamb, "k", label="Decambered VLM")
ax_polar.plot(CL, CD, "k--", label="Linear VLM")
ax_polar.legend()

# Drag components comparison plot
ax_drag_comp.set_title("Comparison between drag components (AR = 10.0)")
ax_drag_comp.set_xlabel(r"$\alpha$ [°]")
ax_drag_comp.set_ylabel("CD [-]")
ax_drag_comp.grid()

ax_drag_comp.plot(alfas, CD_decamb, "k", label="Decambered VLM")
ax_drag_comp.plot(alfas, CD, "k--", label="Linear VLM")
ax_drag_comp.plot(alfas, CDp_decamb, "r", label="Parasite drag")
ax_drag_comp.legend()

fig.set_figwidth(10)
fig.set_figheight(5)
plt.savefig("Images/CL_decamb_drag.png")
plt.show()


# Lift coefficient slope as a function of AR for a rectangular wing with symmetrical airfoil

import VLM
from VLM.WingPatch import DiscretizationType

import numpy as np
import matplotlib.pyplot as plt

# Rectangular wing with symmetrical airfoil
sections = [
    VLM.Section(fy_pos=0.0, fc=1.0, x_offset=0.0, z_offset=0.0, twist_deg=0.0, airfoil_path_str="Assets/naca0012.dat"),
    VLM.Section(fy_pos=1.0, fc=1.0, x_offset=0.0, z_offset=0.0, twist_deg=0.0, airfoil_path_str="Assets/naca0012.dat")
]

# Describe wing discretization
patch = [VLM.WingPatch(16, 32, DiscretizationType.COSINE, DiscretizationType.MINUS_SINE)]

ARs = np.linspace(1.0, 12.0, 10)
CL_alfas = []

for AR in ARs:
    # Make wing geometry
    wing_geom = VLM.WingGeometry(sections, patch, b=1.0, AR=AR)

    # Parameters
    params = VLM.Parameters(
        alfa_deg = 5.0, 
        AR = wing_geom.AR, 
        MAC = wing_geom.MAC,
        r_ref = np.array([0.0, 0.0, 0.0]) 
    )

    # Discretize geometry
    panels = VLM.Panels(wing_geom, params)

    # Initiate solver and solve VLM
    solver = VLM.Solver(panels, params)
    result, _ = solver.solve()

    # CL_alfa = [CL(alfa_2) - CL(alfa_1)] / [alfa_2 - alfa_1]
    # Since the wing is symmetrical, for alfa_1 = 0, CL(alfa_1) = 0. Thus, CL_alfa = CL(alfa_2) / alfa_2
    CL_alfas.append(result.coefs_3D.CL / np.deg2rad(5.0)) 

# Plots
plt.figure()
plt.grid()
plt.xlabel("AR [-]")
plt.ylabel(r"$CL_\alpha$ [1/rad]")
plt.title("Effect of aspect ratio in the lift curve slope")

plt.scatter(ARs, CL_alfas, c="r", marker="x", label="VLM for NACA 0012 rectangular wing")

ARs_ellip = np.linspace(1.0, 12.0, 50)
plt.plot(ARs_ellip, 2.0 * np.pi / (1.0 + (2.0 / ARs_ellip)), "k", label="Elliptic loading")
plt.plot([np.min(ARs), np.max(ARs)], [2.0 * np.pi, 2.0 * np.pi], "k--", label="2D lift slope for thin airfoil")
plt.legend()
plt.savefig("Images/CL_alfa_AR.png")
plt.show()
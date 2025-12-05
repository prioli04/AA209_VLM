# Lift coefficient slope as a function of leading edge sweep angle and AR for a planar wing with symmetrical airfoil

import VLM
from VLM.WingPatch import DiscretizationType

import numpy as np
import matplotlib.pyplot as plt

# Function that setup the solver for this analysis
def init_sim(x_off: float, AR: float):
    # Rectangular wing with sweep with symmetrical airfoil
    sections = [
        VLM.Section(fy_pos=0.0, fc=1.0, x_offset=0.0, z_offset=0.0, twist_deg=0.0, airfoil_path_str="Assets/naca0012.dat"),
        VLM.Section(fy_pos=1.0, fc=1.0, x_offset=x_off, z_offset=0.0, twist_deg=0.0, airfoil_path_str="Assets/naca0012.dat")
    ]

    # Describe wing discretization
    patch = [VLM.WingPatch(16, 32, DiscretizationType.COSINE, DiscretizationType.MINUS_SINE)]

    # Make wing geometry
    wing_geom = VLM.WingGeometry(sections, patch, b=1.0, AR=AR)

    # Parameters
    params = VLM.Parameters(alfa_deg = 5.0, AR = wing_geom.AR, MAC = wing_geom.MAC, r_ref = np.array([0.0, 0.0, 0.0]))

    # Discretize geometry
    panels = VLM.Panels(wing_geom, params)

    # Initiate solver
    return VLM.Solver(panels, params)

x_off_30 = 0.5 * np.tan(np.deg2rad(30.0)) # x_offset = 0.5*b * tan(Lambda); b=1 is used in this example
x_off_45 = 0.5 * np.tan(np.deg2rad(45.0))

ARs = np.linspace(0.2, 7.0, 10)
CL_alfas_0, CL_alfas_30, CL_alfas_45 = [], [], []

for AR in ARs:
    solver_0 = init_sim(0.0, AR)
    solver_30 = init_sim(x_off_30, AR)
    solver_45 = init_sim(x_off_45, AR)

    result_0, _ = solver_0.solve()
    result_30, _ = solver_30.solve()
    result_45, _ = solver_45.solve()

    # CL_alfa = [CL(alfa_2) - CL(alfa_1)] / [alfa_2 - alfa_1]
    # Since the wing is symmetrical, for alfa_1 = 0, CL(alfa_1) = 0. Thus, CL_alfa = CL(alfa_2) / alfa_2
    CL_alfas_0.append(result_0.coefs_3D.CL / np.deg2rad(5.0)) 
    CL_alfas_30.append(result_30.coefs_3D.CL / np.deg2rad(5.0)) 
    CL_alfas_45.append(result_45.coefs_3D.CL / np.deg2rad(5.0)) 

a0 = 2.0 * np.pi
M = 0.0

# Subsonic CLalfa estimation accounting for sweep, straight taper and Mach effects (Lowry; Polhamus, 1957)
CLalfa_general = lambda LamdbaC2_rad: a0 * ARs / ((a0 / np.pi) + np.sqrt((a0 / np.pi)**2 + (ARs / np.cos(LamdbaC2_rad))**2 - (ARs * M)**2))

# Plots
plt.figure()
plt.grid()
plt.xlabel("AR [-]")
plt.ylabel(r"$CL_\alpha$ [1/rad]")
plt.title(r"Effect of LE sweep ($\Lambda$) and AR in the lift curve slope")

plt.plot(ARs, CL_alfas_0, "r-", label=r"$\Lambda = 0$°")
plt.plot(ARs, CL_alfas_30, "g-", label=r"$\Lambda = 30$°")
plt.plot(ARs, CL_alfas_45, "b-", label=r"$\Lambda = 45$°")
plt.plot(ARs, CLalfa_general(np.deg2rad(0)), "r--", label=r"Polhamus eq. for $\Lambda_{C/2} = 0$°")
plt.plot(ARs, CLalfa_general(np.deg2rad(30)), "g--", label=r"Polhamus eq. for $\Lambda_{C/2} = 30$°")
plt.plot(ARs, CLalfa_general(np.deg2rad(45)), "b--", label=r"Polhamus eq. for $\Lambda_{C/2} = 45$°")
plt.plot(ARs[:4], 0.5 * np.pi * ARs[:4], "k", label="Slender wing elliptic distribution")
plt.legend()
plt.savefig("Images/CL_alfa_sweep.png")
plt.show()
# Lift coefficient slope as a function of distance to ground and dihedral angle for a rectangular wing with symmetrical airfoil

import VLM
from VLM.WingPatch import DiscretizationType

import numpy as np
import matplotlib.pyplot as plt

# Function that setup the solver for this analysis
def init_sim(AR: float, Z: float, z_off: float, ground_on: bool):
    # Rectangular wing with symmetrical airfoil
    sections = [
        VLM.Section(fy_pos=0.0, fc=1.0, x_offset=0.0, z_offset=0.0, twist_deg=0.0, airfoil_path_str="Assets/naca0012.dat"),
        VLM.Section(fy_pos=1.0, fc=1.0, x_offset=0.0, z_offset=z_off, twist_deg=0.0, airfoil_path_str="Assets/naca0012.dat")
    ]

    # Describe wing discretization
    patch = [VLM.WingPatch(16, 32, DiscretizationType.COSINE, DiscretizationType.MINUS_SINE)]

    # Make wing geometry
    wing_geom = VLM.WingGeometry(sections, patch, b=1.0, AR=AR)

    # Parameters
    params = VLM.Parameters(ground=ground_on, Z=Z, alfa_deg = 5.0, AR = wing_geom.AR, MAC = wing_geom.MAC, r_ref = np.array([0.0, 0.0, 0.0]))

    # Discretize geometry
    panels = VLM.Panels(wing_geom, params)

    # Initiate solver
    return VLM.Solver(panels, params)

CL_alfas_05, CL_alfas_1, CL_alfas_inf = [], [], []
dihedrals_05 = [-14.5, -14.0, -13.0, -12.0, 0.0, 15.0, 30.0]
dihedrals_1 = [-28.0, -27.5, -25.0, -20.0, -15.0, 0.0, 15.0, 30.0]
dihedrals_inf = [-30.0, -15.0, 0.0, 15.0, 30.0]
AR = 4.0

# 05 -> h/c = 0.5
for dihedral in dihedrals_05:
    z_off = 0.5 * np.tan(np.deg2rad(dihedral)) # z_offset = 0.5*b * tan(dihedral); b=1 is used in this example
    solver_05 = init_sim(AR, 0.5 / AR, z_off, True) # MAC = b / AR for rectangular wing. Using b = 1
    result_05, _ = solver_05.solve() # Solve VLM

    # CL_alfa = [CL(alfa_2) - CL(alfa_1)] / [alfa_2 - alfa_1]
    # Since the wing is symmetrical, for alfa_1 = 0, CL(alfa_1) = 0. Thus, CL_alfa = CL(alfa_2) / alfa_2
    CL_alfas_05.append(result_05.coefs_3D.CL / np.deg2rad(5.0)) 

# 1 -> h/c = 1.0
for dihedral in dihedrals_1:
    z_off = 0.5 * np.tan(np.deg2rad(dihedral))

    solver_1 = init_sim(AR, 1.0 / AR, z_off, True) 
    result_1, _ = solver_1.solve()
    CL_alfas_1.append(result_1.coefs_3D.CL / np.deg2rad(5.0)) 

# inf -> h/c = infinity
for dihedral in dihedrals_inf:
    z_off = 0.5 * np.tan(np.deg2rad(dihedral))

    solver_inf = init_sim(AR, 1e6, z_off, False)
    result_inf, _ = solver_inf.solve()
    CL_alfas_inf.append(result_inf.coefs_3D.CL / np.deg2rad(5.0)) 

# Plots
plt.figure()
plt.grid()
plt.xlabel("Dihedral angle [°]")
plt.ylabel(r"$CL_\alpha$ [1/rad]")
plt.title("Dihedral angle influence on the lift curve slope with ground effect")

plt.plot(dihedrals_05, CL_alfas_05, label="h/c = 0.5")
plt.plot(dihedrals_1, CL_alfas_1, label="h/c = 1.0")
plt.plot(dihedrals_inf, CL_alfas_inf, label=r"h/c = $\infty$")
plt.legend()
plt.savefig("Images/CL_alfa_dihedral.png")
plt.show()
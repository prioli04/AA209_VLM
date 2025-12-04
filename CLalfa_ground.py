# Lift coefficient slope as a function of distance to ground for a rectangular wing with symmetrical airfoil

import VLM
from VLM.WingPatch import DiscretizationType

import numpy as np
import matplotlib.pyplot as plt

# Function that setup the solver for this analysis
def init_sim(AR: float, Z: float):
    # Rectangular wing with symmetrical airfoil
    sections = [
        VLM.Section(fy_pos=0.0, fc=1.0, x_offset=0.0, z_offset=0.0, twist_deg=0.0, airfoil_path_str="Assets/naca0012.dat"),
        VLM.Section(fy_pos=1.0, fc=1.0, x_offset=0.0, z_offset=0.0, twist_deg=0.0, airfoil_path_str="Assets/naca0012.dat")
    ]

    # Describe wing discretization
    patch = [VLM.WingPatch(16, 32, DiscretizationType.COSINE, DiscretizationType.MINUS_SINE)]

    # Make wing geometry
    wing_geom = VLM.WingGeometry(sections, patch, b=1.0, AR=AR)

    # Parameters
    params = VLM.Parameters(ground=True, Z=Z, alfa_deg = 5.0, AR = wing_geom.AR, MAC = wing_geom.MAC, r_ref = np.array([0.0, 0.0, 0.0]))

    # Discretize geometry
    panels = VLM.Panels(wing_geom, params)

    # Initiate solver
    return VLM.Solver(panels, params)

h_c_vec = np.linspace(0.1, 1.5, 10) # Chord lengths above ground
CL_alfas_4, CL_alfas_6, CL_alfas_8, CL_alfas_10 = [], [], [], []

for h_c in h_c_vec:
    solver_4 = init_sim(4.0, h_c / 4.0) # MAC = b / AR for rectangular wing. Using b = 1
    solver_6 = init_sim(6.0, h_c / 6.0) 
    solver_8 = init_sim(8.0, h_c / 8.0) 
    solver_10 = init_sim(10.0, h_c / 10.0) 

    # Solve VLM
    result_4, _ = solver_4.solve()
    result_6, _ = solver_6.solve()
    result_8, _ = solver_8.solve()
    result_10, _ = solver_10.solve()

    # CL_alfa = [CL(alfa_2) - CL(alfa_1)] / [alfa_2 - alfa_1]
    # Since the wing is symmetrical, for alfa_1 = 0, CL(alfa_1) = 0. Thus, CL_alfa = CL(alfa_2) / alfa_2
    CL_alfas_4.append(result_4.coefs_3D.CL / np.deg2rad(5.0)) 
    CL_alfas_6.append(result_6.coefs_3D.CL / np.deg2rad(5.0)) 
    CL_alfas_8.append(result_8.coefs_3D.CL / np.deg2rad(5.0)) 
    CL_alfas_10.append(result_10.coefs_3D.CL / np.deg2rad(5.0)) 

# Plots
plt.figure()
plt.grid()
plt.xlabel("h/c [-]")
plt.ylabel(r"$CL_\alpha$ [1/rad]")
plt.title("Ground effect influence on the lift curve slope")

plt.plot(h_c_vec, CL_alfas_4, label="AR = 4.0")
plt.plot(h_c_vec, CL_alfas_6, label="AR = 6.0")
plt.plot(h_c_vec, CL_alfas_8, label="AR = 8.0")
plt.plot(h_c_vec, CL_alfas_10, label="AR = 10.0")
plt.legend()
plt.savefig("Images/CL_alfa_ground.png")
plt.show()
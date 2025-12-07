# Aspect ratio influence on the stall angle, using the decambering model
import VLM
from VLM.WingPatch import DiscretizationType

import numpy as np
import matplotlib.pyplot as plt

# Function that setup the solver for this analysis
def init_sim(AR: float, alfa: float):
    # Rectangular wing with NACA4415 airfoil
    sections = [
        VLM.Section(fy_pos=0.0, fc=1.0, x_offset=0.0, z_offset=0.0, twist_deg=0.0, airfoil_path_str="Assets/naca4415.dat", xfoil_path_str="Assets/NACA4415_XFOIL.out"),
        VLM.Section(fy_pos=1.0, fc=1.0, x_offset=0.0, z_offset=0.0, twist_deg=0.0, airfoil_path_str="Assets/naca4415.dat", xfoil_path_str="Assets/NACA4415_XFOIL.out")
    ]

    # Describe wing discretization
    patch = [VLM.WingPatch(16, 32, DiscretizationType.COSINE, DiscretizationType.MINUS_SINE)]

    # Make wing geometry
    wing_geom = VLM.WingGeometry(sections, patch, b=1.0, AR=AR)

    # Parameters
    params = VLM.Parameters(decambering=True, alfa_deg = alfa, AR = wing_geom.AR, MAC = wing_geom.MAC, r_ref = np.array([0.0, 0.0, 0.0]))

    # Discretize geometry
    panels = VLM.Panels(wing_geom, params)

    # Initiate solver
    return VLM.Solver(panels, params)

CL_AR6, CL_AR9, CL_AR12 = [], [], []
alfas = np.linspace(0.0, 25.0, 25)

for alfa in alfas:
    # Initiate solvers and 
    solver_AR6 = init_sim(6.0, alfa)
    solver_AR9 = init_sim(9.0, alfa)
    solver_AR12 = init_sim(12.0, alfa)

    # Solve VLM
    result_AR6, _ = solver_AR6.solve()
    result_AR9, _ = solver_AR9.solve()
    result_AR12, _ = solver_AR12.solve()

    CL_AR6.append(result_AR6.coefs_3D.CL)
    CL_AR9.append(result_AR9.coefs_3D.CL)
    CL_AR12.append(result_AR12.coefs_3D.CL)

# Compute alfa stalls
alfa_stall_AR6 = alfas[np.argmax(CL_AR6)]
alfa_stall_AR9 = alfas[np.argmax(CL_AR9)]
alfa_stall_AR12 = alfas[np.argmax(CL_AR12)]

# Plot
plt.figure()
plt.xlabel(r"$\alpha$ [°]")
plt.ylabel("CL [-]")
plt.title("Aspect ratio influence on the stall angle")
plt.grid()

# Plot CL x alfa for each wing
plt.plot(alfas, CL_AR6, "r", label=f"AR = 6.0")
plt.plot(alfas, CL_AR9, "g", label=f"AR = 9.0")
plt.plot(alfas, CL_AR12, "b", label=f"AR = 12.0")

# Plot airfoil Cl x alfa
alfa_foil = solver_AR6._wing_panels._airfoils[0]._alfa_visc
Cl_foil = solver_AR6._wing_panels._airfoils[0]._Cl_visc
alfa_stall_foil = alfa_foil[np.argmax(Cl_foil)]
plt.plot(alfa_foil, Cl_foil, "k", label="NACA4415 Airfoil")

# Plot stall angles
plt.plot([alfa_stall_foil, alfa_stall_foil], [-10, 10], "k--")
plt.plot([alfa_stall_AR6, alfa_stall_AR6], [-10, 10], "r--")
plt.plot([alfa_stall_AR9, alfa_stall_AR9], [-10, 10], "g--")
plt.plot([alfa_stall_AR12, alfa_stall_AR12], [-10, 10], "b--")
plt.xlim(left=0.0, right=25.0)
plt.ylim(bottom=0.0, top=1.8)

plt.legend()
plt.savefig("Images/CL_decamb_AR.png")
plt.show()


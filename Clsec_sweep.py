# Sectional lift distribution as a function of leading edge sweep angle for a planar wing with symmetrical airfoil

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

AR = 4.0
x_off_45 = 0.5 * np.tan(np.deg2rad(45.0)) # x_offset = 0.5*b * tan(Lambda); b=1 is used in this example

solver_0 = init_sim(0.0, AR)
solver_45 = init_sim(x_off_45, AR)

result_0, _ = solver_0.solve()
result_45, _ = solver_45.solve()

span_frac = 2.0 * result_0.y_sec # 2y/b with b = 1
Clsec_0_frac = result_0.Cl_sec / result_0.coefs_3D.CL
Clsec_45_frac = result_45.Cl_sec / result_45.coefs_3D.CL

# Plots
plt.figure()
plt.grid()
plt.xlabel("2y/b, fraction of semispan")
plt.ylabel("Cl/CL")
plt.title(r"Effect of 1/4 chord sweep ($\Lambda_{C/4}$)" + f" in the lift distribution (AR = {AR:.1f})")

plt.plot(span_frac, Clsec_0_frac, label=r"$\Lambda = 0$°")
plt.plot(span_frac, Clsec_45_frac, label=r"$\Lambda = 45$°")
plt.legend()
plt.savefig("Images/Clsec_sweep.png")
plt.show()
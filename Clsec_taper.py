# Sectional lift distribution as a function of the taper ratio for a planar wing with symmetrical airfoil

import VLM
from VLM.WingPatch import DiscretizationType

import numpy as np
import matplotlib.pyplot as plt

# Function that setup the solver for this analysis
def init_sim(taper_ratio: float, AR: float):
    # Tapered wing with symmetrical airfoil
    sections = [
        VLM.Section(fy_pos=0.0, fc=1.0, x_offset=0.0, z_offset=0.0, twist_deg=0.0, airfoil_path_str="Assets/naca0012.dat"),
        VLM.Section(fy_pos=1.0, fc=taper_ratio, x_offset=0.0, z_offset=0.0, twist_deg=0.0, airfoil_path_str="Assets/naca0012.dat")
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

AR = 7.28
solver_t1 = init_sim(1.0, AR)
solver_t07 = init_sim(0.7, AR)
solver_t04 = init_sim(0.4, AR)
solver_t01 = init_sim(0.1, AR)

result_t1, _ = solver_t1.solve()
result_t07, _ = solver_t07.solve()
result_t04, _ = solver_t04.solve()
result_t01, _ = solver_t01.solve()

span_frac = 2.0 * result_t1.y_sec # 2y/b with b = 1

Clsec_t1_frac = result_t1.Cl_sec / result_t1.coefs_3D.CL
Clsec_t07_frac = result_t07.Cl_sec / result_t07.coefs_3D.CL
Clsec_t04_frac = result_t04.Cl_sec / result_t04.coefs_3D.CL
Clsec_t01_frac = result_t01.Cl_sec / result_t01.coefs_3D.CL

# Plots
plt.figure()
plt.grid()
plt.xlabel("2y/b, fraction of semispan")
plt.ylabel("Cl/CL")
plt.title(rf"Effect of taper ratio ($\lambda$) in the lift distribution (AR = {AR:.2f})")

plt.plot(span_frac, Clsec_t1_frac, label=r"$\lambda = 1.0$")
plt.plot(span_frac, Clsec_t07_frac, label=r"$\lambda = 0.7$")
plt.plot(span_frac, Clsec_t04_frac, label=r"$\lambda = 0.4$")
plt.plot(span_frac, Clsec_t01_frac, label=r"$\lambda = 0.1$")
plt.legend()
plt.savefig("Images/Clsec_taper.png")
plt.show()
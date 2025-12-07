# Lift distribution for various alfas of a rectangular NACA0012 wing wtih the decambering model

import VLM
from VLM.WingPatch import DiscretizationType

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Rectangular wing with symmetrical airfoil
sections = [
    VLM.Section(fy_pos=0.0, fc=1.0, x_offset=0.0, z_offset=0.0, twist_deg=0.0, airfoil_path_str="Assets/naca0012.dat", xfoil_path_str="Assets/NACA0012_XFOIL.out"),
    VLM.Section(fy_pos=1.0, fc=1.0, x_offset=0.0, z_offset=0.0, twist_deg=0.0, airfoil_path_str="Assets/naca0012.dat", xfoil_path_str="Assets/NACA0012_XFOIL.out")
]

# Describe wing discretization
patch = [VLM.WingPatch(16, 32, DiscretizationType.COSINE, DiscretizationType.MINUS_SINE)]

# Make wing geometry
wing_geom = VLM.WingGeometry(sections, patch, b=1.0, AR=10.0)

alfas= [10.0, 15.0, 20.0, 25.0] # Angles of attack to run

# Create Plots
fig, axs = plt.subplots(2, 2)
axs = axs.flatten()
fig.subplots_adjust(wspace=0.4, hspace=0.7)

for (i, alfa) in enumerate(alfas):
    ax: Axes = axs[i]
    ax.set_xlabel("2y/b, fraction of semispan")
    ax.set_ylabel("Cl [-]")
    ax.grid()
    ax.set_title(rf"$\alpha = {alfa:.1f}$°")
    ax.set_ylim(bottom=0.0, top=2.5)

    # Parameters
    params = VLM.Parameters(alfa_deg = alfa, AR = wing_geom.AR, MAC = wing_geom.MAC, r_ref = np.array([0.0, 0.0, 0.0]))

    # Discretize geometry
    panels = VLM.Panels(wing_geom, params)

    # Initiate solver and solve VLM without decambering
    solver = VLM.Solver(panels, params)
    result, _ = solver.solve()

    # Initiate solver and solve VLM with decambering
    params.decambering = True
    solver_decamb = VLM.Solver(panels, params)
    result_decamb, _ = solver_decamb.solve()

    span_frac = 2.0 * result.y_sec # 2y/b with b = 1
    ax.plot(span_frac, result.Cl_sec, "k--", label="Linear VLM")
    ax.plot(span_frac, result_decamb.Cl_sec, "k", label="Decambered VLM")
    ax.legend()

fig.set_figwidth(10)
fig.set_figheight(6)
fig.savefig("Images/Clsec_naca0012_rect_decamb.png")
plt.show()


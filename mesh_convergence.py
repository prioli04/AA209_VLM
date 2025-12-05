# Estimate ideal values for panel density for convergence

import VLM
from VLM.WingPatch import DiscretizationType

import numpy as np
import matplotlib.pyplot as plt

# Function that setup the solver for this analysis
def init_sim(nx: int, ny: int, disc_type_x: DiscretizationType, disc_type_y: DiscretizationType):
    # Rectangular wing with symmetrical airfoil
    sections = [
        VLM.Section(fy_pos=0.0, fc=1.0, x_offset=0.0, z_offset=0.0, twist_deg=0.0, airfoil_path_str="Assets/naca0012.dat"),
        VLM.Section(fy_pos=1.0, fc=1.0, x_offset=0.0, z_offset=0.0, twist_deg=0.0, airfoil_path_str="Assets/naca0012.dat")
    ]

    # Describe wing discretization
    patch = [VLM.WingPatch(nx, ny, disc_type_x, disc_type_y)]

    # Make wing geometry
    wing_geom = VLM.WingGeometry(sections, patch, b=1.0, AR=8.0)

    # Parameters
    params = VLM.Parameters(alfa_deg = 5.0, AR = wing_geom.AR, MAC = wing_geom.MAC, r_ref = np.array([0.0, 0.0, 0.0]))

    # Discretize geometry
    panels = VLM.Panels(wing_geom, params)

    # Initiate solver
    return VLM.Solver(panels, params)

n_panels = []
CL_uniform, CL_cosine, CL_cosine_sine = [], [], []

for i in range(2, 23, 2):
    # Number of panels in the wing
    nx = i
    ny = nx * 2
    n_panels.append(nx * ny)

    # Initiate solvers for each geometry
    solver_uniform = init_sim(nx, ny, DiscretizationType.UNIFORM, DiscretizationType.UNIFORM)
    solver_cosine = init_sim(nx, ny, DiscretizationType.COSINE, DiscretizationType.UNIFORM)
    solver_cosine_sine = init_sim(nx, ny, DiscretizationType.COSINE, DiscretizationType.MINUS_SINE)

    # Solve VLM
    result_uniform, _ = solver_uniform.solve()
    result_cosine, _ = solver_cosine.solve()
    result_cosine_sine, _ = solver_cosine_sine.solve()

    CL_uniform.append(result_uniform.efficiency)
    CL_cosine.append(result_cosine.efficiency)
    CL_cosine_sine.append(result_cosine_sine.efficiency)

# Plots
plt.figure()
plt.grid()
plt.xlabel("Number of panels")
plt.ylabel("e [-]")
plt.title("Convergence of span efficiency")

plt.plot(n_panels, CL_uniform, label="Spacing: Uniform")
plt.plot(n_panels, CL_cosine, label="Spacing: Cosine chordwise; Uniform spanwise")
plt.plot(n_panels, CL_cosine_sine, label="Spacing: Cosine chordwise; Minus sine spanwise")
plt.legend()
plt.savefig("Images/mesh_convergence.png")
plt.show()
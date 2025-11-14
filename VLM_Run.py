# VLM - Time Stepping Wake
# No dihedral

import cProfile
import VLM
from VLM.WingPatch import DiscretizationType

sections = [
    VLM.Section(fy_pos=0.0, fc=1.0, x_offset=0.0, twist_deg=0.0, airfoil_path_str="naca0012.dat"),
    VLM.Section(fy_pos=0.5, fc=1.0 ,x_offset=0.0, twist_deg=0.0, airfoil_path_str="naca0012.dat"),
    VLM.Section(fy_pos=1.0, fc=1.0, x_offset=0.0, twist_deg=0.0, airfoil_path_str="naca0012.dat")
]

patches = [
    VLM.WingPatch(12, 24, DiscretizationType.UNIFORM, DiscretizationType.UNIFORM),
    VLM.WingPatch(12, 24, DiscretizationType.UNIFORM, DiscretizationType.UNIFORM)
]

wing_geom = VLM.WingGeometry(sections, patches, b=2.5, AR=5.0)

params = VLM.Parameters(
    V_inf = 12.0, 
    alfa_deg = 0.0, 
    beta_deg = 5.0,
    rho = 1.225, 
    AR = wing_geom.AR, 
    b = wing_geom.b, 

    wake_fixed = True,
    n_wake_deform = 5, 
    wake_dt_fact = 0.5, 
    wake_dx_fact = 0.3,

    CL_tol = 1e-4,
    CD_tol = 1e-5,

    sym=False,
    ground=False
)

Z = 0.15
panels = VLM.Panels(params, wing_geom, Z, plot=False)
solver = VLM.Solver(panels, params)
results = solver.solve()

input("Press any key to exit.")
# cProfile.run("solver.solve()")
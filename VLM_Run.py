# VLM - Time Stepping Wake
# No twist
# No dihedral

import cProfile
import VLM
from VLM.WingPatch import DiscretizationType

sections = [
    VLM.Section(fy_pos=0.0, fc=1.0, x_offset=0.0, twist_deg=0.0, airfoil_path_str="S1223.dat"),
    VLM.Section(fy_pos=0.5, fc=0.75 ,x_offset=0.03, twist_deg=-5.0, airfoil_path_str="naca0012.dat"),
    VLM.Section(fy_pos=1.0, fc=0.5, x_offset=0.1, twist_deg=-10.0, airfoil_path_str="naca0012.dat")
]

patches = [
    VLM.WingPatch(20, 5, DiscretizationType.COSINE, DiscretizationType.UNIFORM),
    VLM.WingPatch(20, 8, DiscretizationType.COSINE, DiscretizationType.MINUS_SINE)
]

wing_geom = VLM.WingGeometry(sections, patches, b=2.5, AR=5.0)

params = VLM.Parameters(
    V_inf = 12.0, 
    alfa_deg = 15.0, 
    rho = 1.225, 
    AR = wing_geom.AR, 
    b = wing_geom.b, 

    n_wake_deform = 0, 
    wake_dt_fact = 0, 
    wake_dx_fact = 0.3,

    CL_tol = 1e-0,
    CD_tol = 1e-0,

    sym=True,
    ground=False
)

Z = 0.15
panels = VLM.Panels(params, wing_geom, Z, plot=True)
solver = VLM.Solver(panels, params)
results = solver.solve()

input("Press any key to exit.")
# cProfile.run("solver.solve()")
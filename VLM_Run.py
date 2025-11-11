# VLM - Time Stepping Wake
# No twist
# No dihedral

import cProfile
import VLM

sections = [
    VLM.Section(fy_pos=0.0, fc=1.0, x_offset=0.0, twist_deg=0.0, airfoil_path_str="S1223.dat"),
    VLM.Section(fy_pos=0.5, fc=1.0 ,x_offset=0.0, twist_deg=0.0, airfoil_path_str="S1223.dat"),
    VLM.Section(fy_pos=1.0, fc=1.0, x_offset=0.0, twist_deg=0.0, airfoil_path_str="S1223.dat")
]

patches = [
    VLM.WingPatch(20, 15),
    VLM.WingPatch(20, 15)
]

wing_geom = VLM.WingGeometry(sections, patches, b=2.5, AR=3.876)

params = VLM.Parameters(
    V_inf = 12.0, 
    alfa_deg = 15.0, 
    rho = 1.225, 
    AR = wing_geom.AR, 
    b = wing_geom.b, 

    n_wake_deform = 0, 
    wake_dt_fact = 1e6, 
    wake_dx_fact = 0.3,

    CL_tol = 1e-4,
    CD_tol = 1e-5,

    sym=True,
    ground=False
)

Z = 0.15
panels = VLM.Panels(params, wing_geom, Z, plot=True)
solver = VLM.Solver(panels, params)
results = solver.solve()

input("Press any key to exit.")
# cProfile.run("solver.solve()")
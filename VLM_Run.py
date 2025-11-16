# VLM - Time Stepping Wake
# No dihedral

import cProfile
import VLM
from VLM.WingPatch import DiscretizationType

sections = [
    VLM.Section(fy_pos=0.0, fc=1.0, x_offset=0.0, twist_deg=0.0, airfoil_path_str="naca0012.dat", xfoil_path_str="NACA0012_XFOIL.OUT"),
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
    alfa_deg = 5.0, 
    beta_deg = 0.0,
    rho = 1.225, 
    AR = wing_geom.AR, 
    b = wing_geom.b, 

    wake_fixed = True,
    sym = False,
    ground = True,
    Z = 0.3
)

panels = VLM.Panels(params, wing_geom, plot=False)
solver = VLM.Solver(panels, params)
results = solver.solve()

# decamb = VLM.Decambering(sections[0], 20)
# decamb.solve(18.2)

input("Press any key to exit.")
# cProfile.run("solver.solve()")
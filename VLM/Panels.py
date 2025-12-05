import matplotlib.pyplot as plt
from .Parameters import Parameters
from .Wake import Wake
from .WingGeometry import WingGeometry
from .WingPanels import WingPanels

# Object for holding all meshes (wing + wake) information
class Panels:
    def __init__(self, wing_geometry: WingGeometry, params: Parameters, plot=False):
        self._wake_panels = None
        
        self._wing_geom = wing_geometry
        self._wing_panels = WingPanels(wing_geometry, params.Z, params.wake_dx, params.sym) # Create the wing mesh based on the geometry and solver parameters
        TE_points = self._wing_panels.extract_TE_points() # Get the wing trailing edge points

        self._plot_ax = self._create_plot() if plot else None

        # Create a wake mesh if the time stepping wake is active (fixed wake does not need a mesh in this implementation)
        if not params.wake_fixed:
            self._wake_panels = Wake(params.n_wake_deform, self._wing_panels._ny, params.wake_dt, TE_points, params.wake_dx, params.sym, params.ground, self._plot_ax)

    # Create the mesh plot
    def _create_plot(self):
        _, ax = plt.subplots(subplot_kw={"projection": "3d", "computed_zorder": False})
        self._wing_panels.plot_mesh(ax)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        ax.set_axis_off()
        plt.show(block=False)

        return ax

    # Print wing geometry data
    def print_wing_geom(self):
        self._wing_geom.print_wing_geom()

    # Getter for the wing mesh
    def get_wing_panels(self):
        return self._wing_panels
    
    # Getter for the wake mesh
    def get_wake_panels(self):
        return self._wake_panels


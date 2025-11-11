import matplotlib.pyplot as plt
from .Parameters import Parameters
from .Wake import Wake
from .WingGeometry import WingGeometry
from .WingPanels import WingPanels

class Panels:
    def __init__(self, params: Parameters, wing_geometry: WingGeometry, Z: float, plot=False):
        self._wing_geom = wing_geometry
        self._wing_panels = WingPanels(wing_geometry, Z, wake_dx=params.wake_dx)
        TE_points = self._wing_panels.extract_TE_points()

        self._plot_ax = self._create_plot() if plot else None
        self._wake_panels = Wake(params.n_wake_deform, self._wing_panels._ny, params.wake_dt, TE_points, params.wake_dx, params.ground, self._plot_ax)

    def _create_plot(self):
        _, ax = plt.subplots(subplot_kw={"projection": "3d", "computed_zorder": False})
        self._wing_panels.plot_mesh(ax)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        plt.show(block=False)

        return ax

    def print_wing_geom(self):
        self._wing_geom.print_wing_geom()

    def get_wing_panels(self):
        return self._wing_panels
    
    def get_wake_panels(self):
        return self._wake_panels


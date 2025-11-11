import matplotlib.pyplot as plt
from .Parameters import Parameters
from .Section import Section
from .Wake import Wake
from .Wing import Wing
from typing import List

class Panels:
    def __init__(self, params: Parameters, sections: List[Section], nx: int, ny: int, Z: float, plot=False):
        self._wing_panels = Wing(params.b, params.S, nx, ny, Z, sections, wake_dx=params.wake_dx)
        TE_points = self._wing_panels.extract_TE_points()

        self._plot_ax = self._create_plot() if plot else None
        self._wake_panels = Wake(params.n_wake_deform, ny, params.wake_dt, TE_points, params.wake_dx, params.ground, self._plot_ax)

    def _create_plot(self):
        _, ax = plt.subplots(subplot_kw={"projection": "3d", "computed_zorder": False})
        self._wing_panels.plot_mesh(ax)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        plt.show(block=False)

        return ax

    def get_wing_panels(self):
        return self._wing_panels
    
    def get_wake_panels(self):
        return self._wake_panels


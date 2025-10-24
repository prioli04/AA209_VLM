import matplotlib.pyplot as plt
from .Wake import Wake
from .Wing import Wing

class Panels:
    def __init__(
            self,
            b: float,
            AR: float, 
            nx: int, 
            ny: int,
            wake_dx: float,
            nt: int):
        
        self._wing_panels = Wing(b, AR, nx, ny, wake_dx=wake_dx)
        self._wake_panels = Wake(nt, ny)

    def get_wing_panels(self):
        return self._wing_panels
    
    def get_wake_panels(self):
        return self._wake_panels

    def plot_model(self):
        _, ax = plt.subplots(subplot_kw={"projection": "3d", "computed_zorder": False})
        self._wing_panels.plot_mesh(ax)
        self._wake_panels.plot_mesh(ax)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        plt.show()
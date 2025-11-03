from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np

class Airfoil:
    def __init__(self, name: str, x: List[float], y: List[float]):
        self._name = name
        self._x_upper, self._y_upper, self._x_lower, self._y_lower = self._split_upper_lower(x, y)
        self._x_camber, self._y_camber = self._compute_camber_line()
        # self.plot_foil()

    def _split_upper_lower(self,  x: List[float], y: List[float]):
        eps = 1e-6

        x_upper: List[float] = []
        y_upper: List[float] = []
        x_lower: List[float] = []
        y_lower: List[float] = []

        upper = True

        for i in range(len(x)):
            if y[i] < 0.0 and upper:
                upper = False
                x_lower.append(0.0)
                y_lower.append(0.0)

                if x[i - 1] > eps:
                    x_upper.append(0.0)
                    y_upper.append(0.0)  

            if upper:
                x_upper.append(x[i])
                y_upper.append(y[i])

            else:
                x_lower.append(x[i])
                y_lower.append(y[i])

        x_upper.reverse()
        y_upper.reverse()
        return x_upper, y_upper, x_lower, y_lower

    def _compute_camber_line(self):
        x_camber = self._x_upper
        y_camber = np.zeros_like(x_camber)

        for (i, x) in enumerate(x_camber):
            y_upper = self._y_upper[i]

            idx_lower = np.searchsorted(self._x_lower, x, side="right")
            x_interp_lower = self._x_lower[idx_lower - 1:idx_lower + 1]
            y_interp_lower = self._y_lower[idx_lower - 1:idx_lower + 1]
            y_lower = np.interp(x, x_interp_lower, y_interp_lower)

            y_camber[i] = (y_upper + y_lower) / 2.0

        return x_camber, y_camber
    
    def get_camber_line(self, nx: int, chord: float):
        x_vals = np.linspace(0.0, 1.0, nx)
        camber_z = np.zeros_like(x_vals)

        for (i, x) in enumerate(x_vals):
            idx = np.searchsorted(self._x_camber, x, side="right")
            x_interp = self._x_camber[idx - 1:idx + 1]
            y_interp = self._y_camber[idx - 1:idx + 1]
            camber_z[i] = np.interp(x, x_interp, y_interp)

        return camber_z * chord

    def plot_foil(self):
        fig = plt.figure()
        ax = fig.gca()
        plt.plot(self._x_upper, self._y_upper, "k")
        plt.plot(self._x_lower, self._y_lower, "k")
        plt.plot(self._x_camber, self._y_camber, "r--")

        plt.title(f"{self._name}")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        ax.set_aspect("equal")
        plt.show(block=False)

    @staticmethod
    def read(airfoil_path: Path):
        with airfoil_path.open() as f:
            lines = f.read().splitlines()

        name = ""
        x: List[float] = []
        y: List[float] = []

        for (i, line) in enumerate(lines):
            if i == 0:
                name = line

            elif not line.startswith("#"):
                coords = line.split()

                if len(coords) != 2:
                    raise ValueError("Wrong airfoil file format (incorrect number of coordinates found).")

                try:
                    x_num = float(coords[0])
                    y_num = float(coords[1])

                except ValueError:
                    print("Wrong airfoil file format (coordinates could not be converted to float).")

                x.append(x_num)
                y.append(y_num)

        return Airfoil(name, x, y)
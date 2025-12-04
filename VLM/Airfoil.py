from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np

class Airfoil:
    def __init__(self, name: str, x: List[float], y: List[float], xfoil_path: Path | None):
        self._name = name
        self._x_upper, self._y_upper, self._x_lower, self._y_lower = self._split_upper_lower(np.array(x), np.array(y))
        self._x_camber, self._y_camber = self._compute_camber_line()
        self._alfa_visc, self._Cl_visc, self._Cm_visc, self._alfa0 = np.empty(0), np.empty(0), np.empty(0), 0.0
        self._viscous_data = False

        if xfoil_path is not None:
            self._alfa_visc, self._Cl_visc, self._Cm_visc, self._alfa0 = self._read_xfoil(xfoil_path)
            self._viscous_data = True

    def _split_upper_lower(self,  x: np.ndarray, y: np.ndarray):
        eps = 1e-6

        x_min = np.min(x)
        x_max = np.max(x)

        x -= x_min
        chord = x_max - x_min

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
                x_upper.append(x[i] / chord)
                y_upper.append(y[i] / chord)

            else:
                x_lower.append(x[i] / chord)
                y_lower.append(y[i] / chord)



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
    
    def _read_xfoil(self, xfoil_path: Path):
        with xfoil_path.open() as f:
            lines = f.read().splitlines()

        found_params, found_results = False, False
        alfa_id, cl_id, cm_id = 0, 0, 0
        alfas, Cls, Cms = np.empty(0), np.empty(0), np.empty(0)

        for line in lines:
            line = line.strip()
            tokens = line.split()

            if found_params and found_results and "--" not in line:
                try:
                    alfas = np.hstack([alfas, float(tokens[alfa_id])]) 
                    Cls = np.hstack([Cls, float(tokens[cl_id])]) 
                    Cms = np.hstack([Cms, float(tokens[cm_id])]) 

                except ValueError:
                    raise ValueError("Could not parse result values. Check file provided!")

            if line.startswith("Mach"):
                try:
                    re_id = tokens.index("Re")
                    n_crit_id = tokens.index("Ncrit")
                    re = float("".join(tokens[re_id + 2:n_crit_id]))        

                except ValueError:
                    raise ValueError("Could not parse the Reynolds number of the run. Check file provided!")

                found_params = True

            if line.startswith("alpha"):
                try:
                    alfa_id = tokens.index("alpha")
                    cl_id = tokens.index("CL")
                    cm_id = tokens.index("CM")

                except ValueError:
                    raise ValueError("Could not find alfa or Cl columns. Check file provided!")
                
                found_results = True

        if not found_params:
            raise ValueError("Could not find run parameters information. Check file provided!")
        
        if not found_results:
            raise ValueError("Could not find results. Check file provided")

        alfa0 = self._compute_alfa0(alfas, Cls)
        return alfas, Cls, Cms, alfa0

    def _compute_alfa0(self, alfa_visc: np.ndarray, Cl_visc: np.ndarray):
        closest_id = np.argmin(np.abs(Cl_visc))
        return np.deg2rad(alfa_visc[closest_id]) - Cl_visc[closest_id] / (2.0 * np.pi)

    def get_camber_line(self, x_vals: np.ndarray, chord: float, twist_deg: float):
        rot_angle = -np.deg2rad(twist_deg)

        camber_x = np.zeros_like(x_vals)
        camber_y = np.zeros_like(x_vals)

        for (i, x) in enumerate(x_vals):
            idx = np.searchsorted(self._x_camber, x, side="right")
            x_interp = self._x_camber[idx - 1:idx + 1]
            y_interp = self._y_camber[idx - 1:idx + 1]

            Px = x_vals[i] - 0.25
            Py = np.interp(x, x_interp, y_interp)

            Px_rot = Px * np.cos(rot_angle) - Py * np.sin(rot_angle)
            Py_rot = Px * np.sin(rot_angle) + Py * np.cos(rot_angle)

            camber_x[i] = Px_rot + 0.25
            camber_y[i] = Py_rot

        return camber_x * chord, camber_y * chord
    
    def get_visc_coefs(self):
        return self._alfa_visc, self._Cl_visc, self._Cm_visc, self._alfa0
        
    def has_viscous_data(self):
        return self._viscous_data

    @staticmethod
    def read(airfoil_path: Path, xfoil_path: Path | None = None):
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

        return Airfoil(name, x, y, xfoil_path)
from pathlib import Path
from typing import List
import numpy as np

# Object for reading and processing airfoil data
class Airfoil:
    def __init__(self, name: str, x: List[float], y: List[float], xfoil_path: Path | None):
        self._name = name # Airfoil name
        self._x_upper, self._y_upper, self._x_lower, self._y_lower = self._split_upper_lower(np.array(x), np.array(y)) # Split points into upper and lower surface
        self._x_camber, self._y_camber = self._compute_camber_line() # Compute the camber line

        # Viscous data initialization (for decambering routine)
        self._alfa_visc, self._Cl_visc, self._Cm_visc, self._alfa0 = np.empty(0), np.empty(0), np.empty(0), 0.0
        self._viscous_data = False

        # Read xfoil data if the file is specified
        if xfoil_path is not None:
            self._alfa_visc, self._Cl_visc, self._Cm_visc, self._alfa0 = self._read_xfoil(xfoil_path)
            self._viscous_data = True

    # Function that splits a list of (x, y) coordinates into upper and lower surface
    # Assumes Selig's format
    def _split_upper_lower(self,  x: np.ndarray, y: np.ndarray):
        eps = 1e-6 # Equality threshold

        x_min = np.min(x)
        x_max = np.max(x)

        x -= x_min # Translate airfoil such that the leading edge is at x = 0
        chord = x_max - x_min # Airfoil chord

        x_upper: List[float] = []
        y_upper: List[float] = []
        x_lower: List[float] = []
        y_lower: List[float] = []

        upper = True

        # Selig's format: 
        # - Points start from trailing edge at (1, 0)
        # - Traverse the upper surface towards the leading edge at (0, 0)
        # - Change to lower surface when y becomes negative
        # - Comes back towards the trailing edge

        for i in range(len(x)):
            # Condition for stop appending to upper surface and start appending to lower surface
            if y[i] < 0.0 and upper:
                upper = False
                x_lower.append(0.0)
                y_lower.append(0.0)

                # If point (0, 0) is not part of the dataset include it
                if x[i - 1] > eps:
                    x_upper.append(0.0)
                    y_upper.append(0.0)  

            # Append to upper surface (normalizing by the chord)
            if upper:
                x_upper.append(x[i] / chord)
                y_upper.append(y[i] / chord)

            # Append to lower surface (normalizing by the chord)
            else:
                x_lower.append(x[i] / chord)
                y_lower.append(y[i] / chord)

        # Reverse upper surface so it goes from leading edge to trailing edge
        x_upper.reverse()
        y_upper.reverse()
        return x_upper, y_upper, x_lower, y_lower

    # Function that computes the airfoil's camber line
    def _compute_camber_line(self):
        x_camber = self._x_upper # It uses the upper surface's x coordinates as default (might differ from the points on the lower surface)
        y_camber = np.zeros_like(x_camber)

        for (i, x) in enumerate(x_camber):
            y_upper = self._y_upper[i] # Since the x coordinates from upper surface are used, y coordinates are trivial in this case
            y_lower = np.interp(x, self._x_lower, self._y_lower) # y coordinates of the lower surface are determined by linear interpolation 
            y_camber[i] = (y_upper + y_lower) / 2.0 # y coordinates of the camber line defined as the average between upper and lower

        return x_camber, y_camber
    
    # Function that reads an xfoil result file
    def _read_xfoil(self, xfoil_path: Path):
        with xfoil_path.open() as f:
            lines = f.read().splitlines()

        found_params, found_results = False, False
        alfa_id, cl_id, cm_id = 0, 0, 0
        alfas, Cls, Cms = np.empty(0), np.empty(0), np.empty(0)

        for line in lines:
            line = line.strip() # Remove all leading and trailing whitespaces and linebreaks
            tokens = line.split() # Split line on whitespace

            # Only try to parse values if the previous info (params and results) were found
            # The "----" line before the actual values is also being discarded here
            if found_params and found_results and "--" not in line:
                try:
                    alfas = np.hstack([alfas, float(tokens[alfa_id])]) 
                    Cls = np.hstack([Cls, float(tokens[cl_id])]) 
                    Cms = np.hstack([Cms, float(tokens[cm_id])]) 

                except ValueError:
                    raise ValueError("Could not parse result values. Check file provided!")

            # Line containing run parameters starts with "Mach"
            if line.startswith("Mach"):
                try:
                    # Reynolds need to be parsed in a special way since it's written like "1.000 e 6"
                    re_id = tokens.index("Re")
                    n_crit_id = tokens.index("Ncrit")
                    re = float("".join(tokens[re_id + 2:n_crit_id])) 

                except ValueError:
                    raise ValueError("Could not parse the Reynolds number of the run. Check file provided!")

                found_params = True

            # The line starting with "alpha" contains the headers for the results
            if line.startswith("alpha"):
                try:
                    # Find the column id of the variables needed
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

        alfa0 = self._compute_alfa0(alfas, Cls) # Compute the zero lift angle of attack
        return alfas, Cls, Cms, alfa0

    # Function for computing the zero lift angle of attack
    def _compute_alfa0(self, alfa_visc: np.ndarray, Cl_visc: np.ndarray):
        # Find the closest data point to Cl=0 and extrapolate using thin airfoil formula
        closest_id = np.argmin(np.abs(Cl_visc))  
        return np.deg2rad(alfa_visc[closest_id]) - Cl_visc[closest_id] / (2.0 * np.pi) # From Cl = 2pi * (alfa - alfa0)

    # Function that return the transformed camber line for a given wing section
    def get_camber_line(self, x_vals: np.ndarray, chord: float, twist_deg: float):
        rot_angle = -np.deg2rad(twist_deg) # Rotation matrix uses the opposite convention for the angle

        camber_x = np.zeros_like(x_vals)
        camber_y = np.zeros_like(x_vals)

        for (i, x) in enumerate(x_vals):
            Px = x - 0.25 # Translate 1/4 chord to the origin
            Py = np.interp(x, self._x_camber, self._y_camber) # Interpolate y coordinates based on the points on the wing mesh

            # Apply twist
            Px_rot = Px * np.cos(rot_angle) - Py * np.sin(rot_angle)
            Py_rot = Px * np.sin(rot_angle) + Py * np.cos(rot_angle)

            camber_x[i] = Px_rot + 0.25 # Return 1/4 chord to the original position
            camber_y[i] = Py_rot

        return camber_x * chord, camber_y * chord # return coordinates scaled by the local chord
    
    # Getter for the viscous data
    def get_visc_coefs(self):
        return self._alfa_visc, self._Cl_visc, self._Cm_visc, self._alfa0
        
    # Flag for indicating if viscous data was provided
    def has_viscous_data(self):
        return self._viscous_data

    # Function for reading the airfoil coordinates file
    @staticmethod
    def read(airfoil_path: Path, xfoil_path: Path | None = None):
        with airfoil_path.open() as f:
            lines = f.read().splitlines()

        name = ""
        x: List[float] = []
        y: List[float] = []

        for (i, line) in enumerate(lines):
            # Name is defined as whatever is written in the first line of the file
            if i == 0:
                name = line

            # Skip lines starting with '#' as those are defined as comments by the Selig format
            elif not line.startswith("#"):
                coords = line.split()

                # Coords must have 2 values only [x, y]. If not, something else is written within the file 
                if len(coords) != 2:
                    raise ValueError("Wrong airfoil file format (incorrect number of coordinates found).")

                try:
                    x_num = float(coords[0])
                    y_num = float(coords[1])

                except ValueError:
                    print("Wrong airfoil file format (coordinates could not be converted to float).")

                x.append(x_num)
                y.append(y_num)

        return Airfoil(name, x, y, xfoil_path) # Create the Airfoil object and return
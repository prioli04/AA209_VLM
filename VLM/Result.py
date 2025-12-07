from dataclasses import dataclass

import numpy as np
import sys

# Object for holding the results of post-processing the VLM output
@dataclass
class Result:
    # Object for packing 3D coefficients
    @dataclass(frozen=True)
    class Coefs_3D:
        CL: float = 0.0
        CD: float = 0.0
        CY: float = 0.0
        CMl: float = 0.0
        CM: float = 0.0
        CN: float = 0.0

        def to_array(self):
            return np.array([self.CL, self.CD, self.CY, self.CMl, self.CM, self.CN])

    def __init__(self):
        self.y_sec = np.empty(0) # y coordinates of each section
        self.Cl_sec = np.empty(0) # Sectional lift coefficient distribution
        self.CDp = 0.0 # Parasite drag coefficient
        self.coefs_3D_prev = Result.Coefs_3D() # Previous iteration 3D coefficients
        self.coefs_3D = Result.Coefs_3D() # Current iteration 3D coefficients
        self.residuals = Result.Coefs_3D() # 3D coefficients residuals
        self.CL_CD = 0.0 # Lift-to-drag ratio [-]
        self.efficiency = 0.0 # Span efficiency [-] e = CL^2/(pi * AR * CD)

    # Residuals defined as coefs_new - coefs_old
    def _compute_residuals(self):
        CL_res = self.coefs_3D.CL - self.coefs_3D_prev.CL
        CD_res = self.coefs_3D.CD - self.coefs_3D_prev.CD
        CY_res = self.coefs_3D.CY - self.coefs_3D_prev.CY

        CMl_res = self.coefs_3D.CMl - self.coefs_3D_prev.CMl
        CM_res = self.coefs_3D.CM - self.coefs_3D_prev.CM
        CN_res = self.coefs_3D.CN - self.coefs_3D_prev.CN

        self.residuals = Result.Coefs_3D(CL_res, CD_res, CY_res, CMl_res, CM_res, CN_res)
    
    # Update results with current iteration coefficients
    def update(self, y_sec: np.ndarray, Cl_sec: np.ndarray, CDp: float, coefs_3D: Coefs_3D, AR: float):
        self.y_sec = y_sec
        self.Cl_sec = Cl_sec
        self.CDp = CDp
        self.CL_CD = coefs_3D.CL / coefs_3D.CD
        self.efficiency = coefs_3D.CL**2 / (coefs_3D.CD * np.pi * AR)

        self.coefs_3D_prev = self.coefs_3D
        self.coefs_3D = coefs_3D
        self._compute_residuals()

    # Print results to the screen
    def print(self, iteration: int, wake_fixed: bool):
        fields = {
            "CL": self.coefs_3D.CL, "CD": self.coefs_3D.CD, "CM": self.coefs_3D.CM,
            "CY": self.coefs_3D.CY, "CMl": self.coefs_3D.CMl, "CN": self.coefs_3D.CN,
            "CL/CD": self.CL_CD, "Efficiency": self.efficiency, "CDp": self.CDp
        }

        N = len(fields) + 1

        # '\x1b' -> St}art escape sequence
        # '[' -> Control sequence introducer
        # '2K' ->  Clear entire line
        ERASE_LINE = "\x1b[2K"

        # '\x1b' -> Start escape sequence
        # '[' -> Control sequence introducer
        # 'NA' ->  Move cursor up N lines
        CURSOR_UP_N = f"\x1b[{N}A"
        sys.stdout.write(CURSOR_UP_N) if iteration != 0 else print()

        # Write the iteration and residuals line
        if not wake_fixed:
            CL_res = self.residuals.CL
            CD_res = self.residuals.CD

            sys.stdout.write(ERASE_LINE)
            sys.stdout.write(f"Iteration {iteration}: CL_res = {CL_res:.5e} \t CD_res = {CD_res:.5e} \n")

        # Write the results
        for (k, v) in fields.items():
            sys.stdout.write(ERASE_LINE)
            sys.stdout.write(f"{k}: {v:.5f}\n")

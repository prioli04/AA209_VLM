from dataclasses import dataclass

import numpy as np
import sys

@dataclass
class Result:
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
        self.y_sec = np.empty(0)
        self.Cl_sec = np.empty(0)
        self.Cd_sec = np.empty(0)
        self.coefs_3D_prev = Result.Coefs_3D()
        self.coefs_3D = Result.Coefs_3D()
        self.residuals = Result.Coefs_3D()
        self.CL_CD = 0.0
        self.efficiency = 0.0

    def _compute_residuals(self):
        CL_res = self.coefs_3D.CL - self.coefs_3D_prev.CL
        CD_res = self.coefs_3D.CD - self.coefs_3D_prev.CD
        CY_res = self.coefs_3D.CY - self.coefs_3D_prev.CY

        CMl_res = self.coefs_3D.CMl - self.coefs_3D_prev.CMl
        CM_res = self.coefs_3D.CM - self.coefs_3D_prev.CM
        CN_res = self.coefs_3D.CN - self.coefs_3D_prev.CN

        self.residuals = Result.Coefs_3D(CL_res, CD_res, CY_res, CMl_res, CM_res, CN_res)
    
    def update(self, y_sec: np.ndarray, Cl_sec: np.ndarray, Cd_sec: np.ndarray, coefs_3D: Coefs_3D, AR: float):
        self.y_sec = y_sec
        self.Cl_sec, self.Cd_sec = Cl_sec, Cd_sec
        self.CL_CD = coefs_3D.CL / coefs_3D.CD
        self.efficiency = coefs_3D.CL**2 / (coefs_3D.CD * np.pi * AR)

        self.coefs_3D_prev = self.coefs_3D
        self.coefs_3D = coefs_3D
        self._compute_residuals()

    def print(self, iteration: int, wake_fixed: bool):
        fields = {
            "CL": self.coefs_3D.CL, "CD": self.coefs_3D.CD, "CM": self.coefs_3D.CM,
            "CY": self.coefs_3D.CY, "CMl": self.coefs_3D.CMl, "CN": self.coefs_3D.CN,
            "CL/CD": self.CL_CD, "Efficiency": self.efficiency
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

        if not wake_fixed:
            CL_res = self.residuals.CL
            CD_res = self.residuals.CD

            sys.stdout.write(ERASE_LINE)
            sys.stdout.write(f"Iteration {iteration}: CL_res = {CL_res:.5e} \t CD_res = {CD_res:.5e} \n")

        for (k, v) in fields.items():
            sys.stdout.write(ERASE_LINE)
            sys.stdout.write(f"{k}: {v:.5f}\n")

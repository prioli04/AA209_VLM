from dataclasses import dataclass, field
import numpy as np

@dataclass
class Parameters:
    alfa_deg: float
    AR: float
    MAC: float
    r_ref: np.ndarray

    V_inf: float = 1.0
    beta_deg: float = 0.0
    rho: float = 1.0
    b: float = 1.0
    S: float = field(init=False)

    wake_fixed: bool = True

    sym: bool = True
    ground: bool = False
    decambering: bool = False

    # Optional parameters
    Z: float = 0.0

    n_wake_deform: int = 10
    wake_dt_fact: float = 0.5
    wake_dx_fact: float = 0.3
    wake_dt: float = field(init=False)
    wake_dx: float = field(init=False)

    CL_tol: float = 1e-4
    CD_tol: float = 1e-5
    CY_tol: float = 1e-5

    CMl_tol: float = 1e-5
    CM_tol: float = 1e-5
    CN_tol: float = 1e-5

    decamb_Cl_tol: float = 1e-2
    decamb_x2: float = 0.8
    decamb_theta2: float = field(init=False)
    decamb_max_iter: int = 200
    decamb_under_relaxation: float = 0.0
    decamb_smoothing: float = 0.0
    
    def __post_init__(self):
        self.S = self.b**2 / self.AR
        self.decamb_theta2 = np.acos(1.0 - 2.0 * self.decamb_x2)
        self.r_ref = self.r_ref + np.array([0.0, 0.0, self.Z])

        MGC = self.b / self.AR
        self.wake_dt = self.wake_dt_fact * MGC / self.V_inf
        self.wake_dx = self.wake_dx_fact * self.wake_dt * self.V_inf

    def print_run_params(self):
        print("===== Run Parameters =====")
        print(f"Symmetry: {self.sym}")
        print(f"Ground: {self.ground} (Height: {self.Z:.3f} m)")
        print(f"Reference Point: ({self.r_ref[0]:.3f}, {self.r_ref[1]:.3f}, {self.r_ref[2]:.3f}) m")
        print(f"V_inf: {self.V_inf:.2f} m/s (α: {self.alfa_deg:.2f}°; β: {self.beta_deg:.2f}°)")
        print(f"rho: {self.rho:.3f} kg/m³")

        if not self.wake_fixed:
            print(f"CL tolerance: {self.CL_tol:.1e}")
            print(f"CD tolerance: {self.CD_tol:.1e}")
            print(f"CM tolerance: {self.CM_tol:.1e}")

            if not self.sym:
                print(f"CY tolerance: {self.CY_tol:.1e}")
                print(f"CMl tolerance: {self.CMl_tol:.1e}")
                print(f"CN tolerance: {self.CN_tol:.1e}")

        print()

    def V_inf_vec(self):
        c_alfa, s_alfa = np.cos(np.deg2rad(self.alfa_deg)), np.sin(np.deg2rad(self.alfa_deg))
        c_beta, s_beta = np.cos(np.deg2rad(self.beta_deg)), np.sin(np.deg2rad(self.beta_deg))
        return self.V_inf * np.array([c_alfa * c_beta, -s_beta, s_alfa * c_beta])
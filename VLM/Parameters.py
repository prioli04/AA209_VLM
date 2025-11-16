from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class Parameters:
    V_inf: float
    alfa_deg: float
    beta_deg: float
    rho: float
    AR: float
    b: float
    S: float = field(init=False)

    wake_fixed: bool

    sym: bool
    ground: bool

    # Optional parameters
    Z: float = 0.0

    n_wake_deform: int = 10
    wake_dt_fact: float = 0.5
    wake_dx_fact: float = 0.3

    wake_dt: float = field(init=False)
    wake_dx: float = field(init=False)

    CL_tol: float = 1e-4
    CD_tol: float = 1e-5

    Decamb_Cl_tol: float = 1e-4
    Decamb_Cm_tol: float = 1e-5
    
    def __post_init__(self):
        super().__setattr__("S", self.b**2 / self.AR)

        MGC = self.b / self.AR
        super().__setattr__("wake_dt", self.wake_dt_fact * MGC / self.V_inf)
        super().__setattr__("wake_dx", self.wake_dx_fact * self.wake_dt * self.V_inf)

    def print_run_params(self):
        print("===== Run Parameters =====")
        print(f"Symmetry: {self.sym}")
        print(f"Ground: {self.ground} (Height: {self.Z:.3f} m)")
        print(f"V_inf: {self.V_inf:.2f} m/s (α: {self.alfa_deg:.2f}°; β: {self.beta_deg:.2f}°)")
        print(f"rho: {self.rho:.3f} kg/m³")

        if not self.wake_fixed:
            print(f"CL tolerance: {self.CL_tol:.1e}")
            print(f"CD tolerance: {self.CD_tol:.1e}")

        print()

    def V_inf_vec(self):
        c_alfa, s_alfa = np.cos(np.deg2rad(self.alfa_deg)), np.sin(np.deg2rad(self.alfa_deg))
        c_beta, s_beta = np.cos(np.deg2rad(self.beta_deg)), np.sin(np.deg2rad(self.beta_deg))
        return self.V_inf * np.array([c_alfa * c_beta, s_beta, s_alfa * c_beta])
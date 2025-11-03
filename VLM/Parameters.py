from dataclasses import dataclass, field

@dataclass(frozen=True)
class Parameters:
    V_inf: float
    alfa_deg: float
    rho: float
    AR: float
    b: float
    S: float = field(init=False)

    n_wake_deform: int
    wake_steps: int
    wake_dt_fact: float
    wake_dx_fact: float

    wake_dt: float = field(init=False)
    wake_dx: float = field(init=False)

    CL_tol: float
    CD_tol: float

    def __post_init__(self):
        super().__setattr__("S", self.b**2 / self.AR)

        MGC = self.b / self.AR
        super().__setattr__("wake_dt", self.wake_dt_fact * MGC / self.V_inf)
        super().__setattr__("wake_dx", self.wake_dx_fact * self.wake_dt * self.V_inf)
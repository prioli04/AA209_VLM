from typing import NamedTuple

class Parameters(NamedTuple):
    V_inf: float
    alfa_deg: float
    rho: float
    S: float
    AR: float
    b: float

    n_wake_deform: int
    wake_steps: int
    wake_dt: float
    wake_dx: float

    CL_tol: float
    CD_tol: float
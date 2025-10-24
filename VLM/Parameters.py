from typing import NamedTuple

class Parameters(NamedTuple):
    V_inf: float
    alfa_deg: float
    rho: float
    S: float
    AR: float

    wake_steps: int
    wake_dt: float
    wake_dx: float
from dataclasses import dataclass, field
import numpy as np

# Main reference: Drela, M.; Flight Vehicle Aerodynamics

# Object for holding parameters used by the solver and for mesh generation also 
@dataclass
class Parameters:
    alfa_deg: float # Angle of attack [°]
    AR: float # Aspect Ratio, AR=b^2/S [-]
    MAC: float # Mean Aerodynamic Chord [m]
    r_ref: np.ndarray # Reference point for moment calculation (x_ref, y_ref, z_ref) [m]

    # Optional parameters
    V_inf: float = 1.0 # Freestream velocity magnitude [m/s]
    beta_deg: float = 0.0 # Slip angle [°]
    rho: float = 1.0 # Freestream density [kg/m^3]
    b: float = 1.0 # Reference span [m]
    S: float = field(init=False) # Reference area [m^2]
    Z: float = 0.0 # Wing height [m]

    # Boolean flags
    wake_fixed: bool = True # Switch between fixed and time stepping wake models
    sym: bool = True # Symmetry flag
    ground: bool = False # Ground flag
    decambering: bool = False # Switch for the decambering model

    # Time stepping wake model parameters
    n_wake_deform: int = 10 # Number of chordwise panel rows that get deformed at each iteration
    wake_dt_fact: float = 0.5 # Sets how many chords the wing travels after one time step. 
    wake_dx_fact: float = 0.3 # Sets the gap between trailing edge and 1st wake panel, in fraction of distance covered in 1 time step.  
    wake_dt: float = field(init=False) # wake_dt = wake_dt_fact * chord / V_inf 
    wake_dx: float = field(init=False) # wake_dx = wake_dx_fact * wake_dt * V_inf
 
    # Time stepping wake coefficient tolerances (stopping criteria)
    CL_tol: float = 1e-4
    CD_tol: float = 1e-5
    CY_tol: float = 1e-5
    CMl_tol: float = 1e-5
    CM_tol: float = 1e-5
    CN_tol: float = 1e-5

    # Decambering parameters
    decamb_Cl_tol: float = 1e-2 # Residual tolerance
    decamb_max_iter: int = 200 # Max iterations
    decamb_under_relaxation: float = 0.0 # Under relaxation factor
    decamb_smoothing: float = 0.0 # Smoothing factor
    
    # Compute values that depend on other parameters provided
    def __post_init__(self):
        self.S = self.b**2 / self.AR
        self.r_ref = self.r_ref + np.array([0.0, 0.0, self.Z])

        MGC = self.b / self.AR
        self.wake_dt = self.wake_dt_fact * MGC / self.V_inf
        self.wake_dx = self.wake_dx_fact * self.wake_dt * self.V_inf

    # Print parameters for the simulation
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

    # Get rotated freestream velocity vector. Equation 6.38 of (Drela)
    def V_inf_vec(self):
        c_alfa, s_alfa = np.cos(np.deg2rad(self.alfa_deg)), np.sin(np.deg2rad(self.alfa_deg))
        c_beta, s_beta = np.cos(np.deg2rad(self.beta_deg)), np.sin(np.deg2rad(self.beta_deg))
        return self.V_inf * np.array([c_alfa * c_beta, -s_beta, s_alfa * c_beta])
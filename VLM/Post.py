from .Parameters import Parameters
from .Result import Result
from .WingPanels import WingPanels

import matplotlib.pyplot as plt
import numpy as np

# References: 
# - Drela, M.; Flight Vehicle Aerodynamics
# - Katz, J.; Plotkin, A.; Low-Speed Aerodynamics

# Object for performing the post processing operations
class Post:
    def __init__(self, wing_mesh: WingPanels, B_trefftz: np.ndarray, V_hat_bound: np.ndarray, params: Parameters):
        self._converged = False # Flag to indicate the solution converged
        self._iter = 0 # Iteration counter
        self._sym = params.sym # Symmetry flag
        self._wake_fixed = params.wake_fixed # Wake fixed flag

        self._B_trefftz = B_trefftz # Influence matrix used for computing the downwash it the Trefftz plane
        self._V_hat_bound = V_hat_bound # Influence matrix used for computing local velocities at the bound vortices 

        # Saving some parameters
        self._rho = params.rho
        self._V_inf = params.V_inf
        self._V_inf_vec = params.V_inf_vec()
        self._AR = params.AR
        self._S = params.S
        self._b = params.b
        self._MAC = params.MAC
        self._r_ref = params.r_ref
        self._q_inf = 0.5 * self._rho * self._V_inf**2
        self._alfa_deg = params.alfa_deg
        self._beta_deg = params.beta_deg

        self._C14_x, self._C14_y, self._C14_z = wing_mesh.get_C14() # Ring vortices' corners
        self._delta_y = np.abs(np.diff(self._C14_y[-1, :])) # Span of each panel strip
        self._chords = wing_mesh.get_chords() # Chords of each panel strip
        self._y_sec = self._C14_y[-1, :-1] + self._delta_y * 0.5 # y coordinate of each panel strip

        self._result = Result()
        self._tolerances = Result.Coefs_3D(params.CL_tol, params.CD_tol, params.CY_tol, params.CMl_tol, params.CM_tol, params.CN_tol)

    # Verify if convergence criteria were met
    def _check_convergence(self):
        self._converged = np.all((self._result.residuals.to_array() - self._tolerances.to_array()) < 0.0) or self._wake_fixed

    # Getter for the results object
    def export_results(self):
        return self._result
    
    # Getter for the sectional coefficients
    def get_sectional_results(self):
        return self._result.Cl_sec, self._result.Cd_sec
    
    # Print results to the console
    def print_results(self):
        self._result.print(self._iter, self._wake_fixed)

    # Getter for the converged flag
    def is_converged(self):
        return self._converged

    # Main routine that computes the aero coefficients based on the vortices' intensities (Gammas)
    def compute_coefficients(self, Gammas: np.ndarray, Cd_p: np.ndarray, plot=False):
        S_ref = 0.5 * self._S if self._sym else self._S # For symmetric analysis, reference area need to be half of the provided
        w_ind = self._B_trefftz @ Gammas[-1, :].T # Compute downwash in the Trefftz plane. Equation 5.81 of (Drela)

        # Get the effective Gammas for the bound vortices (effective means subtracting the intensity from the trailing vortex of the panel in front)
        Gamma_bound = np.vstack([Gammas[0, :], np.diff(Gammas, axis=0)]).reshape(-1, 1)  

        # Compute local velocities at the bound vortices. Equation 6.42 of (Drela)
        Vi_boundX = self._V_hat_bound[:, :, 0] @ Gamma_bound + self._V_inf_vec[0]
        Vi_boundY = self._V_hat_bound[:, :, 1] @ Gamma_bound + self._V_inf_vec[1]
        Vi_boundZ = self._V_hat_bound[:, :, 2] @ Gamma_bound + self._V_inf_vec[2]

        Vi_boundX = Vi_boundX.reshape(Gammas.shape[0], -1)
        Vi_boundY = Vi_boundY.reshape(Gammas.shape[0], -1)
        Vi_boundZ = Vi_boundZ.reshape(Gammas.shape[0], -1)

        # Trefftz plane panels geometry calculations
        dy, dz = np.diff(self._C14_y[-1, :]), np.diff(self._C14_z[-1, :])
        theta_i = np.atan2(dz, dy) # Panel inclination
        delta_s_i = np.sqrt(dy**2 + dz**2) # Panel length

        # Compute moments following procedure in section 6.5.5 of (Drela)
        delta_M = np.zeros((Gammas.size, 3))
        k = 0
        for i in range(Gammas.shape[0]):
            for j in range(Gammas.shape[1]):
                r_a = np.array([self._C14_x[i, j], self._C14_y[i, j], self._C14_z[i, j]]) # Figure 6.8 of (Drela)
                r_b = np.array([self._C14_x[i, j + 1], self._C14_y[i, j + 1], self._C14_z[i, j + 1]]) # Figure 6.8 of (Drela)
                l_i = r_b - r_a # Equation 6.44 of (Drela)
                r_i = 0.5 * (r_a + r_b) # Equation 6.43 of (Drela)

                Vi_bound = np.array([Vi_boundX[i, j], Vi_boundY[i, j], Vi_boundZ[i, j]])
                delta_F = self._rho * np.cross(Vi_bound, l_i) * Gamma_bound[k] # Equation 6.44 of (Drela)
                delta_M[k, :] = np.cross(r_i - self._r_ref, delta_F) # Equation 6.47 of (Drela)
                k += 1

        ca, sa = np.cos(np.deg2rad(self._alfa_deg)), np.sin(np.deg2rad(self._alfa_deg))
        T_stab = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]]) # Equation 6.4 of (Drela)

        M = np.sum(delta_M, axis=0) # Equation 6.47 of (Drela)
        M_stab = T_stab @ M # Equation 6.5 of (Drela)

        # Equation 6.49 of (Drela)
        CMl = -M_stab[0] / (self._q_inf * S_ref * self._b) if not self._sym else 0.0 
        CM = M_stab[1] / (self._q_inf * S_ref * self._MAC)
        CN = -M_stab[2] / (self._q_inf * S_ref * self._b) if not self._sym else 0.0

        # Forces are computed in the Trefftz plane
        delta_phi_i = Gammas[-1, :] # Equation 5.79 of (Drela)
        delta_L = self._rho * self._V_inf * delta_phi_i * np.cos(theta_i) * delta_s_i # Equation 5.78 of (Drela)
        delta_Y = -self._rho * self._V_inf * delta_phi_i * np.sin(theta_i) * delta_s_i # Equation 5.77 of (Drela)
        delta_Di = - 0.5 * self._rho * w_ind * delta_phi_i * delta_s_i # Equation 5.82 of (Drela)

        # Sectional coefficients are obtained by non-dimensionalizing with S_ref = (delta_y * chord)_sec
        Cl_sec = delta_L / (self._q_inf * self._delta_y * self._chords)
        Cd_sec = delta_Di / (self._q_inf * self._delta_y * self._chords)

        # 3D force coefficients
        CL = np.sum(delta_L) / (self._q_inf * S_ref)
        CY = np.sum(delta_Y) / (self._q_inf * S_ref) if not self._sym else 0.0

        # Compute full drag coefficient
        Dp_sec = self._q_inf * self._delta_y * self._chords * Cd_p # Parasite drag per section 
        CDp = np.sum(Dp_sec) / (self._q_inf * S_ref) # Parasite drag coefficient 
        CD = (np.sum(delta_Di)) / (self._q_inf * S_ref) + CDp

        coefs_3D = Result.Coefs_3D(CL, CD, CY, CMl, CM, CN)
        self._result.update(self._y_sec, Cl_sec, CDp, coefs_3D, self._AR) # Update the results
        self._check_convergence()

        if plot:
            self._plot_trefftz() 

    # Compute sectional coefficients for the decambering routine
    def compute_coefficients_decambering(self, Gammas: np.ndarray):
        ny = Gammas.shape[1]
        delta_L = np.zeros(ny)
        Cl_sec = np.zeros(ny)

        for j in range(ny):
            # Computation in the Trefftz plane
            delta_L[j] = self._rho * self._V_inf * Gammas[-1, j] * self._delta_y[j]
            Cl_sec[j] = delta_L[j] / (0.5 * self._rho * self._V_inf**2 * self._delta_y[j] * self._chords[j])

        return Cl_sec

    # Plot Cl distribution in the Trefftz plane
    def _plot_trefftz(self):
        plt.figure()
        plt.plot((self._C14_y[-1, :-1] + self._C14_y[-1, 1:]) / 2.0, self._result.Cl_sec)
        plt.xlabel("Y [m]")
        plt.ylabel("Cl [-]")
        plt.show(block=False)

    # Plot decambering distribution
    def plot_delta(self, delta: np.ndarray):
        plt.figure()
        plt.plot((self._C14_y[-1, :-1] + self._C14_y[-1, 1:]) / 2.0, np.rad2deg(delta))
        plt.xlabel("Y [m]")
        plt.ylabel("delta [°]")
        plt.ylim([-10.0, 10.0])
        plt.show()
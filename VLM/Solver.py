from .Decambering import Decambering
from .Flows import Flows
from .Panels import Panels
from .Parameters import Parameters
from .Post import Post
import numpy as np

# References: 
# - Katz, J.; Plotkin, A.; Low-Speed Aerodynamics
# - Drela, M.; Flight Vehicle Aerodynamics
# - de Vargas, L. A. T.; de Oliveira, P. H. I. A; A Fast Aerodynamic Procedure for a Complete Aircraft Design Using the Know Airfoil Characteristics, 2006

# Main VLM class. Given a mesh and parameters, coordinate all the steps for running the simulation
class Solver:
    def __init__(self, panels: Panels, params: Parameters, decamb_delta_init: np.ndarray | None = None):
        # Checks if a simulation with sideslip is being attempted with symmetry activated
        if params.sym and params.beta_deg != 0.0:
            raise ValueError("Sideslip angle is different than 0° and the symmetry flag is activated.")

        self._params = params       
        self._wake_panels = panels.get_wake_panels() # Get wake mesh

        self._wing_panels = panels.get_wing_panels() # Get wing mesh
        self._wing_C14X, self._wing_C14Y, self._wing_C14Z = self._wing_panels.C14_VORING() # Get vortices' corners in the format needed for the VORING routine
        self._wing_nx, self._wing_ny = self._wing_panels.get_dimensions() # Get wing mesh dimensions
        self._n_wing_panels = self._wing_nx * self._wing_ny # Number of total panels

        self._RHS: np.ndarray = np.zeros((self._n_wing_panels, 1))

        self._AIC, self._inv_AIC = self._compute_aerodynamic_influence_coefs() # Compute the aerodynamic influence matrix and its inverse
        V_hat_bound = self._compute_bound_vortex_velocity_coefs() # Compute bound vortices induced velocity influence matrix
        B_trefftz = self._compute_trefftz_influence_coefs() # Compute downwash influence matrix in the Trefftz plane

        self._post = Post(self._wing_panels, B_trefftz, V_hat_bound, params) # Initialize post-processing object
        self._decambering = Decambering(self._wing_panels, params, decamb_delta_init) # Initialize decambering object (with optional non-zero starting decambering)

        panels.print_wing_geom() # Print wing's geometric info
 
        if self._wake_panels is not None:
            self._wake_panels.print_wake_params() # Print time stepping wake parameters
            
        self._params.print_run_params() # Print run parameters

    # Routine that runs the simulation
    def solve(self, plot: bool = False):
        d_wake = self._params.wake_dt * self._params.V_inf_vec() # Distance covered during 1 time step

        while not self._post.is_converged():
            # Solve [AIC][Gammas] = [RHS]. Step 'e' of section 12.3 of (Katz, Plotkin)
            Gammas = self._solve_linear_system(decambering=self._params.decambering) 
            self._post.compute_coefficients(Gammas) # Post-process results
            self._post.print_results() # Print results
                
            if self._params.decambering:
                Gammas = self._decamber_wing(Gammas) # Run the decambering routine

            if self._wake_panels is not None:
                self._wake_panels.wake_rollup(self._wing_C14X, self._wing_C14Y, self._wing_C14Z, Gammas, d_wake) # Update wake in the time stepping model

        # Plot results
        if plot:
            self._post.compute_coefficients(Gammas, plot=True)
            self._post.plot_delta(self._decambering._delta)

        return self._post.export_results(), self._decambering._delta
    
    # Compute the aerodynamic infulence coefficients. Vectorized version of the procedure described in section 12.3 of (Katz, Plotkin)
    def _compute_aerodynamic_influence_coefs(self):
        control_points = self._wing_panels.control_points_VORING(self._n_wing_panels) # Control points in the format needed for the VORING routine
        normals = self._wing_panels.normal_VORING(self._n_wing_panels) # Normals in the format needed for the VORING routine

        # Compute influence coefficients. Step 'c' in section 12.3 of (Katz, Plotkin)
        V = Flows.VORING(self._wing_C14X, self._wing_C14Y, self._wing_C14Z, control_points, np.ones((self._n_wing_panels, self._n_wing_panels, 1)), self._params.sym, self._params.ground)
        AIC = np.sum(V * normals, axis=2)
        
        # For the fixed wake, apply the modification described in equation 12.22a of (Katz, Plotkin)
        if self._params.wake_fixed:
            C14X_w = np.hstack([self._wing_C14X[-self._wing_ny:, -1:-3:-1], 1e6 * np.ones((self._wing_ny, 2))]) # Consider the wake extending to infinity
            V_w = Flows.VORING(C14X_w, self._wing_C14Y[-self._wing_ny:, :], self._wing_C14Z[-self._wing_ny:, :], control_points[:, -self._wing_ny:, :], np.ones((self._n_wing_panels, self._wing_ny, 1)), self._params.sym, self._params.ground)
            AIC[:, -self._wing_ny:] += np.sum(V_w * normals[:, -self._wing_ny:], axis=2)

        inv_AIC = np.linalg.inv(AIC)
        return AIC, inv_AIC
    
    # Compute the infulence coefficients for calculating induced velocities at the bound vortices. Equations 6.32 and 6.42 of (Drela)
    def _compute_bound_vortex_velocity_coefs(self):
        control_points = self._wing_panels.control_points_bound_vortex(self._n_wing_panels) # Bound vortices midpoints in the format needed for the VORING routine
        V_hat = Flows.VORING(self._wing_C14X, self._wing_C14Y, self._wing_C14Z, control_points, np.ones((self._n_wing_panels, self._n_wing_panels, 1)), self._params.sym, self._params.ground, horseshoe=True)
        return V_hat

    # Compute the infulence coefficients for calculating downwash in the Trefftz plane. Equations 5.80 and 5.81 of (Drela)
    def _compute_trefftz_influence_coefs(self):
        control_points = self._wing_panels.control_points_TREFFTZ() # Trefftz plane panels' control points
        C14 = self._wing_panels.C14_TREFFTZ() # Trefftz plane panels' vortices locations
        normals = self._wing_panels.normal_TREFFTZ() # Trefftz plane panels' normals

        B_trefftz = np.zeros((self._wing_ny, self._wing_ny))

        for i in range(self._wing_ny):
            CP, normal = control_points[i, :], normals[i, :]

            for j in range(self._wing_ny):
                P1, P2 = C14[j, :], C14[j + 1, :]

                # Compute induced velocities for considering 2D point vortices. Equation 5.80 of (Drela)
                V_ind_trefftz = Solver.bij_trefftz(CP, P1, P2, 1.0, self._params.sym, self._params.ground)
                B_trefftz[i, j] = np.dot(V_ind_trefftz, normal)

        return B_trefftz

    # Compute the RHS vector at each iteration
    def _update_RHS(self, decambering: bool):
        V_inf_vec = self._params.V_inf_vec() # Freestream velocity vector
        normals = self._wing_panels.normal_RHS() # Panels' normals

        if decambering:
            normals = self._decambering.decamber_normals(normals) # Apply the decambering transformation to the normals

        # For the time stepping wake, its Gammas are known, thus its influence is summed to the RHS vector
        if self._wake_panels is not None:
            wake_influence = self._wake_panels.compute_wake_influence(self._wing_panels, self._wing_ny)
            self._RHS[:] = -np.sum((V_inf_vec + wake_influence) * normals, axis=1).reshape(-1, 1) # Add wake influence to the freestream velocity

        else:
            self._RHS[:] = -np.sum(V_inf_vec * normals, axis=1).reshape(-1, 1) # For the fixed wake, equation 12.24 from (Katz, Plotkin)

    # Solve [AIC][Gammas] = [RHS]. Step 'e' of section 12.3 of (Katz, Plotkin)
    def _solve_linear_system(self, decambering: bool = False):
        self._update_RHS(decambering) # Compute RHS
        Gammas = self._inv_AIC @ self._RHS # Multiply the already inverted AIC by RHS
        return Gammas.reshape(self._wing_nx, -1)

    # Decambering routine. Procedure described section "Non linearity" of (de Vargas, de Oliveira, 2006)
    def _decamber_wing(self, Gammas: np.ndarray):
        iter = 0
        delta_Cl = np.inf * np.ones((self._wing_ny, 1))

        while np.linalg.norm(delta_Cl) > self._params.decamb_Cl_tol and iter < self._params.decamb_max_iter:
            Cl_sec, _ = self._post.compute_coefficients_decambering(Gammas)
            alfa_eff = self._decambering.compute_effective_alfas(Cl_sec) # Compute effective angle of attack
            delta_Cl = self._decambering.compute_residuals(alfa_eff, Cl_sec) # Compute Cl residuals

            self._decambering.update_deltas(delta_Cl) # Update decambering values based on the residuals
            Gammas = self._solve_linear_system(decambering=True) # Re-solve the linear system
            iter += 1

        return Gammas
    
    # Compute induced velocities for considering 2D point vortices
    @staticmethod
    def bij_trefftz(P: np.ndarray, P1: np.ndarray, P2: np.ndarray, Gamma: float, sym: bool, ground: bool):
        V_ind_trefftz1 = Flows.VOR2D(P1[1], P1[2], P[1], P[2], Gamma) # Contribution of the left trailing vortex (starts at infinity and ends at the bound vortex)
        V_ind_trefftz2 = Flows.VOR2D(P2[1], P2[2], P[1], P[2], Gamma) # Contribution of the right trailing vortex (starts at the bound vortex and ends at infinity)
        V_ind = V_ind_trefftz1 - V_ind_trefftz2 # Changes signal because the vortices have opposite orientation

        # Sum symmetry influence. Similar to equation 12.11 from (Katz, Plotkin)
        if sym:
            P_sym = P * np.array([1.0, -1.0, 1.0])
            V_ind_trefftz1 = Flows.VOR2D(P1[1], P1[2], P_sym[1], P_sym[2], Gamma)
            V_ind_trefftz2 = Flows.VOR2D(P2[1], P2[2], P_sym[1], P_sym[2], Gamma)
            V_ind += (V_ind_trefftz1 - V_ind_trefftz2) * np.array([-1.0, 1.0])

            # Sum ground + symmetry influence
            if ground:
                P_ground_sym = P_sym * np.array([1.0, 1.0, -1.0])

                V_ind_trefftz1 = Flows.VOR2D(P1[1], P1[2], P_ground_sym[1], P_ground_sym[2], Gamma)
                V_ind_trefftz2 = Flows.VOR2D(P2[1], P2[2], P_ground_sym[1], P_ground_sym[2], Gamma)
                V_ind += (V_ind_trefftz1 - V_ind_trefftz2) * np.array([-1.0, -1.0])

        # Sum ground image influence. Similar to equation 12.13 from (Katz, Plotkin)
        if ground:
            P_ground = P * np.array([1.0, 1.0, -1.0])

            V_ind_trefftz1 = Flows.VOR2D(P1[1], P1[2], P_ground[1], P_ground[2], Gamma)
            V_ind_trefftz2 = Flows.VOR2D(P2[1], P2[2], P_ground[1], P_ground[2], Gamma)
            V_ind += (V_ind_trefftz1 - V_ind_trefftz2) * np.array([1.0, -1.0])

        return V_ind
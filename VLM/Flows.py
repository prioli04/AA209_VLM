import numpy as np

# Main reference: Katz, J.; Plotkin, A.; Low-Speed Aerodynamics

# Object that serves as singularity elements repository for the solver
class Flows:
    __eps = 1e-6 # Vortex core size

    # Compute velocity induced by a vortex ring (with option for using a horseshoe vortex instead)
    @classmethod
    def VORING(cls, C14X: np.ndarray, C14Y: np.ndarray, C14Z: np.ndarray, P: np.ndarray, Gamma: np.ndarray, sym: bool, ground: bool, horseshoe: bool = False):
        # n_vortices -> number of vortex rings
        # n_control -> number of control points
        # These might differ. When computing wake influence, for example
        
        # C14X, C14Y, C14Z -> (n_vortices, 4) matrix. Each row is a vortex ring and each column is a corner of said vortex rign
        # P -> (n_vortices, n_control, 3) matrix. Control points locations, already tiled for vectorized computation. 
        # Tiles are in the row dimensions (stacked horizontally). 3rd dimension has x, y, z values

        # Gamma -> (n_vortices, n_control, 1) matrix. Vortex intensity of each vortex ring, already tiled for vectorized computation. 
        # Tiles are in the row dimensions (stacked horizontally)

        # Extracts the 1/4 chords location of the vortices into a (nx*ny, 3) matrix
        P1 = np.vstack((C14X[:, 0], C14Y[:, 0], C14Z[:, 0])).T
        P2 = np.vstack((C14X[:, 1], C14Y[:, 1], C14Z[:, 1])).T
        P3 = np.vstack((C14X[:, 2], C14Y[:, 2], C14Z[:, 2])).T
        P4 = np.vstack((C14X[:, 3], C14Y[:, 3], C14Z[:, 3])).T

        if horseshoe:
            # Assumes P3 and P4 are the points defining the trailing vortices and sets their x position to a huge number
            P3[:, 0] += 1e20
            P4[:, 0] += 1e20

        # Repeat 'P1', 'P2', 'P3' and 'P4' nx*ny times. Repeated values over the column dimension (stacked vertically)
        P1 = np.tile(P1, [P.shape[0], 1, 1])
        P2 = np.tile(P2, [P.shape[0], 1, 1])
        P3 = np.tile(P3, [P.shape[0], 1, 1])
        P4 = np.tile(P4, [P.shape[0], 1, 1])

        # Sums VORTXL routine results for each segment of the vortex ring, as equation 12.16 from (Katz, Plotkin)
        V1 = Flows._VORTXL(P, P1, P2, Gamma)
        V2 = Flows._VORTXL(P, P2, P3, Gamma)
        V3 = Flows._VORTXL(P, P3, P4, Gamma)
        V4 = Flows._VORTXL(P, P4, P1, Gamma)
        V = V1 + V2 + V3 + V4

        # Sum ground image influence. Equation 12.13 from (Katz, Plotkin)
        if ground:
            P_ground = P * np.array([1.0, 1.0, -1.0])

            V1 = Flows._VORTXL(P_ground, P1, P2, Gamma)
            V2 = Flows._VORTXL(P_ground, P2, P3, Gamma)
            V3 = Flows._VORTXL(P_ground, P3, P4, Gamma)
            V4 = Flows._VORTXL(P_ground, P4, P1, Gamma)

            V += (V1 + V2 + V3 + V4) * np.array([1.0, 1.0, -1.0])

        # Sum symmetry influence. Equation 12.11 from (Katz, Plotkin)
        if sym:
            P_sym = P * np.array([1.0, -1.0, 1.0])
            V1 = Flows._VORTXL(P_sym, P1, P2, Gamma)
            V2 = Flows._VORTXL(P_sym, P2, P3, Gamma)
            V3 = Flows._VORTXL(P_sym, P3, P4, Gamma)
            V4 = Flows._VORTXL(P_sym, P4, P1, Gamma)

            V += (V1 + V2 + V3 + V4) * np.array([1.0, -1.0, 1.0])

            # Sum ground + symmetry influence
            if ground:
                P_ground_sym = P_sym * np.array([1.0, 1.0, -1.0])

                V1 = Flows._VORTXL(P_ground_sym, P1, P2, Gamma)
                V2 = Flows._VORTXL(P_ground_sym, P2, P3, Gamma)
                V3 = Flows._VORTXL(P_ground_sym, P3, P4, Gamma)
                V4 = Flows._VORTXL(P_ground_sym, P4, P1, Gamma)

                V += (V1 + V2 + V3 + V4) * np.array([1.0, -1.0, -1.0])

        return V

    # Compute velocity induced by a 2D point vortex
    @classmethod
    def VOR2D(cls, x0: float, z0: float, x: float, z: float, Gamma: float):
        # x0, z0 -> Vortex location
        # x, z -> collocation point location

        V = np.array([0.0, 0.0])
        r2 = (x - x0)**2 + (z - z0)**2 # Distance, squared, to the vortex

        # Only compute for distances greater than the vortex core
        if r2 > cls.__eps:
            u = (Gamma / (2 * np.pi * r2)) * (z - z0)
            w = - (Gamma / (2 * np.pi * r2)) * (x - x0)
            V[0], V[1] = u, w

        return V

    # Compute velocity induced by a finite vortex line. Implementation of procedure from section 10.4.5 of (Katz, Plotkin).
    @classmethod
    def _VORTXL(cls, P: np.ndarray, P1: np.ndarray, P2: np.ndarray, Gamma: np.ndarray):
        r1_vec = P - P1
        r2_vec = P - P2

        # Note: In numpy, having a (nx, ny) matrix is different from a (nx, ny, 1) matrix
        # So [:, :, np.newaxis] is used to create a 3rd dimension of size 1 in order to make the vectorized operations work
       
        # Step 1 - Compute cross products
        r1_cross_r2 = np.cross(r1_vec, r2_vec, axis=2)
        norm_r1_cross_r2 = np.linalg.norm(r1_cross_r2, axis=2)[:, :, np.newaxis]

        # Step 2 - Compute distances
        r1 = np.linalg.norm(r1_vec, axis=2)[:, :, np.newaxis]
        r2 = np.linalg.norm(r2_vec, axis=2)[:, :, np.newaxis]

        # Step 3 - Locate invalid points
        r_cut = np.logical_and(r1 > cls.__eps, r2 > cls.__eps) # Inside the vortex core
        r_cut = np.logical_and(r_cut, norm_r1_cross_r2 > cls.__eps).squeeze() # Vortex line too small (r1 and r2 nearly parallel)

        # Step 4 - Compute dot products
        r0_vec = P2 - P1
        r0_dot_r1 = np.sum(r0_vec * r1_vec, axis=2)[:, :, np.newaxis]
        r0_dot_r2 = np.sum(r0_vec * r2_vec, axis=2)[:, :, np.newaxis]

        # Step 5 - Compute induced velocity components. Equation 10.115 from (Katz, Plotkin)
        with np.errstate(divide='ignore', invalid='ignore'):
            V = (Gamma / (4.0 * np.pi * norm_r1_cross_r2**2)) * (r0_dot_r1 / r1 - r0_dot_r2 / r2) * r1_cross_r2

        V[~r_cut, :] = 0.0 # Set invalid points to 0
        return V
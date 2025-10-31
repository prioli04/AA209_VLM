import numpy as np

class Flows:
    __eps = 1e-6

    @classmethod
    def VORING(cls, C14X: np.ndarray, C14Y: np.ndarray, C14Z: np.ndarray, P: np.ndarray, Gamma: np.ndarray, sym: bool):
        P1 = np.vstack((C14X[:, 0], C14Y[:, 0], C14Z[:, 0])).T
        P2 = np.vstack((C14X[:, 1], C14Y[:, 1], C14Z[:, 1])).T
        P3 = np.vstack((C14X[:, 2], C14Y[:, 2], C14Z[:, 2])).T
        P4 = np.vstack((C14X[:, 3], C14Y[:, 3], C14Z[:, 3])).T

        P1 = np.tile(P1, [P.shape[0], 1, 1])
        P2 = np.tile(P2, [P.shape[0], 1, 1])
        P3 = np.tile(P3, [P.shape[0], 1, 1])
        P4 = np.tile(P4, [P.shape[0], 1, 1])

        V1 = Flows._VORTXL(P, P1, P2, Gamma)
        V2 = Flows._VORTXL(P, P2, P3, Gamma)
        V3 = Flows._VORTXL(P, P3, P4, Gamma)
        V4 = Flows._VORTXL(P, P4, P1, Gamma)

        V = V1 + V2 + V3 + V4

        if sym:
            P_sym = P * np.array([1.0, -1.0, 1.0])
            V1 = Flows._VORTXL(P_sym, P1, P2, Gamma)
            V2 = Flows._VORTXL(P_sym, P2, P3, Gamma)
            V3 = Flows._VORTXL(P_sym, P3, P4, Gamma)
            V4 = Flows._VORTXL(P_sym, P4, P1, Gamma)

            V += (V1 + V2 + V3 + V4) * np.array([1.0, -1.0, 1.0])

        return V

    @classmethod
    def VOR2D(cls, x0: float, z0: float, x: float, z: float, Gamma: float):
        V = np.array([0.0, 0.0, 0.0])
        r2 = (x - x0)**2 + (z - z0)**2

        if r2 > cls.__eps:
            u = (Gamma / (2 * np.pi * r2)) * (z - z0)
            w = - (Gamma / (2 * np.pi * r2)) * (x - x0)
            V[1], V[2]  = u, w

        return V

    @classmethod
    def _VORTXL(cls, P: np.ndarray, P1: np.ndarray, P2: np.ndarray, Gamma: np.ndarray):
        r1_vec = P - P1
        r2_vec = P - P2

        r1 = np.linalg.norm(r1_vec, axis=2)[:, :, np.newaxis]
        r2 = np.linalg.norm(r2_vec, axis=2)[:, :, np.newaxis]

        r1_cross_r2 = np.cross(r1_vec, r2_vec, axis=2)
        norm_r1_cross_r2 = np.linalg.norm(r1_cross_r2, axis=2)[:, :, np.newaxis]

        r_cut = np.logical_and(r1 > cls.__eps, r2 > cls.__eps)
        r_cut = np.logical_and(r_cut, norm_r1_cross_r2 > cls.__eps).squeeze()

        r0_vec = P2 - P1
        r0_dot_r1 = np.sum(r0_vec * r1_vec, axis=2)[:, :, np.newaxis]
        r0_dot_r2 = np.sum(r0_vec * r2_vec, axis=2)[:, :, np.newaxis]

        with np.errstate(divide='ignore', invalid='ignore'):
            V = (Gamma / (4.0 * np.pi * norm_r1_cross_r2**2)) * (r0_dot_r1 / r1 - r0_dot_r2 / r2) * r1_cross_r2

        V[~r_cut, :] = 0.0
        return V
from .Flows import Flows
from .PanelGrid import PanelGrid
from .WingPanels import WingPanels
from mpl_toolkits.mplot3d.axes3d import Axes3D # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np

class Wake(PanelGrid):
    def __init__(self, n_rows_deform_max: int, ny: int, dt: float, wing_TE_points: PanelGrid.GridVector3, TE_dx: float, sym: bool, ground: bool, plot_ax: Axes3D | None):
        super().__init__(0, ny)
        self._it = 0
        self._dt = dt
        self._TE_dx = TE_dx
        self._n_rows_deform_max = n_rows_deform_max
        self._wing_TE_points = wing_TE_points
        self._ground = ground
        self._sym = sym
        self._plot_ax = plot_ax
        self._wake_lines = None
        self._add_TE(start=True)
        self._Gammas = np.zeros((0, ny))

    def _C14_VORING(self):
        if self._it < 1:
            raise ValueError("C14_as_controls_points should not be called for 'self._it < 1'.")

        return super()._C14_VORING_base(self._C14X, self._C14Y, self._C14Z)

    def _add_TE(self, start: bool):
        if start:
            self._C14X[0, :] = self._wing_TE_points.X
            self._C14Y[0, :] = self._wing_TE_points.Y
            self._C14Z[0, :] = self._wing_TE_points.Z

        else:
            self._C14X = np.vstack((self._wing_TE_points.X, self._C14X))
            self._C14Y = np.vstack((self._wing_TE_points.Y, self._C14Y))
            self._C14Z = np.vstack((self._wing_TE_points.Z, self._C14Z))
            self._nx += 1

    def _step_wake(self, d_wake: np.ndarray):
        self._C14X += d_wake[0]
        self._C14Y += d_wake[1]
        self._C14Z += d_wake[2]
        
        self._add_TE(start=False)
        self._control_pointX, self._control_pointY, self._control_pointZ = self._compute_control_points(self._C14X, self._C14Y, self._C14Z)
        self._normalX, self._normalY, self._normalZ = self._compute_normals(self._C14X, self._C14Y, self._C14Z)

    def _update_Gammas(self, TE_Gammas: np.ndarray):
        self._Gammas = np.vstack((TE_Gammas, self._Gammas))

    def _C14_as_control_points(self, n_rows: int, n_tiles: int):
        if self._it < 1:
            raise ValueError("C14_as_controls_points should not be called for 'self._it < 1'.")

        CPX = np.tile(self._C14X[1:n_rows + 1, :].reshape(-1, 1), [1, n_tiles])
        CPY = np.tile(self._C14Y[1:n_rows + 1, :].reshape(-1, 1), [1, n_tiles])
        CPZ = np.tile(self._C14Z[1:n_rows + 1, :].reshape(-1, 1), [1, n_tiles])

        control_points = np.zeros((CPX.shape[0], n_tiles, 3))
        control_points[:, :, 0], control_points[:, :, 1], control_points[:, :, 2] = CPX, CPY, CPZ

        return control_points

    def _build_offset_map(self, wing_C14X: np.ndarray, wing_C14Y: np.ndarray, wing_C14Z: np.ndarray, wing_Gammas: np.ndarray):
        n_panel_rows = self._n_rows_deform_max if self._normalX.shape[0] >= self._n_rows_deform_max else self._normalX.shape[0]
        wake_C14X, wake_C14Y, wake_C14Z = self._C14_VORING()

        wake_C14_as_CP_wing = self._C14_as_control_points(n_panel_rows, wing_C14X.shape[0])
        wake_C14_as_CP_wake = self._C14_as_control_points(n_panel_rows, wake_C14X.shape[0])
        
        wing_Gammas = np.tile(wing_Gammas.reshape(1, -1), [wake_C14_as_CP_wing.shape[0], 1])[:, :, np.newaxis]
        wake_Gammas = np.tile(self._Gammas.reshape(1, -1), [wake_C14_as_CP_wake.shape[0], 1])[:, :, np.newaxis]

        offset_map_X = np.zeros((self._nx + 1, self._ny + 1))
        offset_map_Y = np.zeros((self._nx + 1, self._ny + 1))
        offset_map_Z = np.zeros((self._nx + 1, self._ny + 1))

        V = np.zeros((wake_C14_as_CP_wake.shape[0], 3))

        dV_wing = Flows.VORING(wing_C14X, wing_C14Y, wing_C14Z, wake_C14_as_CP_wing, wing_Gammas, self._sym, self._ground)
        V += np.sum(dV_wing, axis=1)

        dV_wake = Flows.VORING(wake_C14X, wake_C14Y, wake_C14Z, wake_C14_as_CP_wake, wake_Gammas, self._sym, self._ground)
        V += np.sum(dV_wake, axis=1)
            
        offset_point = self._dt * V
        offset_map_X[1:n_panel_rows + 1, :] = offset_point[:, 0].reshape(-1, self._ny + 1)
        offset_map_Y[1:n_panel_rows + 1, :] = offset_point[:, 1].reshape(-1, self._ny + 1)
        offset_map_Z[1:n_panel_rows + 1, :] = offset_point[:, 2].reshape(-1, self._ny + 1)

        return PanelGrid.GridVector3(offset_map_X, offset_map_Y, offset_map_Z)
    
    def _offset_wake(self, offset_map: PanelGrid.GridVector3):
        self._C14X += offset_map.X
        self._C14Y += offset_map.Y
        self._C14Z += offset_map.Z

        if self._sym:
            self._C14Y[self._C14Y < 0.0] = 0.0

        if self._ground:
            self._C14Z[self._C14Z < 0.01] = 0.01

        self._compute_control_points(self._C14X, self._C14Y, self._C14Z)
        self._compute_normals(self._C14X, self._C14Y, self._C14Z)

    def _update_wake_plot(self):
        if self._wake_lines is not None:
            self._wake_lines.remove()   

        if self._it > 0 and self._plot_ax is not None:
            if self._sym:
                x_sym = np.hstack([np.flip(self._C14X, axis=1), self._C14X[:, 1:]])
                y_sym = np.hstack([-np.flip(self._C14Y, axis=1), self._C14Y[:, 1:]])
                z_sym = np.hstack([np.flip(self._C14Z, axis=1), self._C14Z[:, 1:]])
                self._wake_lines = self._plot_ax.plot_wireframe(x_sym, y_sym, z_sym, rstride=0)

            else:
                self._wake_lines = self._plot_ax.plot_wireframe(self._C14X, self._C14Y, self._C14Z, rstride=0)
            
            self._plot_ax.set_aspect("equal")
            plt.pause(0.1)

    def wake_rollup(self, wing_C14X: np.ndarray, wing_C14Y: np.ndarray, wing_C14Z: np.ndarray, wing_Gammas: np.ndarray, d_wake: np.ndarray):
        self._step_wake(d_wake)
        self._update_Gammas(wing_Gammas[-1, :])

        if self._it != 0:
            offset_map = self._build_offset_map(wing_C14X, wing_C14Y, wing_C14Z, wing_Gammas)
            self._offset_wake(offset_map)

        self._it += 1
        self._update_wake_plot()

    def compute_wake_influence(self, wing_panels: WingPanels, wing_ny: int):
        if self._it == 0:
            return 0.0

        wake_C14X, wake_C14Y, wake_C14Z = self._C14_VORING()
        control_points = wing_panels.control_points_VORING(wake_C14X.shape[0])
        wake_Gammas = np.tile(self._Gammas.reshape(1, -1), [control_points.shape[0], 1])[:, :, np.newaxis]

        dV_w = Flows.VORING(wake_C14X, wake_C14Y, wake_C14Z, control_points, wake_Gammas, self._sym, self._ground)
        V_w = np.sum(dV_w, axis=1)

        self._w_wake = V_w[:, 2].reshape(-1, wing_ny)
        return V_w
    
    def print_wake_params(self):
        print("===== Wake Parameters =====")
        print(f"Maximum number of deformed rows: {self._n_rows_deform_max:d}")
        print(f"Time step: {self._dt:.3f} s")
        print(f"Wake shed {self._TE_dx:.3f} m after TE")
        print()

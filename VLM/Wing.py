from .Airfoil import Airfoil
from .PanelGrid import PanelGrid
from .Section import Section
from mpl_toolkits.mplot3d.axes3d import Axes3D # type: ignore[import-untyped]
from typing import List

import copy
import numpy as np


class Wing(PanelGrid):
    def __init__(self, b: float, S: float, nx: int, ny: int, Z: float, sections: List[Section], wake_dx: float):
        if len(sections) == 1:
            sections.append(Section(1.0, sections[0].fc, sections[0].x_offset, sections[0].airfoil_path_str))

        self._sections = sections
        self._b = b
        self._S = S
        self._AR = b**2 / S
        self._root_chord = self._compute_root_chord()
        self._MAC = self._compute_MAC()
        self._taper_ratio = self._sections[-1].fc 
        self._C14_sweep = self._compute_C14_sweep() 

        self._points = self._compute_points(nx, ny, Z)
        super().__init__(nx, ny, self._points, wake_dx=wake_dx)
        self._w_ind_trefftz = np.zeros(ny)

    def _compute_points(self, nx: int, ny: int, Z: float):
        x = np.linspace(0, self._root_chord, nx + 1)
        y = np.linspace(0, self._b / 2.0, ny + 1)

        corners_x, corners_y = np.meshgrid(x, y, indexing="ij")
        corners_z = Z * np.ones_like(corners_x)

        corners_x, corners_z = self._apply_sections(corners_x, corners_y, corners_z, self._b / 2.0)

        return super().GridVector3(corners_x, corners_y, corners_z)
    
    def _compute_root_chord(self):
        sum_fS = 0.0

        for i in range(len(self._sections) - 1):
            b_i = (self._b / 2.0) * (self._sections[i + 1].fy_pos - self._sections[i].fy_pos)
            fcr_i = self._sections[i].fc
            fct_i = self._sections[i + 1].fc

            sum_fS += b_i * (fcr_i + fct_i) / 2.0

        return 0.5 * self._S / sum_fS
    
    def _compute_MAC(self):
        sum_Si_MACi = 0.0
        
        for i in range(len(self._sections) - 1):
            b_i = (self._b / 2.0) * (self._sections[i + 1].fy_pos - self._sections[i].fy_pos)
            fcr_i = self._sections[i].fc
            fct_i = self._sections[i + 1].fc

            t_i = fct_i / fcr_i
            S_i = b_i * self._root_chord * (fcr_i + fct_i) / 2.0

            root_chord_i = self._root_chord * fcr_i
            MAC_i = (2.0 / 3.0) * root_chord_i * ((1 + t_i + t_i**2) / (1 + t_i))
            sum_Si_MACi += S_i * MAC_i

        return sum_Si_MACi / (0.5 * self._S)
    
    def _compute_C14_sweep(self):
        C14_sweep = 0.0

        if len(self._sections) != 1:
            xC14_root = 0.25 * self._root_chord * self._sections[0].fc + self._sections[0].x_offset
            xC14_tip = 0.25 * self._root_chord * self._sections[-1].fc + self._sections[-1].x_offset

            dx = xC14_tip - xC14_root
            dy = 0.5 * self._b

            C14_sweep = np.rad2deg(np.atan(dx / dy))

        return C14_sweep

    def _apply_sections(self, corners_x: np.ndarray, corners_y: np.ndarray, corners_z: np.ndarray, b_max: float):
        fc_current = self._sections[0].fc
        x_offset_current = self._sections[0].x_offset
        fy_current = self._sections[0].fy_pos
        airfoil_current = Airfoil.read(self._sections[0].airfoil_path)

        fc_next = self._sections[1].fc
        x_offset_next = self._sections[1].x_offset
        fy_next = self._sections[1].fy_pos
        airfoil_next = Airfoil.read(self._sections[1].airfoil_path)

        next_counter = 1

        for i_sec in range(corners_x.shape[1]):
            fy_sec = corners_y[0, i_sec] / b_max

            fc = np.interp(fy_sec, [fy_current, fy_next], [fc_current, fc_next])            
            x_offset = np.interp(fy_sec, [fy_current, fy_next], [x_offset_current, x_offset_next])

            corners_x[:, i_sec] *= fc
            corners_x[:, i_sec] += x_offset
            corners_z[:, i_sec] += airfoil_current.get_camber_line(corners_z.shape[0], fc * self._root_chord)

            if fy_sec > fy_next:
                fc_current = self._sections[next_counter].fc
                x_offset_current = self._sections[next_counter].x_offset
                fy_current = self._sections[next_counter].fy_pos
                airfoil_current = copy.deepcopy(airfoil_next)

                fc_next = self._sections[next_counter + 1].fc
                x_offset_next = self._sections[next_counter + 1].x_offset
                fy_next = self._sections[next_counter + 1].fy_pos
                airfoil_next = Airfoil.read(self._sections[next_counter + 1].airfoil_path)

                next_counter += 1

        return corners_x, corners_z
    
    def update_w_ind_trefftz(self, w_ind: np.ndarray):
        self._w_ind_trefftz[:] = w_ind

    def update_Gammas(self, Gammas: np.ndarray):
        self._Gammas[:] = Gammas

    def extract_TE_points(self):
        return super().GridVector3(self._C14X[-1, :], self._C14Y[-1, :], self._C14Z[-1, :])

    def C14_VORING(self):
        return super()._C14_VORING_base(self._C14X, self._C14Y, self._C14Z)
    
    def C14_TREFFTZ(self):
        C14X = self._C14X[-1, :].reshape(-1, 1)
        C14Y = self._C14Y[-1, :].reshape(-1, 1)
        C14Z = self._C14Z[-1, :].reshape(-1, 1)
        return np.hstack((C14X, C14Y, C14Z))
    
    def control_points_VORING(self, n_tiles: int):
        n_points = self._nx * self._ny

        CPX = np.tile(self._control_pointX.reshape(-1, 1), [1, n_tiles])
        CPY = np.tile(self._control_pointY.reshape(-1, 1), [1, n_tiles])
        CPZ = np.tile(self._control_pointZ.reshape(-1, 1), [1, n_tiles])

        control_points = np.zeros((n_points, n_tiles, 3))
        control_points[:, :, 0], control_points[:, :, 1], control_points[:, :, 2] = CPX, CPY, CPZ
        return control_points
    
    def control_points_TREFFTZ(self):
        CPX = self._control_pointX[-1, :].reshape(-1, 1)
        CPY = self._control_pointY[-1, :].reshape(-1, 1)
        CPZ = self._control_pointZ[-1, :].reshape(-1, 1)
        return np.hstack((CPX, CPY, CPZ))

    def normal_RHS(self):
        normalX = self._normalX.reshape(-1, 1)
        normalY = self._normalY.reshape(-1, 1)
        normalZ = self._normalZ.reshape(-1, 1)
        return np.hstack((normalX, normalY, normalZ))
    
    def normal_VORING(self, n_tiles: int):
        n_panels = self._nx * self._ny

        NX = np.tile(self._normalX.reshape(-1, 1), [1, n_tiles])
        NY = np.tile(self._normalY.reshape(-1, 1), [1, n_tiles])
        NZ = np.tile(self._normalZ.reshape(-1, 1), [1, n_tiles])

        normals = np.zeros((n_panels, n_tiles, 3))
        normals[:, :, 0], normals[:, :, 1], normals[:, :, 2] = NX, NY, NZ
        return normals
    
    def normal_TREFFTZ(self):
        normalX = self._normalX[-1, :].reshape(-1, 1)
        normalY = self._normalY[-1, :].reshape(-1, 1)
        normalZ = self._normalZ[-1, :].reshape(-1, 1)
        return np.hstack((normalX, normalY, normalZ))

    def plot_mesh(self, ax: Axes3D):
        ax.plot_surface(self._points.X, self._points.Y, self._points.Z)

    def print_wing_geom(self):
        print("===== Wing Geometry =====")
        print(f"Wing Span: {self._b:.3f} m")
        print(f"Wing Area: {self._S:.3f} m²")
        print(f"Aspect Ratio: {self._AR:.3f}")
        print(f"MAC: {self._MAC:.3f} m")
        print(f"Root Chord: {self._root_chord:.3f} m (Taper Ratio = {self._taper_ratio:.3f})")
        print(f"Root-Tip Sweep: {self._C14_sweep:.3f}°")
        print(f"Tip Twist: {0:.3f}°")
        print()
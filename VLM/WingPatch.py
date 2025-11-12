from .Airfoil import Airfoil
from .Section import Section

import numpy as np

class WingPatch:
    def __init__(self, nx: int, ny: int):
        self.nx, self.ny = nx, ny
        self._section_left: Section | None = None
        self._section_right: Section | None = None

    def _apply_sections(self, root: Section, tip: Section, corners_x: np.ndarray, corners_y: np.ndarray, corners_z: np.ndarray, wing_root_chord: float, wing_semi_span: float):
        airfoil_root = Airfoil.read(root.airfoil_path)
        airfoil_tip = Airfoil.read(tip.airfoil_path)

        for i_sec in range(corners_x.shape[1]):
            fy_sec = corners_y[0, i_sec] / wing_semi_span
           
            fc = np.interp(fy_sec, [root.fy_pos, tip.fy_pos], [root.fc, tip.fc])            
            x_offset = np.interp(fy_sec, [root.fy_pos, tip.fy_pos], [root.x_offset, tip.x_offset])
            twist_deg = np.interp(fy_sec, [root.fy_pos, tip.fy_pos], [root.twist_deg, tip.twist_deg])

            root_camber_x, root_camber_z = airfoil_root.get_camber_line(corners_x[:, i_sec], fc * wing_root_chord, twist_deg)
            tip_camber_x, tip_camber_z = airfoil_tip.get_camber_line(corners_x[:, i_sec], fc * wing_root_chord, twist_deg)

            for j in range(corners_x.shape[0]):
                corners_x[j, i_sec] = np.interp(fy_sec, [root.fy_pos, tip.fy_pos], [root_camber_x[j], tip_camber_x[j]])
                corners_z[j, i_sec] = np.interp(fy_sec, [root.fy_pos, tip.fy_pos], [root_camber_z[j], tip_camber_z[j]])
            
            corners_x[:, i_sec] += x_offset

        return corners_x, corners_z

    def compute_points(self, wing_root_chord: float, wing_semi_span: float, Z: float):
        root, tip = self._section_left, self._section_right

        if root is None or tip is None:
            raise ValueError("Sections not initialized.")

        x = np.linspace(0.0, 1.0, self.nx + 1)
        y = wing_semi_span * np.linspace(root.fy_pos, tip.fy_pos, self.ny + 1)

        corners_x, corners_y = np.meshgrid(x, y, indexing="ij")
        corners_z = np.zeros_like(corners_x)

        corners_x, corners_z = self._apply_sections(root, tip, corners_x, corners_y, corners_z, wing_root_chord, wing_semi_span)
        corners_z += Z

        return corners_x, corners_y, corners_z

    def set_left_section(self, section: Section):
        self._section_left = section

    def set_right_section(self, section: Section):
        self._section_right = section

    def left_section(self):
        return self._section_left
    
    def right_section(self):
        return self._section_right
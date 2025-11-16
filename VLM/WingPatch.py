from .Airfoil import Airfoil
from .Decambering import Decambering
from .Section import Section
from enum import Enum, auto

import numpy as np

class DiscretizationType(Enum):
        UNIFORM = auto()
        COSINE = auto()
        SINE = auto()
        MINUS_SINE = auto()

class WingPatch:
    def __init__(self, nx: int, ny: int, disc_type_x: DiscretizationType, disc_type_y: DiscretizationType):
        self.nx, self.ny = nx, ny
        self._disc_type_x, self._disc_type_y = disc_type_x, disc_type_y
        self._section_left: Section | None = None
        self._section_right: Section | None = None

        self._x_vals, self._y_vals = np.zeros(nx + 1), np.zeros(ny + 1)

    def _discretize_vals(self, x_min: float, x_max: float, n: int, disc_type: DiscretizationType):
        x_uniform = np.linspace(0.0, 1.0, n)
        theta = x_uniform * np.pi

        if disc_type == DiscretizationType.UNIFORM:
            x = x_uniform
        
        elif disc_type == DiscretizationType.COSINE:
            x = (1.0 - np.cos(theta)) / 2.0

        elif disc_type == DiscretizationType.SINE:
            x = 1 - np.cos(theta / 2.0)
        
        elif disc_type == DiscretizationType.MINUS_SINE:
            x = np.sin(theta / 2.0)
        
        return (x_max - x_min) * x + x_min

    def _apply_sections(self, root: Section, tip: Section, x_grid: np.ndarray, y_grid: np.ndarray, wing_root_chord: float, wing_semi_span: float):
        airfoil_root = Airfoil.read(root.airfoil_path)
        airfoil_tip = Airfoil.read(tip.airfoil_path)

        corners_x, corners_y, corners_z = np.zeros_like(x_grid), np.zeros_like(x_grid), np.zeros_like(x_grid)

        for i_sec in range(x_grid.shape[1]):
            fy_sec = y_grid[0, i_sec]
           
            fc = np.interp(fy_sec, [root.fy_pos, tip.fy_pos], [root.fc, tip.fc])            
            x_offset = np.interp(fy_sec, [root.fy_pos, tip.fy_pos], [root.x_offset, tip.x_offset])
            twist_deg = np.interp(fy_sec, [root.fy_pos, tip.fy_pos], [root.twist_deg, tip.twist_deg])

            root_camber_x, root_camber_z = airfoil_root.get_camber_line(x_grid[:, i_sec], fc * wing_root_chord, twist_deg)
            tip_camber_x, tip_camber_z = airfoil_tip.get_camber_line(x_grid[:, i_sec], fc * wing_root_chord, twist_deg)

            for j in range(x_grid.shape[0]):
                corners_x[j, i_sec] = np.interp(fy_sec, [root.fy_pos, tip.fy_pos], [root_camber_x[j], tip_camber_x[j]])
                corners_z[j, i_sec] = np.interp(fy_sec, [root.fy_pos, tip.fy_pos], [root_camber_z[j], tip_camber_z[j]])
            
            corners_x[:, i_sec] += x_offset

        corners_y = wing_semi_span * y_grid 
        return corners_x, corners_y, corners_z

    def compute_points(self, wing_root_chord: float, wing_semi_span: float, Z: float):
        root, tip = self._section_left, self._section_right

        if root is None or tip is None:
            raise ValueError("Sections not initialized.")

        self._x_vals = self._discretize_vals(0.0, 1.0, self.nx + 1, self._disc_type_x)
        self._y_vals = self._discretize_vals(root.fy_pos, tip.fy_pos, self.ny + 1, self._disc_type_y)

        x_grid, y_grid = np.meshgrid(self._x_vals, self._y_vals, indexing="ij")

        corners_x, corners_y, corners_z = self._apply_sections(root, tip, x_grid, y_grid, wing_root_chord, wing_semi_span)
        corners_z += Z

        return corners_x, corners_y, corners_z

    def compute_chords(self, wing_root_chord: float):
        root, tip = self._section_left, self._section_right

        if root is None or tip is None:
            raise ValueError("Sections not initialized.")
        
        chords = np.zeros_like(self._x_vals)

        for i_sec in range(len(chords)):
            fy_sec = self._y_vals[0, i_sec]
            chords[i_sec] = np.interp(fy_sec, [root.fy_pos, tip.fy_pos], [root.fc, tip.fc]) * wing_root_chord

        return chords

    def set_left_section(self, section: Section):
        self._section_left = section

    def set_right_section(self, section: Section):
        self._section_right = section

    def left_section(self):
        return self._section_left
    
    def right_section(self):
        return self._section_right
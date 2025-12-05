from .Airfoil import Airfoil
from .Section import Section
from enum import Enum, auto
from typing import List

import numpy as np

# Enumeration of the avaliable discretization types
class DiscretizationType(Enum):
        UNIFORM = auto()
        COSINE = auto()
        SINE = auto()
        MINUS_SINE = auto()

# Object that creates the mesh for each part of the wing
class WingPatch:
    def __init__(self, nx: int, ny: int, disc_type_x: DiscretizationType, disc_type_y: DiscretizationType):
        self.nx, self.ny = nx, ny # Patch mesh dimensions
        self._disc_type_x, self._disc_type_y = disc_type_x, disc_type_y # Discretization type in each axis

        self._root_sec: Section | None = None # Root section
        self._tip_sec: Section | None = None # Tip section

        self._root_foil: Airfoil | None = None # Root section
        self._tip_foil: Airfoil | None = None # Tip section

        self._root_foil_id, self._tip_foil_id = 0, 0
        self._x_vals, self._y_vals = np.zeros(nx + 1), np.zeros(ny + 1)

    # Apply the different discretization types to an interval of values
    def _discretize_vals(self, x_min: float, x_max: float, n: int, disc_type: DiscretizationType):
        x_uniform = np.linspace(0.0, 1.0, n)
        theta = x_uniform * np.pi

        # These functions are the same as the ones implemented in the Athena Vortex Lattice (AVL) program
        if disc_type == DiscretizationType.UNIFORM:
            x = x_uniform
        
        elif disc_type == DiscretizationType.COSINE:
            x = (1.0 - np.cos(theta)) / 2.0

        elif disc_type == DiscretizationType.SINE:
            x = 1 - np.cos(theta / 2.0)
        
        elif disc_type == DiscretizationType.MINUS_SINE:
            x = np.sin(theta / 2.0)
        
        return (x_max - x_min) * x + x_min

    # Take a grid of [x, y] points and apply the section data to form the patch's mesh
    def _apply_sections(self, x_grid: np.ndarray, y_grid: np.ndarray, wing_root_chord: float, wing_semi_span: float):
        if self._root_sec is None or self._tip_sec is None:
            raise ValueError("Sections not initialized.")

        if self._root_foil is None or self._tip_foil is None:
            raise ValueError("Airfoils not initialized.")

        corners_x, corners_y, corners_z = np.zeros_like(x_grid), np.zeros_like(x_grid), np.zeros_like(x_grid)

        for i_sec in range(x_grid.shape[1]):
            fy_sec = y_grid[0, i_sec] # Position of the panel strip, in fractions of semi span
            fy_root, fy_tip = self._root_sec.fy_pos, self._tip_sec.fy_pos # Position of the root section and tip section, in fractions of semi span
           
            # Linearly interpolate between root and tip values
            fc = np.interp(fy_sec, [fy_root, fy_tip], [self._root_sec.fc, self._tip_sec.fc])            
            x_offset = np.interp(fy_sec, [fy_root, fy_tip], [self._root_sec.x_offset, self._tip_sec.x_offset])
            z_offset = np.interp(fy_sec, [fy_root, fy_tip], [self._root_sec.z_offset, self._tip_sec.z_offset])
            twist_deg = np.interp(fy_sec, [fy_root, fy_tip], [self._root_sec.twist_deg, self._tip_sec.twist_deg])

            # Get camber line for root and tip airfoils scaled by chords and rotate by twist angle
            root_camber_x, root_camber_z = self._root_foil.get_camber_line(x_grid[:, i_sec], fc * wing_root_chord, twist_deg)
            tip_camber_x, tip_camber_z = self._tip_foil.get_camber_line(x_grid[:, i_sec], fc * wing_root_chord, twist_deg)

            # Linearly interpolate the camber lines
            for j in range(x_grid.shape[0]):
                corners_x[j, i_sec] = np.interp(fy_sec, [fy_root, fy_tip], [root_camber_x[j], tip_camber_x[j]])
                corners_z[j, i_sec] = np.interp(fy_sec, [fy_root, fy_tip], [root_camber_z[j], tip_camber_z[j]])
            
            # Add the x and z offsets
            corners_x[:, i_sec] += x_offset
            corners_z[:, i_sec] += z_offset

        corners_y = wing_semi_span * y_grid # Stretch mesh by the semi span
        return corners_x, corners_y, corners_z

    # Compute the patch's mesh points
    def compute_points(self, wing_root_chord: float, wing_semi_span: float, Z: float):
        if self._root_sec is None or self._tip_sec is None:
            raise ValueError("Sections not initialized.")

        self._x_vals = self._discretize_vals(0.0, 1.0, self.nx + 1, self._disc_type_x) # x values assuming unitary chord
        self._y_vals = self._discretize_vals(self._root_sec.fy_pos, self._tip_sec.fy_pos, self.ny + 1, self._disc_type_y) # y values discretized in fractions of semi span

        x_grid, y_grid = np.meshgrid(self._x_vals, self._y_vals, indexing="ij") # Base grid

        # Apply the transformations to the base grid
        corners_x, corners_y, corners_z = self._apply_sections(x_grid, y_grid, wing_root_chord, wing_semi_span)
        corners_z += Z # Sum the distance to the ground to z coordinate

        return corners_x, corners_y, corners_z

    # Compute each panel strip chord    
    def compute_chords(self, wing_root_chord: float):
        if self._root_sec is None or self._tip_sec is None:
            raise ValueError("Sections not initialized.")
        
        chords = np.zeros(self.ny)

        # Linear interpolation between root and tip chords
        for i_sec in range(self.ny):
            fy_sec = self._y_vals[i_sec]
            fy_root, fy_tip = self._root_sec.fy_pos, self._tip_sec.fy_pos
            chords[i_sec] = np.interp(fy_sec, [fy_root, fy_tip], [self._root_sec.fc, self._tip_sec.fc]) * wing_root_chord

        return chords

    # Fill the root and tip sections
    def set_root_tip(self, patch_id, sections: List[Section], airfoils: List[Airfoil]):
        self._root_sec, self._tip_sec = sections[patch_id], sections[patch_id + 1]
        self._root_foil, self._tip_foil = airfoils[patch_id], airfoils[patch_id + 1]
        self._root_foil_id, self._tip_foil_id = patch_id, patch_id + 1

    # Getter for the root section
    def root(self):
        return self._root_sec
    
    # Getter for the tip section
    def tip(self):
        return self._tip_sec
from .Section import Section
from .WingPatch import WingPatch
from typing import List

import numpy as np

class WingGeometry:
    def __init__(self, sections: List[Section], patches: List[WingPatch], b: float, AR: float):
        if len(sections) != len(patches) + 1:
            raise ValueError(f"Expected {len(patches) + 1} sections, but {len(sections)} were given.")
        
        self._patches = patches

        for i in range(len(patches)):
            self._patches[i].set_left_section(sections[i])
            self._patches[i].set_right_section(sections[i + 1])

        self.b = b
        self.AR = AR
        self.S = b**2 / AR
        self.root_chord = self._compute_root_chord()
        self.MAC = self._compute_MAC()
        self.taper_ratio = self._patches[-1].right_section().fc
        self.C14_sweep = self._compute_C14_sweep() 

    def _compute_root_chord(self):
        sum_fS = 0.0

        for patch in self._patches:
            b_i = (self.b / 2.0) * (patch.right_section().fy_pos - patch.left_section().fy_pos)
            fcr_i = patch.left_section().fc
            fct_i = patch.right_section().fc

            sum_fS += b_i * (fcr_i + fct_i) / 2.0

        return 0.5 * self.S / sum_fS
    
    def _compute_MAC(self):
        sum_Si_MACi = 0.0
        
        for patch in self._patches:
            b_i = (self.b / 2.0) * (patch.right_section().fy_pos - patch.left_section().fy_pos)
            fcr_i = patch.left_section().fc
            fct_i = patch.right_section().fc

            t_i = fct_i / fcr_i
            S_i = b_i * self.root_chord * (fcr_i + fct_i) / 2.0

            root_chord_i = self.root_chord * fcr_i
            MAC_i = (2.0 / 3.0) * root_chord_i * ((1 + t_i + t_i**2) / (1 + t_i))
            sum_Si_MACi += S_i * MAC_i

        return sum_Si_MACi / (0.5 * self.S)
    
    def _compute_C14_sweep(self):
        root, tip = self._patches[0].left_section(), self._patches[-1].right_section()

        xC14_root = 0.25 * self.root_chord * root.fc + root.x_offset
        xC14_tip = 0.25 * self.root_chord * tip.fc + tip.x_offset

        dx = xC14_tip - xC14_root
        dy = 0.5 * self.b

        return np.rad2deg(np.atan(dx / dy))

    def get_patches(self):
        return self._patches

    def print_wing_geom(self):
        print("===== Wing Geometry =====")
        print(f"Wing Span: {self.b:.3f} m")
        print(f"Wing Area: {self.S:.3f} m²")
        print(f"Aspect Ratio: {self.AR:.3f}")
        print(f"MAC: {self.MAC:.3f} m")
        print(f"Root Chord: {self.root_chord:.3f} m (Taper Ratio = {self.taper_ratio:.3f})")
        print(f"Root-Tip Sweep: {self.C14_sweep:.3f}°")
        print(f"Tip Twist: {0:.3f}°")
        print()
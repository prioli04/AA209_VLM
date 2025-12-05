from .Airfoil import Airfoil
from .Section import Section
from .WingPatch import WingPatch
from typing import List

import numpy as np

# Object that describes the wing used in the simulations
class WingGeometry:
    def __init__(self, sections: List[Section], patches: List[WingPatch], b: float, AR: float):
        # Must have compatible numbers of sections and patches
        if len(sections) != len(patches) + 1:
            raise ValueError(f"Expected {len(patches) + 1} sections, but {len(sections)} were given.")
        
        self._patches = patches
        self._airfoils = [Airfoil.read(section.airfoil_path, section.xfoil_path) for section in sections] # Load the airfoils

        # Match the sections provided with the patches provided
        for i in range(len(patches)):
            self._patches[i].set_root_tip(i, sections, self._airfoils)

        # Geometrical parameters computation
        self.b = b # Span [m]
        self.AR = AR # Aspect Ratio [-]
        self.S = b**2 / AR # Planform area [m^2]
        self.root_chord = self._compute_root_chord() # Root chord [m]
        self.MAC = self._compute_MAC() # Mean aerodynamic chord [m]
        self.taper_ratio = self._patches[-1].tip().fc # Taper ratio [-]
        self.C14_sweep = self._compute_C14_sweep() # 1/4 chord sweep angle [°]

    # Compute wing's root chord
    def _compute_root_chord(self):
        sum_fS = 0.0

        for patch in self._patches:
            b_i = (self.b / 2.0) * (patch.tip().fy_pos - patch.root().fy_pos) # Patch span
            fcr_i = patch.root().fc # Patch root section root chord fraction
            fct_i = patch.tip().fc # Patch tip section root chord fraction

            # Patch area divided by root chord: fS = S_patch/c_root = b * (fcr + fct) / 2
            sum_fS += b_i * (fcr_i + fct_i) / 2.0

        # 0.5 * S_wing = sum(S_patch) = sum(fS * c_root) = c_root * sum(fS)
        # c_root = 0.5 * S_wing / sum(fS)
        # 1/2 the wing area since the patches model only the right side of the wing)
        return 0.5 * self.S / sum_fS # Compute the root chord for matching the provided wing area 
    
    # Compute wing's Mean Aerodynamic Chord
    def _compute_MAC(self):
        sum_Si_MACi = 0.0
        
        for patch in self._patches:
            b_i = (self.b / 2.0) * (patch.tip().fy_pos - patch.root().fy_pos) # Patch span
            fcr_i = patch.root().fc  # Patch root section root chord fraction
            fct_i = patch.tip().fc # Patch tip section root chord fraction

            t_i = fct_i / fcr_i # Patch taper ratio
            S_i = b_i * self.root_chord * (fcr_i + fct_i) / 2.0 # Patch area

            root_chord_i = self.root_chord * fcr_i # Patch root chord
            MAC_i = (2.0 / 3.0) * root_chord_i * ((1 + t_i + t_i**2) / (1 + t_i)) # Mean aerodynamic chord for a straight tapered wing
            sum_Si_MACi += S_i * MAC_i

        # MAC for the whole wing will be the weighted average 
        return sum_Si_MACi / (0.5 * self.S)
    
    # Compute the 1/4 sweep angle
    def _compute_C14_sweep(self):
        root, tip = self._patches[0].root(), self._patches[-1].tip()

        xC14_root = 0.25 * self.root_chord * root.fc + root.x_offset # 1/4 chord location of the wing root
        xC14_tip = 0.25 * self.root_chord * tip.fc + tip.x_offset # 1/4 location location of the wing tip

        dx = xC14_tip - xC14_root
        dy = 0.5 * self.b

        # tan(sweep_1/4) = dx/dy from 90° triangle
        return np.rad2deg(np.atan(dx / dy))

    # Getter for the wing patches
    def get_patches(self):
        return self._patches
    
    # Getter for the wing airfoils
    def get_airfoils(self):
        return self._airfoils

    # Print geometry information
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
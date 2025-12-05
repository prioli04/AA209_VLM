from dataclasses import dataclass, field
from pathlib import Path

# Object that represents a wing section (wing is built by linearly interpolating sections)
@dataclass(frozen=True)
class Section:
    fy_pos: float # Position along the span, given in fractions of the semispan 
    fc: float # Chord, given as fractions of the root chord
    x_offset: float # Offset along the x-axis [m]
    z_offset: float # Offset along the z-axis [m]
    twist_deg: float # Geometric twist angle [°]
    airfoil_path_str: str # Path string to the airfoil geometry file (both absolute and relative paths work)
    xfoil_path_str: str | None = None # Path string to the airfoil xfoil results file (both absolute and relative paths work)

    airfoil_path: Path = field(init=False) # Airfoil geometry path object
    xfoil_path: Path = field(init=False) # Airfoil xfoil results path object

    # Parses the file paths provided
    def __post_init__(self):
        airfoil_path = Path(self.airfoil_path_str)

        if not airfoil_path.is_file():
            raise FileNotFoundError(f"Unable to find file: {airfoil_path.resolve()}")
        
        super().__setattr__("airfoil_path", airfoil_path)

        if self.xfoil_path_str is not None:
            xfoil_path = Path(self.xfoil_path_str)

            if not xfoil_path.is_file():
                raise FileNotFoundError(f"Unable to find file: {xfoil_path.resolve()}")
            
            super().__setattr__("xfoil_path", xfoil_path)

        else:
            super().__setattr__("xfoil_path", None)

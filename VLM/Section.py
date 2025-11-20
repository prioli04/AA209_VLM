from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class Section:
    fy_pos: float 
    fc: float
    x_offset: float
    z_offset: float
    twist_deg: float
    airfoil_path_str: str
    xfoil_path_str: str | None = None

    airfoil_path: Path = field(init=False)
    xfoil_path: Path = field(init=False)

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

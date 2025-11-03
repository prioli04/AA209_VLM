from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class Section:
    fy_pos: float 
    fc: float
    x_offset: float
    airfoil_path_str: str
    airfoil_path: Path = field(init=False)

    def __post_init__(self):
        path = Path(self.airfoil_path_str)

        if not path.is_file():
            raise FileNotFoundError(f"Unable to find file: {path.resolve()}")
        
        super().__setattr__("airfoil_path", path)
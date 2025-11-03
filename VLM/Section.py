from dataclasses import dataclass, field

@dataclass(frozen=True)
class Section:
    fy_pos: float 
    fc: float
    x_offset: float
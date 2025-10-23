from enum import Enum, auto
import numpy as np
from .Mesh import Mesh

class Wake(Mesh):
    class Type(Enum):
        Fixed = auto()
        TimeStepping = auto()

    def __init__(self, TE_quarter_chords: Mesh.GridVector3, type: Type):
        if type == Wake.Type.Fixed:
            points = self._compute_fixed_wake_points(TE_quarter_chords)
            super().__init__(points)

        elif type == Wake.Type.TimeStepping:
            pass

    def _compute_fixed_wake_points(self, TE_quarter_chords: Mesh.GridVector3):
        TE_quarter_chords_x, TE_quarter_chords_y, TE_quarter_chords_z = TE_quarter_chords
        wake_points_x = np.vstack((TE_quarter_chords_x, TE_quarter_chords_x + 1e6))
        wake_points_y = np.vstack((TE_quarter_chords_y, TE_quarter_chords_y))
        wake_points_z = np.vstack((TE_quarter_chords_z, TE_quarter_chords_z))

        return super().GridVector3(wake_points_x, wake_points_y, wake_points_z)
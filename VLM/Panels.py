from .Mesh import Mesh
from .Wake import Wake
from .Wing import Wing

class Panels:
    def __init__(
            self,
            b: float,
            AR: float, 
            nx: int, 
            ny: int, 
            wake_type: Wake.Type):
        
        self._wing_mesh = Wing(b, AR, nx, ny)
        wing_C14 = self._wing_mesh.get_quarter_chords()
        TE_quarter_chords = Mesh.GridVector3(wing_C14.X[-1,:], wing_C14.Y[-1,:], wing_C14.Z[-1,:])
        self._wake_mesh = Wake(TE_quarter_chords, wake_type)

    def get_wing_n_panels(self):
        return self._wing_mesh.get_n_panels()
    
    def get_panels_combined(self):
        return Mesh.combine_meshes(self._wing_mesh, self._wake_mesh)
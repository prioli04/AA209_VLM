import numpy as np

class Parameters:
    def __init__(self, V_inf: float, alfa_deg: float, rho: float, S: float, AR: float):
        self.V_inf = V_inf
        self.alfa_rad = np.deg2rad(alfa_deg)
        self.rho = rho
        self.S = S
        self.AR = AR
        
    def V_inf_vec(self):
        return self.V_inf * np.array([np.cos(self.alfa_rad), 0.0, np.sin(self.alfa_rad)])
    
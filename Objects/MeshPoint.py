import numpy as np

class MeshPoint():
    def __init__(self, position:np.ndarray, normal:np.ndarray, tangent:np.ndarray, color:np.ndarray, velocity:np.ndarray=np.zeros(3)):
        self.position = position
        self.velocity = velocity
        self.normal = normal
        self.tangent = tangent
        self.color = color
    
    @property
    def velocity_magnitude(self) -> float:
        return float(np.linalg.norm(self.velocity))
    
    @property
    def velocity_direction(self) -> np.ndarray:
        magnitude = self.velocity_magnitude

        if magnitude > 0:
            return self.velocity / magnitude

        return np.array([0.0, 0.0, 0.0])
         
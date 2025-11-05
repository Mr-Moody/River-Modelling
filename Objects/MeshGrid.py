import numpy as np

from Objects.MeshPoint import MeshPoint

class MeshGrid():
    def __init__(self, width:int=10, height:int=10, cell_size:float=1):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.points = []  # MeshPoints for simulation calculations
        self.vertices = np.array([])  # Vertices for mesh rendering
        self.triangles = np.array([])  # Triangles for mesh rendering

    def generateMesh(self):
        num_vertices_x = self.width + 1
        num_vertices_z = self.height + 1
        total_vertices = num_vertices_x * num_vertices_z
        
        self.vertices = np.zeros((total_vertices, 3))
        self.points = []
        
        vertex_index = 0

        for i in range(num_vertices_x):
            for j in range(num_vertices_z):
                x = i * self.cell_size 
                y = j * self.cell_size  
                z = np.random.normal(0, 0.05)

                position = np.array([x, y, z])
                
                self.vertices[vertex_index] = position
                
                normal = np.array([0, 0, 1])
                tangent = np.array([1, 0, 0])
                color = np.array([np.random.random(), np.random.random(), np.random.random()])
                velocity = np.array([0.0, 0.0, 0.0])
                
                point = MeshPoint(position=position, normal=normal, tangent=tangent, color=color, velocity=velocity)
                self.points.append(point)
                
                vertex_index += 1
        
        # Allocate triangles for mesh rendering
        num_triangles = self.width * self.height * 2
        self.triangles = np.zeros(num_triangles * 3, dtype=int)
        
        triangle_index = 0

        # Convert square grid into triangles for mesh rendering
        for i in range(self.width):
            for j in range(self.height):
                bottom_left = i * num_vertices_z + j
                bottom_right = i * num_vertices_z + (j + 1)
                top_left = (i + 1) * num_vertices_z + j
                top_right = (i + 1) * num_vertices_z + (j + 1)
                
                # First triangle: bottom left, top left, bottom right
                self.triangles[triangle_index] = bottom_left
                self.triangles[triangle_index + 1] = top_left
                self.triangles[triangle_index + 2] = bottom_right

                triangle_index += 3
                
                # Second triangle: bottom right, top left, top right
                self.triangles[triangle_index] = bottom_right
                self.triangles[triangle_index + 1] = top_left
                self.triangles[triangle_index + 2] = top_right

                triangle_index += 3
    
    def updateVerticesFromPoints(self):
        for i, point in enumerate(self.points):
            self.vertices[i] = point.position
    
    def updatePointsFromVertices(self):
        for i, point in enumerate(self.points):
            point.position = self.vertices[i].copy()

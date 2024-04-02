# import pymesh
import numpy as np

class MeshGeneratorDelegate(object):
    def __init__(self) -> None:
        pass

    def build(self) -> None:
        raise NotImplementedError

class GenerateVTKPolyData(MeshGeneratorDelegate):
    def __init__(self) -> None:
        super().__init__()
    
    def build(self) -> None:
        raise NotImplementedError

class GenerateTrivialCube(MeshGeneratorDelegate):
    def __init__(self, extents : np.ndarray) -> None:
        super().__init__()

        # Define vertices of the cube 
        vertices = np.array([ 
            [-0.5, -0.5, -0.5], 
            [-0.5, -0.5, 0.5], 
            [-0.5, 0.5, -0.5], 
            [-0.5, 0.5, 0.5], 
            [0.5, -0.5, -0.5], 
            [0.5, -0.5, 0.5], 
            [0.5, 0.5, -0.5], 
            [0.5, 0.5, 0.5] 
        ]) 
        
        # Define indices of the cube 
        indices = np.array([ 
            [0, 1, 3], 
            [0, 3, 2], 
            [0, 2, 4], 
            [2, 6, 4], 
            [0, 4, 1], 
            [1, 4, 5], 
            [2, 3, 6], 
            [3, 7, 6], 
            [4, 6, 5], 
            [5, 6, 7], 
            [1, 5, 7], 
            [1, 7, 3] 
        ])

        self.triangles = vertices[indices]
        # Flatten the array
        self.triangles = self.triangles.reshape(-1, 3)


        # Add a fourth column of 0s
        self.triangles = np.hstack((self.triangles, np.zeros((len(self.triangles), 1)))) * extents

        self.nbTriangles = len(indices)

class GenerateRandomPoints(MeshGeneratorDelegate):
    def __init__(self, nbPoints : int) -> None:
        super().__init__()
        self.nbPoints = nbPoints

    def build(self) -> None:
        # Generate random points
        self.vertices = np.random.rand(self.nbPoints, 4)

class GenerateMesh(object):
    def __init__(self, _MeshGeneratorDelegate : MeshGeneratorDelegate) -> None:
        self._delegate = _MeshGeneratorDelegate

    def __getattr__(self, name):
        return getattr(self._delegate, name)
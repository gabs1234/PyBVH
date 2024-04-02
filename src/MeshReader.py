import numpy as np
import vtk

class MeshReaderDelegate(object):
    def __init__(self):
        self.nbTriangles = None
        self.triangles = None
        self.vertices = None
        self.normals = None

class StlReader(MeshReaderDelegate):
    def __init__(self, filename : str):
        super().__init__()
        reader = vtk.vtkSTLReader()
        reader.SetFileName(filename)
        reader.Update()
        polydata = reader.GetOutput()
        self.nbTriangles = polydata.GetNumberOfCells()
        self.triangles = np.array(polydata.GetPolys())
        all_vertices = []
        for i in range(polydata.GetNumberOfCells()):
            cell = polydata.GetCell(i)
            nb_points = cell.GetNumberOfPoints()
            cell_points = np.array([cell.GetPoints().GetPoint(pid) for pid in range(nb_points)])
            all_vertices.append(cell_points.flatten())
        self.vertices = np.array(all_vertices).flatten().reshape(-1, 3)
        # Add a fourth column of 0s
        self.vertices = np.hstack((self.vertices, np.zeros((len(self.vertices), 1))))
        self.normals = np.array(polydata.GetPointData().GetNormals())

class MeshReader():
    def __init__(self, meshReader: MeshReaderDelegate):
        self._delegate = meshReader
    
    def __getattr__(self, name):
        return getattr(self._delegate, name)
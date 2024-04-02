from Scene import DeviceScene, HostScene, Scene
from CudaPipeline import CudaPipeline
import numpy as np
import cupy as cp

class SceneManager:
    def __init__(self, pipeline: CudaPipeline, nbTriangles : np.uint32, vertices : np.ndarray):
        self.pipeline = pipeline

        self.nbTriangles = nbTriangles
        self.vertices = vertices
        self.bBoxMin = np.min(vertices, axis=0)
        self.bBoxMax = np.max(vertices, axis=0)
        self.hostScene = Scene(HostScene(self.nbTriangles, self.vertices, self.bBoxMin, self.bBoxMax))

        nbTriangles_tmp = cp.uint32(self.nbTriangles)
        bboxMin_tmp = cp.asarray(self.bBoxMin, dtype=cp.float32)
        bboxMax_tmp = cp.asarray(self.bBoxMax, dtype=cp.float32)
        vertices_tmp = cp.asarray(self.vertices, dtype=cp.float32)
        self.deviceScene = Scene(DeviceScene(nbTriangles_tmp, vertices_tmp, bboxMin_tmp, bboxMax_tmp))

        # Check that bBoxMin and bBoxMax have 4 components
        assert(len(self.bBoxMin) == 4)
        assert(len(self.bBoxMax) == 4)
from BVHTree import *
from TreeManager import TreeManager
from Scene import Scene
from CudaTimer import CudaTimer
from CudaPipeline import CudaPipeline

import cupy as cp
from cupy.cuda import MemoryPointer
import numpy as np

# TODO Handle errors

class BuilderDelegate(object):
    def __init__(self):
        self._builder = None
        self._builderName = None
        self._builderDescription = None
        self._builderParameters = None

        # TODO: Make this a parameter in the project
        self.mortonCodeType = np.uint32
    
    def build(self, Scene : Scene) -> BVHTree:
        raise NotImplementedError

class LBVHBuilder(BuilderDelegate):
    def __init__(self, pipeline : CudaPipeline, parallelGeometry=False):
        super().__init__()
        self.treeName = "LBVH"
        self.treeDescription = "LBVH Builder"
        self.treeParameters = None
        self.parallelGeometry = parallelGeometry

        self.builderSourceFile = "source.cu"
        self.builderName = "LBVH"

        self._pipeline = pipeline

        # self.LVBH_template = ["buildTreeKernel<unsigned int>"]
        
        # TODO: move this higher up in the builder hierarchy
        self._pipeline.readModuleFromFile(
            self.builderSourceFile, self.builderName,
            options=['-std=c++11'])
        
        self.kernelNames = [
            "bytesInAABB"
            # "bytesInBVHNode",
            # "bytesInCollisionList",
            # "bytesInInterceptDistances",
            # "kernelMortonCode",
            # "kernelRayBox",
            # "kernelBVHReset",
            # "kernelBVHConstruct",
            # "kernelBVHIntersection1",
            # "kernelBVHIntersection2",
            # "kernelBVHIntersection3"
        ]

        self.bindCudaKernels(self.builderName, *self.kernelNames)
    
    def bindCudaKernels(self, moduleName : str, *kernel : str):
        for k in kernel:
            self._pipeline.getKernelFromModule(moduleName, k)
        

    def build(self, scene : Scene) -> BVHTree:
        globalMemoryUsed = 0

        stream = cp.cuda.get_current_stream()
        
        timer = CudaTimer(stream)
        timer.start()

        
        
        timer.stop()
        print(f"Generate morton codes: {timer.elapsedTime()} ms")

        return None

class TRBVHBuilder(BuilderDelegate):
    def __init__(self, parallelGeometry=False):
        super().__init__()
        self._optimizationName = "TRBVH"
        self._optimizationDescription = "TRBVH Builder"
        self._optimizationParameters = None
        self._parallelGeometry = parallelGeometry

    def build(self, Scene : Scene) -> BVHTree:
        print ("TRBVHBuilder")
        return BVHTree(AoSTree(self._parallelGeometry))

class Builder:
    def __init__(self, _BuilderDelegate : BuilderDelegate) -> None:
        self._delegate = _BuilderDelegate

    def build(self, Scene : Scene) -> BVHTree:
        self._delegate.build(Scene)

    def __getattr__(self, name):
        return getattr(self._delegate, name)
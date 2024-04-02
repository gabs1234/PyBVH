from BVHTree import BVHTree
from CudaPipeline import CudaPipeline
import numpy as np
import cupy as cp

class TreeManager:
    def __init__(self, pipeline: CudaPipeline, tree : BVHTree):
        self.pipeline = pipeline
        self.tree = tree
    
    def toDevice(self, builderName : str, nbTriangles : int) -> cp.ndarray:
        treeSizePtr = cp.zeros(1, dtype=cp.uint32)
        kernel = self.pipeline.getKernelFromModule(builderName, "getTreeSizeKernel")
        kernel((1,), (1,), (treeSizePtr.data.ptr,))
        treeSize = treeSizePtr[0]

        # Allocate memory for tree structure
        self.deviceTreeStructurePtr = cp.cuda.alloc(treeSize)

        # Allocate tree arrays on device
        self.deviceTreeStructure = {}
        for key,value in self.tree.treeStructure.items():
            self.deviceTreeStructure[key] = cp.asarray(value)

        # Setup the tree structure on device
        kernel = self.pipeline.getKernelFromModule(builderName, "setTreeStructureKernel")
        kernel((1,), (1,), (
            self.deviceTreeStructurePtr,
            nbTriangles,
            self.deviceTreeStructure["parentNodes"].data.ptr,
            self.deviceTreeStructure["leftNodes"].data.ptr,
            self.deviceTreeStructure["rightNodes"].data.ptr,
            self.deviceTreeStructure["dataNodes"].data.ptr,
            self.deviceTreeStructure["bboxMinNodes"].data.ptr,
            self.deviceTreeStructure["bboxMaxNodes"].data.ptr,
            self.deviceTreeStructure["areaNodes"].data.ptr
        ))
        
        return self.deviceTreeStructurePtr
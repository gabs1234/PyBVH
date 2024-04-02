from BVHTree import *
from TreeManager import TreeManager
from Scene import Scene
from CudaTimer import CudaTimer
from CudaPipeline import CudaPipeline

import sys
from collections import deque

import matplotlib.pyplot as plt

import cupy as cp
from cupy.cuda import MemoryPointer
import numpy as np

IS_LEAF = 2147483648

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

        self.builderSourceFile = "LBVH_apetrei.cu"
        self.builderName = "LBVH"

        self.sortingSourceFile = "RadixSort.cu"
        self.sortingName = "RadixSort"

        self._pipeline = pipeline

        self.LVBH_template = ["buildTreeKernel<unsigned int>"]

        # self.fig = plt.figure()
        # self.ax = self.fig.add_subplot(111, projection='3d')
        # self.ax.set_xlabel('X')
        # self.ax.set_ylabel('Y')
        # self.ax.set_zlabel('Z')
        
        # TODO: move this higher up in the builder hierarchy
        self._pipeline.readModuleFromFile(
            self.builderSourceFile, self.builderName,
            options=['-std=c++11'])
        
        # self._pipeline.readModuleFromFile(
        #     self.sortingSourceFile, self.sortingName,
        #     options=['-std=c++14'])
        
        
    def deviceGetTreeStructureSize(self, scene : Scene) -> int:
        size = cp.zeros(1, dtype=cp.uint32)
        kernel = self._pipeline.getKernelFromModule(self.builderName, "getTreeStructureSize")
        kernel((1, ), (1, ), [size])
        return size[0]

    def deviceGetTreeNodeSize(self, scene : Scene) -> int:
        size = cp.zeros(1, dtype=cp.uint32)
        kernel = self._pipeline.getKernelFromModule(self.builderName, "getTreeNodeSize")
        kernel((1, ), (1, ), [size])
        return size[0]

    def deviceGetRootIndex(self) -> int:
        rootIndex = cp.zeros(1, dtype=cp.uint32)
        kernel = self._pipeline.getKernelFromModule(self.builderName, "getRootIndexKernel")
        kernel((1, ), (1, ), [self.treeStructurePtr, rootIndex])
        return rootIndex[0]
    
    def deviceInitializeTreeStructure(self, scene : Scene) -> None:
        
        kernel = self._pipeline.getKernelFromModule(self.builderName, "initializeTreeStructureKernel")

        # print the scene bounding box
        kernel((1, ), (1, ), [
            self.treeStructurePtr, scene.numberOfTriangles, scene.vertices,
            scene.bboxMin, scene.bboxMax, self.tree.bboxMinLeaf, self.tree.bboxMaxLeaf, self.tree.bboxMinInternal, self.tree.bboxMaxInternal,
            self.tree.mortonCodes, self.tree.sortedIndices, self.tree.enteredLeaf, self.tree.enteredInternal,
            self.tree.child_left, self.tree.child_right, self.tree.left_range, self.tree.right_range])
        
    
    def deviceGenerateMortonCodes2D(self, scene : Scene):
        blockSize = (1024, 1024, 1)
        gridSize = ((scene.numberOfTriangles + blockSize[0] - 1) // blockSize[0], 1, 1)

        # TODO: Optimize cache configuration


    def deviceGenerateMortonCodes3D(self, scene : Scene) -> None:
        nbElements = scene.numberOfTriangles
        blockSize = (1024, 1, 1)
        gridSize = (nbElements , 1, 1)

        kernel = self._pipeline.getKernelFromModule(self.builderName, "generateMortonCodesKernel3D")

        kernel(gridSize, blockSize, [
            scene.numberOfTriangles, scene.vertices, self.tree.mortonCodes,
            self.tree.bboxMinLeaf, self.tree.bboxMaxLeaf, scene.bboxMin, scene.bboxMax])
    
    def deviceSortMortonCodes(self, scene : Scene) -> None:
        blockSize = (1, 1, 1)
        gridSize = ((scene.numberOfTriangles + blockSize[0] - 1) , 1, 1)

        # TODO: Implement radix sort
        # For now, use CuPy's sort
        self.tree.sortedIndices[:] = cp.argsort(self.tree.mortonCodes)[:]
        self.tree.mortonCodes[:] = self.tree.mortonCodes[self.tree.sortedIndices]
        # self.tree.bboxMinLeaf[:] = self.tree.bboxMinLeaf[self.tree.sortedIndices]
        # self.tree.bboxMaxLeaf[:] = self.tree.bboxMaxLeaf[self.tree.sortedIndices]

    def deviceBuildTree(self, scene : Scene) -> None:
        nbElements = scene.numberOfTriangles
        blockSize = (1024, 1, 1)
        gridSize = (nbElements , 1, 1)

        kernel = self._pipeline.getKernelFromModule(self.builderName, "buildLVBHApetrei")
        kernel(gridSize, blockSize, [self.treeStructurePtr])

    def devicePrintTreeStructure(self, scene : Scene) -> None:
        kernel = self._pipeline.getKernelFromModule(self.builderName, "printTreeStructureKernel")
        kernel((1, ), (1, ), [self.treeStructurePtr])    

    def testTraversal(self, scene : Scene) -> None:
        kernel = self._pipeline.getKernelFromModule(self.builderName, "testTraversal")
        kernel((1, ), (1, ), [self.treeStructurePtr])

    def build(self, scene : Scene) -> BVHTree:
        globalMemoryUsed = 0

        stream = cp.cuda.get_current_stream()
        
        timer = CudaTimer(stream)
        print ("LBVHBuilder: begin to build tree")
        print (scene.numberOfTriangles)
        
         # Get tree structure size and allocate memory for tree structure
        self.treeStructureSize = self.deviceGetTreeStructureSize(scene)
        globalMemoryUsed += self.treeStructureSize
        self.treeStructurePtr = cp.cuda.alloc(self.treeStructureSize)

        # Allocate the tree
        self.tree = BVHTree(SoATree(scene.numberOfTriangles, self.treeStructurePtr))

        # Generate morton codes
        timer.start()
        self.deviceGenerateMortonCodes3D(scene)
        timer.stop()
        print(f"Generate morton codes: {timer.elapsedTime()} ms")
        print ("Morton codes generated")


        # Sort morton codes
        timer.reset()
        timer.start()
        self.deviceSortMortonCodes(scene)
        timer.stop()
        print(f"Sort morton codes: {timer.elapsedTime()} ms")
        print ("Morton codes sorted")

        print ("Morton codes: ", self.tree.mortonCodes)

        # Initialize tree structure
        timer.reset()
        timer.start()
        self.deviceInitializeTreeStructure (scene)
        timer.stop()
        print(f"Initialize tree structure: {timer.elapsedTime()} ms")
        print ("Tree structure initialized")

        # Build tree
        timer.reset()
        timer.start()
        self.deviceBuildTree (scene)
        timer.stop()
        print(f"Build tree: {timer.elapsedTime()} ms")

        print (self.tree.child_left)

        # Get the root node of the tree
        n = scene.numberOfTriangles
        self.rootIndex = self.deviceGetRootIndex()

        self.tree.root = self.rootIndex

        return self.tree

    def getTree(self) -> BVHTree:
        return self.tree
    
    def isLeaf(self, node : int) -> bool:
        return node & IS_LEAF
    
    
        
    def testRayTriangleIntersection(self) -> None:
        self.rayTriangleIntersection = self._pipeline.getKernelFromModule(self.builderName, "testRayTriangleIntersectionKernel")

        rayOriginIn = [[0, 0, 1.5, 0],
                        [0, 1, 1.5, 0],
                        [1, 0, 1.5, 0],
                        [0.5, 0.5, 1.5, 0]]
        
        rayOriginOut = [[-1, 0, 1.5, 0],
                        [0, 2, 1.5, 0],
                        [1, 1, 1.5, 0],
                        [-1, -1, 1.5, 0]]
        
        rayDirectionIn = [[0, 0, 1, 0]*8]
        rayDirectionOut = [[0, 0, -1, 0]*8]

        rayOrigin = cp.array(rayOriginIn + rayOriginOut, dtype=cp.float32).reshape(8, 4)
        rayDirection = cp.array(rayDirectionOut, dtype=cp.float32).reshape(8, 4)

        triangle = cp.array([[0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], dtype=cp.float32).flatten()

        t = cp.zeros(8, dtype=cp.float32)

        self.rayTriangleIntersection((1, 1), (8, 1), [triangle, rayOrigin, rayDirection, t, 8])
        print (t.get())
        

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
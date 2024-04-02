import numpy as np
import cupy as cp

morton_t = np.uint32

class BVHTreeDelegate(object):
    def __init__(self, nbTriangles : int) -> None:
        self.nbKeys = nbTriangles
        self.nbNodes = 2 * nbTriangles - 1
        self.size = 0
        self.root = None
        self.treeStructure = None

class AoSTree(BVHTreeDelegate):
    def __init__(self, nbTriangles : int) -> None:
        super().__init__(nbTriangles)
        self.number_of_triangles = nbTriangles
        self.nb_leaves = nbTriangles
        self.nb_nodes = 2 * nbTriangles - 1
        self.nb_internal_nodes = self.nb_leaves - 1 
class SoATree(BVHTreeDelegate):
    def __init__(self, nbTriangles : int, deviceTreeStructurePtr : int) -> None:
        super().__init__(nbTriangles)
        self.deviceTreeStructurePtr = deviceTreeStructurePtr
    
        self.child_left = cp.zeros(self.nbKeys, dtype=cp.uint32)
        self.child_right = cp.zeros(self.nbKeys, dtype=cp.uint32)
        self.left_range = cp.zeros(self.nbKeys, dtype=cp.uint32)
        self.right_range = cp.zeros(self.nbKeys, dtype=cp.uint32)
        self.enteredInternal = cp.zeros(self.nbKeys, dtype=cp.uint32)
        self.enteredLeaf = cp.zeros(self.nbKeys, dtype=cp.uint32)
        self.ids = cp.zeros(self.nbKeys, dtype=cp.int32)
        self.bboxMinInternal = cp.zeros((self.nbKeys, 4), dtype=cp.float32)
        self.bboxMaxInternal = cp.zeros((self.nbKeys, 4), dtype=cp.float32)
        self.bboxMinLeaf = cp.zeros((self.nbKeys, 4), dtype=cp.float32)
        self.bboxMaxLeaf = cp.zeros((self.nbKeys, 4), dtype=cp.float32)
        self.mortonCodes = cp.zeros(self.nbKeys, dtype=morton_t)
        self.sortedIndices = cp.zeros(self.nbKeys, dtype=cp.uint32)
    
    def getRoot(self):
        # if self.root is None:
        #     self.root = self.child_left[self.nbKeys - 1]
        return self.root
    
    def getStructure(self):
        self.treeStructure = {
            "child_left" : self.child_left,
            "child_right" : self.child_right,
            "left_range" : self.left_range,
            "right_range" : self.right_range,
            "entered" : self.enteredInternal,
            "ids" : self.ids,
            "bboxMinInternal" : self.bboxMinInternal,
            "bboxMaxInternal" : self.bboxMaxInternal,
            "bboxMinLeaf" : self.bboxMinLeaf,
            "bboxMaxLeaf" : self.bboxMaxLeaf,
            "mortonCodes" : self.mortonCodes
        }
        return self.treeStructure
    
    def printStructure(self):
        print ("SoATree structure:")
        print("\t>child_left: ", self.child_left)
        print("\t>child_right: ", self.child_right)
        print("\t>left_range: ", self.left_range)
        print("\t>right_range: ", self.right_range)
        print("\t>entered: ", self.enteredInternal)
        print("\t>ids: ", self.ids)
        print("\t>bboxMinInternal: ", self.bboxMinInternal)
        print("\t>bboxMaxInternal: ", self.bboxMaxInternal)
        print("\t>bboxMinLeaf: ", self.bboxMinLeaf)
        print("\t>bboxMaxLeaf: ", self.bboxMaxLeaf)
        print("\t>mortonCodes: ", self.mortonCodes)


        
# class AoSTree(BVHTreeDelegate):
#     def __init__(self, nbTriangles : int) -> None:
#         super().__init__(nbTriangles)
#         self.nodes = []
#         self.root = None

class BVHTree:
    def __init__(self, _BVHTreeDelegate : BVHTreeDelegate) -> None:
        self._delegate = _BVHTreeDelegate

    def __getattr__(self, name):
        return getattr(self._delegate, name)
    

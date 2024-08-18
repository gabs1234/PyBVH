import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from CudaPipeline import CudaPipeline
from MeshGenerator import GenerateRandomPoints
from MeshReader import MeshReader, StlReader
from CudaTimer import CudaTimer

import pandas as pd

pipeline = CudaPipeline(["/home/lt0649/Dev/PyBVH/src/"])

source_files = ["tree_prokopenko.cu", 
                "Ray.cu",
                "compare_bvh_previous.cu"]

module_name = "compare"
pipeline.readModuleFromFiles(source_files, module_name, backend="nvcc",
                             options=[],  jitify=False)


# def sort_triangles_and_vertices(triangles):
#     """Sort triangles based on the greatest x-coordinate in an ascending order. Also sort
#     vertices inside the triangles so that the greatest one is the last one, however, the
#     position of the two remaining ones is not sorted.
#     """
#     nb_triangles = int(triangles.shape[0] / 3)
#     # Extract x-coordinates of the last vertex of each triangle
#     x_coords = triangles[:, 0]
    
#     # Get indices which sort the triangles based on the greatest x-coordinate
#     sort_indices = cp.argsort(x_coords)
    
#     # Create a new array with sorted triangles
#     sorted_triangles = triangles[sort_indices]
    
#     # For each triangle, sort the vertices so that the greatest x-coordinate is the last vertex
#     for i in range(nb_triangles):
#         # Find the index of the greatest x-coordinate among the vertices of the current triangle
#         ti = i * 3
#         x_values = sorted_triangles[ti:ti+3, 0].get()
#         # print (x_values)
#         max_x_index = np.argmax(x_values)
#         # Swap the greatest x-coordinate vertex to the last position
#         sorted_triangles[ti+2], sorted_triangles[ti+max_x_index] = sorted_triangles[ti+max_x_index], sorted_triangles[ti+2]
    
#     return sorted_triangles

def testPerformance(nb_keys, vertices, bbMin, bbMax, bbMinLeafs, bbMaxLeafs):
    timer = CudaTimer(cp.cuda.get_current_stream())
    treeSize = cp.zeros(1, dtype=cp.uint32)

    getSize = pipeline.getKernelFromModule(module_name, "getTreeClassSize")
    getSize((1, ), (1, ), [treeSize])

    # Allocate tree arrays
    treeClassPtr = cp.cuda.alloc(treeSize.get()[0])



    # Initialize tree
    mortonKeys = cp.zeros(nb_keys, dtype=cp.uint32)
    sorted_indices = cp.arange(nb_keys, dtype=cp.uint32)
    bbMinInternal = cp.zeros((nb_keys, 4), dtype=cp.float32)
    bbMaxInternal = cp.zeros((nb_keys, 4), dtype=cp.float32)
    left_range = cp.zeros(nb_keys, dtype=cp.int32)
    right_range = cp.zeros(nb_keys, dtype=cp.int32)
    left_child = cp.ones(nb_keys, dtype=cp.int32) * -1
    right_child = cp.zeros(nb_keys, dtype=cp.int32)
    entered = cp.ones(nb_keys, dtype=cp.int32) * -1
    rope_leafs = cp.ones(nb_keys, dtype=cp.int32) * -1
    rope_internals = cp.ones(nb_keys, dtype=cp.int32) * -1
    
    
    initTree = pipeline.getKernelFromModule(module_name, "initializeTreeProkopenko")

    args = [treeClassPtr, nb_keys, mortonKeys, sorted_indices,
            bbMin, bbMax, 
            bbMinLeafs, bbMaxLeafs,
            bbMinInternal, bbMaxInternal,
            left_range, right_range, left_child, right_child,
            entered, rope_leafs, rope_internals]
    
    initTree((1, ), (1, ), args)



    # Compute the morton codes
    computeMortonCodes = pipeline.getKernelFromModule(module_name, "projectKeysProkopenko")
    blockSize = (256, 1, 1)
    gridSize = (int(np.ceil(nb_keys/blockSize[0])), 1, 1)
    args = [treeClassPtr, nb_keys]
    timer.start()
    computeMortonCodes(gridSize, blockSize, args)
    timer.stop()
    projectionTime = timer.elapsedTime()
    timer.reset()

    # Sort the morton codes
    sorted = cp.argsort(mortonKeys)
    sorted_indices[:] = sorted_indices[sorted]
    mortonKeys[:] = mortonKeys[sorted]
    bbMinLeafs[:] = bbMinLeafs[sorted]
    bbMaxLeafs[:] = bbMaxLeafs[sorted]

    # Build tree
    buildTree = pipeline.getKernelFromModule(module_name, "buildTreeProkopenko")
    blockSize = (256, 1, 1)
    gridSize = (int(np.ceil(nb_keys/blockSize[0])), 1, 1)
    args = [treeClassPtr, nb_keys]
    timer.start()
    buildTree(gridSize, blockSize, args)
    timer.stop()
    buildTime = timer.elapsedTime()
    
    # print (len(vertices))

    


    # # Test getting candidates
    # sortedTriangles = sort_triangles_and_vertices(vertices)
    # findCandidatesOld = pipeline.getKernelFromModule(module_name, "findCandidatesOld")
    # nbCandidatesOld = cp.zeros(1, dtype=cp.uint32)
    # blockSize = (1,)
    # gridSize = (1,)
    # args = [vertices, len(vertices), nbCandidatesOld]
    
    # findCandidatesOld(gridSize, blockSize, args)
    
    # print (nbCandidates)
    # print(nbCandidatesOld)

    # Test getting candidates
    findCandidates = pipeline.getKernelFromModule(module_name, "findCandidates")
    nbCandidates = cp.zeros(1, dtype=cp.uint32)
    blockSize = (1,)
    gridSize = (1,)
    args = [treeClassPtr, nb_keys, nbCandidates]
    timer.start()
    findCandidates(gridSize, blockSize, args)
    timer.stop()
    findCandidatesTime = timer.elapsedTime()

    # print (nbCandidates)
    # print (f"Time: {timer.elapsedTime()} ms")
    return [projectionTime, buildTime, findCandidatesTime]


def buildFromRandom(nb_keys):
    nb_points = nb_keys * 3
    meshGenerator = GenerateRandomPoints(nb_points)
    meshGenerator.build()
    points = meshGenerator.vertices

    bbMinKeys = np.empty((nb_keys, 4), dtype=np.float32)
    bbMaxKeys = np.empty((nb_keys, 4), dtype=np.float32)

    # Get AABB for each triangle
    for i in range(nb_keys):
        triangle = points[i*3:i*3+3, :]
        bbMin = cp.min(triangle, axis=0)
        bbMax = cp.max(triangle, axis=0)

        bbMinKeys[i, :] = bbMin
        bbMaxKeys[i, :] = bbMax


    bbMinLeafs = cp.array(bbMinKeys, dtype=cp.float32)
    bbMaxLeafs = cp.array(bbMaxKeys, dtype=cp.float32)

    bbMin = cp.min(bbMinLeafs, axis=0)
    bbMax = cp.max(bbMaxLeafs, axis=0)

    vertices = cp.array(points, dtype=cp.float32)

    return testPerformance(nb_keys, vertices, bbMin, bbMax, bbMinLeafs, bbMaxLeafs)

if __name__ == "__main__":
    # nb_keys = 10000
    # buildFromRandom(nb_keys)
    
    range_keys = [10**i for i in range(1, 8)]
    range_keys.insert(0, 10)
    buildTimes = []
    projectionTimes = []
    findCandidatesTimes = []
    nb_candidates = []
    for nb_keys in range_keys:
        build, project, candidates = buildFromRandom(nb_keys)
        buildTimes.append(build)
        projectionTimes.append(project)
        findCandidatesTimes.append(candidates)

    # Save data to csv
    data = {
        "Number of keys": range_keys,
        "Morton codes (ms)": projectionTimes,
        "Tree build times (ms)": buildTimes,
        "Tree descent (ms)": findCandidatesTimes
    }
    df = pd.DataFrame(data)
    df.to_csv("data.csv", index=False)

    # nb_triangles = 8
    # bbMin = cp.array([0, 0, 0, 0], dtype=cp.float32)
    # bbMax = cp.array([1, 1, 1, 1], dtype=cp.float32)
    # bbMinLeafs = cp.zeros((nb_triangles, 4), dtype=cp.float32)
    # bbMaxLeafs = cp.zeros((nb_triangles, 4), dtype=cp.float32)
    # run_tree(nb_triangles, vertices, bbMin, bbMax, bbMinLeafs, bbMaxLeafs)
   
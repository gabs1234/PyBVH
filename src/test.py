import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from CudaPipeline import CudaPipeline
from MeshGenerator import GenerateRandomPoints
from MeshReader import MeshReader, StlReader
from CudaTimer import CudaTimer

# "/usr/lib/gcc/x86_64-pc-linux-gnu/12.3.0/",
# "/usr/lib/gcc/x86_64-pc-linux-gnu/12.3.0/include/c++",
# "/usr/lib/gcc/x86_64-pc-linux-gnu/12.3.0/include/c++/bits/",
# "/usr/lib/gcc/x86_64-pc-linux-gnu/12.3.0/include/c++/backward/",
# "/usr/lib/gcc/x86_64-pc-linux-gnu/12.3.0/include/c++/experimental/",
# "/usr/lib/gcc/x86_64-pc-linux-gnu/12.3.0/include/c++/ext/",
# "/usr/lib/gcc/x86_64-pc-linux-gnu/12.3.0/include/c++/decimal/",
# "/usr/lib/gcc/x86_64-pc-linux-gnu/12.3.0/include/c++/debug/",
# "/usr/lib/gcc/x86_64-pc-linux-gnu/12.3.0/include/c++/parallel/",
# "/usr/lib/gcc/x86_64-pc-linux-gnu/12.3.0/include/c++/pstl/",
# "/usr/lib/gcc/x86_64-pc-linux-gnu/12.3.0/include/c++/tr1/",
# "/usr/lib/gcc/x86_64-pc-linux-gnu/12.3.0/include/c++/tr2/",
# "/usr/lib/gcc/x86_64-pc-linux-gnu/12.3.0/include/c++/x86_64-pc-linux-gnu/",
# "/usr/lib/gcc/x86_64-pc-linux-gnu/12.3.0/include-fixed"

import time

IS_LEAF = 2147483648

keys = [0b00001,0b000010, 0b00100, 0b00101, 0b10011, 0b11000, 0b11001, 0b11110]

pipeline = CudaPipeline(["/home/lt0649/Dev/PyBVH/src/"])

source_files = ["tree_prokopenko.cu", "Ray.cu", "RayTracer.cu", "test.cu"]
module_name = "testy"
pipeline.readModuleFromFiles(source_files, module_name, backend="nvcc",
                             options=[],  jitify=False)

def plot_3d_bounding_box(ax, dimensions, position, color='black'):
    """
    Plot a 3D bounding box.

    Parameters:
    - ax: Axes3D object
    - dimensions: tuple of (width, height, depth)
    - position: tuple of (x, y, z)
    """
    # Define the vertices of the bounding box
    vertices = [
        [position[0], position[1], position[2]],
        [position[0] + dimensions[0], position[1], position[2]],
        [position[0] + dimensions[0], position[1] + dimensions[1], position[2]],
        [position[0], position[1] + dimensions[1], position[2]],
        [position[0], position[1], position[2] + dimensions[2]],
        [position[0] + dimensions[0], position[1], position[2] + dimensions[2]],
        [position[0] + dimensions[0], position[1] + dimensions[1], position[2] + dimensions[2]],
        [position[0], position[1] + dimensions[1], position[2] + dimensions[2]]
    ]

    # Plot lines between the vertices
    for i in range(4):
        ax.plot([vertices[i][0], vertices[i+4][0]], [vertices[i][1], vertices[i+4][1]], [vertices[i][2], vertices[i+4][2]], 'k-', color=color)
        ax.plot([vertices[i+4][0], vertices[(i+1)%4+4][0]], [vertices[i+4][1], vertices[(i+1)%4+4][1]], [vertices[i+4][2], vertices[(i+1)%4+4][2]], 'k-', color=color)
        ax.plot([vertices[i][0], vertices[(i+1)%4][0]], [vertices[i][1], vertices[(i+1)%4][1]], [vertices[i][2], vertices[(i+1)%4][2]], 'k-', color=color)


def count_unique(arr, nb_keys):
    cleaned = cp.where(0 <= arr)
    cleaned = cp.where(arr < nb_keys)
    unique = cp.unique(cleaned)
    return len(unique)

        

def run_tree(nb_keys, vertices, bbMin, bbMax, bbMinLeafs, bbMaxLeafs):
    treeSize = cp.zeros(1, dtype=cp.uint32)

    getSize = pipeline.getKernelFromModule(module_name, "getTreeClassSize")
    getSize((1, ), (1, ), [treeSize])

    # Allocate tree arrays
    treeClassPtr = cp.cuda.alloc(treeSize.get()[0])

    nb_nodes = 2 * nb_keys - 1

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
    computeMortonCodes(gridSize, blockSize, args)

    # Sort the morton codes
    sorted = cp.argsort(mortonKeys)
    sorted_indices[:] = sorted_indices[sorted]
    mortonKeys[:] = mortonKeys[sorted]
    bbMinLeafs[:] = bbMinLeafs[sorted]
    bbMaxLeafs[:] = bbMaxLeafs[sorted]

    # # print the tree
    # printTree = pipeline.getKernelFromModule(module_name, "printTreeProkopenko")
    # printTree((1, ), (1, ), [treeClassPtr])
    
    # Build tree
    buildTree = pipeline.getKernelFromModule(module_name, "buildTreeProkopenko")
    blockSize = (256, 1, 1)
    gridSize = (int(np.ceil(nb_keys/blockSize[0])), 1, 1)
    args = [treeClassPtr, nb_keys]
    buildTree(gridSize, blockSize, args)

    # print (right_child)
    # pause for a sec
    print (len(vertices))
    # print (left_child)
    # print (rope_internals)
    # print (rope_leafs)

    if cp.any(left_child > 2 * nb_keys):
        print ("Error")
    if cp.any(rope_leafs > 2 * nb_keys):
        print ("Error")
    if cp.any(rope_internals > 2 * nb_keys):
        print ("Error")

    # # Test ray box intersection testRayBoxIntersection
    # rayBoxIntersection = pipeline.getKernelFromModule(module_name, "testRayBoxIntersection")
    # rayBoxIntersection((1, ), (1, ), [])

    # # Test ray triangle intersection testRayTriangleItersection
    # rayBoxIntersection = pipeline.getKernelFromModule(module_name, "testRayTriangleIntersection")
    # rayBoxIntersection((1, ), (1, ), [])

    # Count unique occurrences of numbers in the three arrays using set
    # filtered = left_child + rope_internals + rope_leafs
    # filtered = cp.where(0 <= filtered)[0]
    # filtered = cp.where(filtered < nb_keys)[0].get()
    # nb_leafs = len(set(filtered))
    # print (nb_leafs)

    # Get ray tracer size
    tracerSize = cp.zeros(1, dtype=cp.uint32)
    rayTracerSize = pipeline.getKernelFromModule(module_name, "getRayTracerSize")
    rayTracerSize((1, ), (1, ), [tracerSize])

    rayTracerPtr = cp.cuda.alloc(tracerSize.get()[0])

    print (tracerSize.get()[0])

    # Initialize ray tracer
    initRayTracer = pipeline.getKernelFromModule(module_name, "initializeRayTracer")
    parallel = True
    args = [rayTracerPtr, treeClassPtr, vertices, len(vertices), parallel]
    initRayTracer((1, ), (1, ), args)

    # fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    # cmap = plt.get_cmap("tab10")
    # for i in range(nb_keys-1):
    #     dim = bbMaxInternal[i].get() - bbMinInternal[i].get()
    #     pos = bbMinInternal[i].get()[0:3]
    #     plot_3d_bounding_box(ax, dim, pos, color=cmap(i))
    # for i in range(nb_keys):
    #     dim = bbMaxLeafs[i].get() - bbMinLeafs[i].get()
    #     pos = bbMinLeafs[i].get()[0:3]
    #     plot_3d_bounding_box(ax, dim, pos, color='red')
    
    # plt.show()

    # Trace rays
    print (vertices)
    rayTrace = pipeline.getKernelFromModule(module_name, "projectPlaneRays")
    nx = 4000
    ny = 4000
    size_x = cp.float32(3.5)
    size_y = cp.float32(3.5)
    theta = cp.float32(0)
    phi = cp.float32(0)
    r = cp.float32(10)
    origin = cp.array([0,0,0,0], dtype=cp.float32)
    image = cp.zeros(nx * ny, dtype=cp.float32)
    blockSize = (16, 16, 1)
    gridSize = (int(np.ceil(nx/blockSize[0])), int(np.ceil(ny/blockSize[1])), 1)
    args = (rayTracerPtr, image, origin, nx, ny, size_x, size_y, theta, phi, r)

    timer = CudaTimer(cp.cuda.get_current_stream())
    timer.start()
    rayTrace(gridSize, blockSize, args)
    timer.stop()

    print (f"Time: {timer.elapsedTime()} ms")

    # #
    # rayTrace = pipeline.getKernelFromModule(module_name, "projectPlaneRaysSimple")
    # nx = 10
    # ny = 10
    # size_x = cp.float32(2)
    # size_y = cp.float32(2)
    # image = cp.zeros(nx * ny, dtype=cp.float32)
    # blockSize = (16, 16, 1)
    # gridSize = (int(np.ceil(nx/blockSize[0])), int(np.ceil(ny/blockSize[1])), 1)
    # args = (treeClassPtr, image, vertices, nx, ny, size_x, size_y)

    # rayTrace(gridSize, blockSize, args)

    # Show image
    host_image = image.get()
    host_image = host_image.reshape((ny, nx))
    plt.imshow(host_image)
    plt.show()

def testPerformance(nb_keys, vertices, bbMin, bbMax, bbMinLeafs, bbMaxLeafs):
    treeSize = cp.zeros(1, dtype=cp.uint32)

    getSize = pipeline.getKernelFromModule(module_name, "getTreeClassSize")
    getSize((1, ), (1, ), [treeSize])

    # Allocate tree arrays
    treeClassPtr = cp.cuda.alloc(treeSize.get()[0])

    nb_nodes = 2 * nb_keys - 1

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
    computeMortonCodes(gridSize, blockSize, args)

    # Sort the morton codes
    sorted = cp.argsort(mortonKeys)
    sorted_indices[:] = sorted_indices[sorted]
    mortonKeys[:] = mortonKeys[sorted]
    bbMinLeafs[:] = bbMinLeafs[sorted]
    bbMaxLeafs[:] = bbMaxLeafs[sorted]

    # # print the tree
    # printTree = pipeline.getKernelFromModule(module_name, "printTreeProkopenko")
    # printTree((1, ), (1, ), [treeClassPtr])
    
    # Build tree
    buildTree = pipeline.getKernelFromModule(module_name, "buildTreeProkopenko")
    blockSize = (256, 1, 1)
    gridSize = (int(np.ceil(nb_keys/blockSize[0])), 1, 1)
    args = [treeClassPtr, nb_keys]
    buildTree(gridSize, blockSize, args)
    
    print (len(vertices))

    # # Test ray box intersection testRayBoxIntersection
    # rayBoxIntersection = pipeline.getKernelFromModule(module_name, "testRayBoxIntersection")
    # rayBoxIntersection((1, ), (1, ), [])

    # # Test ray triangle intersection testRayTriangleItersection
    # rayBoxIntersection = pipeline.getKernelFromModule(module_name, "testRayTriangleIntersection")
    # rayBoxIntersection((1, ), (1, ), [])

    # Count unique occurrences of numbers in the three arrays using set
    # filtered = left_child + rope_internals + rope_leafs
    # filtered = cp.where(0 <= filtered)[0]
    # filtered = cp.where(filtered < nb_keys)[0].get()
    # nb_leafs = len(set(filtered))
    # print (nb_leafs)

    # Get ray tracer size
    tracerSize = cp.zeros(1, dtype=cp.uint32)
    rayTracerSize = pipeline.getKernelFromModule(module_name, "getRayTracerSize")
    rayTracerSize((1, ), (1, ), [tracerSize])

    rayTracerPtr = cp.cuda.alloc(tracerSize.get()[0])

    print (tracerSize.get()[0])

    # Initialize ray tracer
    initRayTracer = pipeline.getKernelFromModule(module_name, "initializeRayTracer")
    parallel = True
    args = [rayTracerPtr, treeClassPtr, vertices, len(vertices), parallel]
    initRayTracer((1, ), (1, ), args)

    # fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    # cmap = plt.get_cmap("tab10")
    # for i in range(nb_keys-1):
    #     dim = bbMaxInternal[i].get() - bbMinInternal[i].get()
    #     pos = bbMinInternal[i].get()[0:3]
    #     plot_3d_bounding_box(ax, dim, pos, color=cmap(i))
    # for i in range(nb_keys):
    #     dim = bbMaxLeafs[i].get() - bbMinLeafs[i].get()
    #     pos = bbMinLeafs[i].get()[0:3]
    #     plot_3d_bounding_box(ax, dim, pos, color='red')
    
    # plt.show()

    # Trace rays
    print (vertices)
    rayTrace = pipeline.getKernelFromModule(module_name, "projectPlaneRays")
    nx = 2000
    ny = 2000
    size_x = cp.float32(10)
    size_y = cp.float32(10)
    theta = cp.float32(0)
    phi = cp.float32(np.pi/2)
    r = cp.float32(1)
    origin = cp.array([0,0,0,0], dtype=cp.float32)
    image = cp.zeros(nx * ny, dtype=cp.float32)
    blockSize = (16, 16, 1)
    gridSize = (int(np.ceil(nx/blockSize[0])), int(np.ceil(ny/blockSize[1])), 1)
    args = (rayTracerPtr, image, origin, nx, ny, size_x, size_y, theta, phi, r)

    timer = CudaTimer(cp.cuda.get_current_stream())
    timer.start()
    rayTrace(gridSize, blockSize, args)
    timer.stop()

    print (f"Time: {timer.elapsedTime()} ms")


    
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

    run_tree(nb_keys, vertices, bbMin, bbMax, bbMinLeafs, bbMaxLeafs)

def buildFromStl(file_path):
    reader = MeshReader(StlReader(file_path))

    points = reader.vertices
    nb_points = len(points)
    nb_triangles = nb_points // 3

    bbMinKeys = np.empty((nb_triangles, 4), dtype=np.float32)
    bbMaxKeys = np.empty((nb_triangles, 4), dtype=np.float32)

    for i in range(nb_triangles):
        triangle = points[i*3:i*3+3, :]
        bbMin = np.min(triangle, axis=0)
        bbMax = np.max(triangle, axis=0)

        bbMinKeys[i, :] = bbMin
        bbMaxKeys[i, :] = bbMax

    bbMinLeafs = cp.array(bbMinKeys, dtype=cp.float32)
    bbMaxLeafs = cp.array(bbMaxKeys, dtype=cp.float32)

    # print (bbMinLeafs)
    # print (bbMaxLeafs)

    bbMin = cp.min(bbMinLeafs, axis=0)
    bbMax = cp.max(bbMaxLeafs, axis=0)

    vertices = cp.array(points, dtype=cp.float32)
    
    run_tree(nb_triangles, vertices, bbMin, bbMax, bbMinLeafs, bbMaxLeafs)


if __name__ == "__main__":
    # nk_keys = 32
    # buildFromRandom(nk_keys)

    buildFromStl("blender/monkey.stl")

    # nb_triangles = 8
    # bbMin = cp.array([0, 0, 0, 0], dtype=cp.float32)
    # bbMax = cp.array([1, 1, 1, 1], dtype=cp.float32)
    # bbMinLeafs = cp.zeros((nb_triangles, 4), dtype=cp.float32)
    # bbMaxLeafs = cp.zeros((nb_triangles, 4), dtype=cp.float32)
    # run_tree(nb_triangles, vertices, bbMin, bbMax, bbMinLeafs, bbMaxLeafs)
   
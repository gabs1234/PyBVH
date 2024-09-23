import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from CudaPipeline import CudaPipeline
from MeshGenerator import GenerateRandomPoints
from MeshReader import MeshReader, StlReader
from CudaTimer import CudaTimer
import time

pipeline = CudaPipeline(["/home/lt0649/Dev/PyBVH/src/"])

source_files = [
    "Quaternion.cu",
    "RotationQuaternion.cu",
    "Basis.cu",
    "Ray.cu",
    "tree_prokopenko.cu",
    "RayTracer.cu",
    "SceneManager.cu",
]

module_name = "test"
pipeline.readModuleFromFiles(module_name, source_files, backend="nvrtc",
                             options=[],  jitify=True)

def plot_3d_bounding_box(ax, dimensions, position, color='black'):
    """
    Plot a 3D bounding bo
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

def SphericalToCartesian(theta, phi, r):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z, 0 ])
        

def run_tree(nb_keys, vertices, bbMin, bbMax, bbMinLeafs, bbMaxLeafs):
    projectKernel = pipeline.getKernelFromModule(module_name, "projectKeys")
   
    
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
   
import cupy, cupyx
from CudaPipeline import CudaPipeline
import numpy as np
import pyvista as pv
import graphviz
import matplotlib.pyplot as plt
from CudaTimer import CudaTimer
import argparse
from Basis import CameraBasis, Basis

float4 = np.dtype(
    {
        'names': ['x', 'y', 'z', 'w'],
        'formats': [np.float32] * 4,
    }
)

float2 = np.dtype(
    {
        'names': ['x', 'y'],
        'formats': [np.float32] * 2,
    }
)

uint2 = np.dtype(
    {
        'names': ['x', 'y'],
        'formats': [np.uint32] * 2,
    }
)

def make_random_points(n):
    return cupy.random.rand(n, 3).astype(cupy.float32)

def make_triangle_from_points(points, eps, seed=123456):
    '''
    use points as center of triangle,
    '''
    cupy.random.seed(seed)
    n = len(points)
    
    # Create an array of shape (n, 3) for the triangle vertices
    random_offsets = cupy.random.rand(n, 3, 3) * eps  # Random offsets for A, B, C
    triangles = cupy.empty((n * 3, 4), dtype=cupy.float32)
    
    # Repeat points for each triangle vertex
    base_points = cupy.repeat(points[:, cupy.newaxis, :], 3, axis=1)  # Shape (n, 3, 3)
    
    # Add random offsets to the base points
    triangles[:, 0:3] = cupy.reshape(base_points + random_offsets, (n * 3, 3))
    
    return triangles

headers = ("/home/lt0649/Dev/PyBVH/src/",)

pipeline = CudaPipeline (headers)
pipeline.readModuleFromFiles("buildTree", ["Ray.cu", "source.cu"], jitify=False)

timer = CudaTimer(cupy.cuda.Stream.null)

module = pipeline.modules["buildTree"]
# take n as argument
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=10)
args = parser.parse_args()

n = args.n
extents = .5
points = make_random_points(n) * extents
triangles = make_triangle_from_points(points, 0.1)
sceneMin = np.array([0, 0, 0, 0], dtype=np.float32)
sceneMax = np.array([extents, extents, extents, 0], dtype=np.float32)

keys = cupy.zeros(n, dtype=cupy.uint32)
rope = cupy.ones(2 * n, dtype=cupy.int32) * -1
left = cupy.ones(2 * n, dtype=cupy.int32) * -1
entered = cupy.ones(2 * n, dtype=cupy.int32) * -1
bbMin = cupy.zeros((2 * n, 4), dtype=cupy.float32)
bbMax = cupy.zeros((2 * n, 4), dtype=cupy.float32)

# print ("Projecting triangle centroids")
projectKernel = module.get_function("projectTriangleCentroid")
args = (n, triangles, keys, bbMin, bbMax, sceneMin.view(float4), sceneMax.view(float4))
block_size = (256, 1, 1)
grid_size = (n // block_size[0] + 1, 1)
timer.start()
projectKernel(grid_size, block_size, args)
cupy.cuda.Stream.null.synchronize()
timer.stop()
project_t = timer.elapsedTime()
timer.reset()

# Sort the keys
# print ("Sorting keys")
sorted_keys = cupy.argsort(keys)
keys = keys[sorted_keys]
permutation = sorted_keys

# Grow the tree
# print ("Growing tree")
growTreeKernel = module.get_function("growTreeKernel")
args = (n, keys, permutation, rope, left, entered, bbMin, bbMax)
block_size = (256, 1, 1)
grid_size = (n // block_size[0] + 1, 1)
timer.start()
growTreeKernel(grid_size, block_size, args)
cupy.cuda.Stream.null.synchronize()
timer.stop()
grow_t = timer.elapsedTime()
timer.reset()


# Project plane rays
# print ("Projecting plane rays")
nx, ny = 2048, 2048
image_res = np.array([nx, ny], dtype=np.uint32)
image_spatial_extents = np.array([2, 2], dtype=np.float32) * extents
meshBasis = Basis ([1, 0, 0], [0, 1, 0], [0, 0, 1], [.25, .25, 0])

# Use spherical coordinates to define the camera basis
theta, phi, r = 0, 0, 10
# Use euler angles to rotate the camera view
yaw, pitch, roll = 0, 0, 0
cameraBasis = CameraBasis(r, theta, phi, meshBasis.getOrigin())
cameraBasis.rotate_euler(yaw, pitch, roll)
U, V, W = cameraBasis.getBaseVectors()
print (U, V, W)
print (cameraBasis.getOrigin().view(float4))
image = cupy.zeros(nx * ny, dtype=cupy.float32)
projectPlaneRaysKernel = module.get_function("projectPlaneRaysKernel")
args = (
    n, image, image_res.view(uint2), image_spatial_extents.view(float2),
    U.view(float4), V.view(float4), W.view(float4), cameraBasis.getOrigin().view(float4),
    rope, left, bbMin, bbMax, triangles)

# Use spherical coordinates to define the camera basis
# Use euler angles to rotate the camera view
theta, phi, r = -np.pi / 4, -np.pi / 4, 2
yaw, pitch, roll =  0, 0, 0
cameraBasis = CameraBasis(r, theta, phi, meshBasis.getOrigin())
cameraBasis.rotate_euler(yaw, pitch, roll)
U, V, W = cameraBasis.getBaseVectors()
# print (U, V, W)

image2 = cupy.zeros(nx * ny, dtype=cupy.float32)
args2 = (
    n, image2, image_res.view(uint2), image_spatial_extents.view(float2),
    U.view(float4), V.view(float4), W.view(float4), cameraBasis.getOrigin().view(float4),
    rope, left, bbMin, bbMax, triangles)

block_size = (16, 16, 1)
grid_size = (n // block_size[0] + 1, n // block_size[1] + 1, 1)
timer.start()
projectPlaneRaysKernel(grid_size, block_size, args)
cupy.cuda.Stream.null.synchronize()
timer.stop()
projectPlaneRays_t = timer.elapsedTime()
timer.reset()

projectPlaneRaysKernel(grid_size, block_size, args2)
cupy.cuda.Stream.null.synchronize()

# Print all timings in one go
print (grow_t, project_t, projectPlaneRays_t)

image = image.get()
image2 = image2.get()
plt.imshow(image.reshape(nx, ny))
plt.figure()
plt.imshow(image2.reshape(nx, ny))
plt.show()
S = -1

class Nodes:
    def __init__(self):
        self.nb_nodes = 0
        self.left = None
        self.right = None
        self.rope = None

def rope_to_right (internal, leaf, nb_nodes):
    current_node = nb_nodes
    prev_node = 0
    left_child, right_child = 0, 0

    l, r = [0]* 2 * nb_nodes, [0]* 2 * nb_nodes

    while current_node != S:
        # print("current_node", current_node)
        prev_node = current_node

        if current_node < nb_nodes: # is leaf
            left_child = current_node
            right_child = leaf.rope[current_node]

            if right_child < nb_nodes:
                current_node = leaf.rope[right_child]
            else:
                current_node = right_child   
        else:
            # get the left child
            left_child = internal.left[current_node - nb_nodes]
            # get the right child
            if left_child >= nb_nodes:
                right_child = internal.rope[left_child - nb_nodes]
            else:
                right_child = leaf.rope[left_child]

            current_node = left_child

        l[prev_node] = left_child
        r[prev_node] = right_child
    
    return l, r

def visualize_tree(leftNodes, rightNodes, nb_nodes):
    dot = graphviz.Digraph()
    parentNodes = [p for p in range(nb_nodes, 2*nb_nodes-1)]
    for parent in parentNodes:
        if leftNodes[parent] != S:
            # print (parent, leftNodes[parent])
            dot.edge(str(parent), str(leftNodes[parent]))
        if rightNodes[parent] != S:
            # print (parent, rightNodes[parent])
            dot.edge(str(parent), str(rightNodes[parent]))
    dot.render('tree.png', view=True)

def visualize_internal (plotter, bbmin, bbmax, scene_area, theme='gist_rainbow', fill=True):
    cmap = plt.get_cmap(theme)
    num_boxes = len(bbmin)
    norm = plt.Normalize(0, num_boxes)

    if fill:
        color = lambda i: cmap(norm(i))
    else:
        color = lambda i: 'white'

    # find max height
    lengths = [bbmax[i][2] - bbmin[i][2] for i in range(num_boxes)]
    max_height = max(lengths)

    for i in range(num_boxes):
        zmin = bbmin[i][2] + i * max_height
        zmax = bbmax[i][2] + i * max_height
        box = pv.Box([bbmin[i][0], bbmax[i][0], bbmin[i][1], bbmax[i][1], zmin, zmax])
        plotter.add_mesh(box, color=color(i), opacity=0.5, edge_color=cmap(norm(i)), show_edges=True)

def visualize_leaf (plotter, bbmin, bbmax, scene_area, theme='gist_rainbow', fill=True):
    cmap = plt.get_cmap(theme)
    num_boxes = len(bbmin)
    norm = plt.Normalize(0, num_boxes)
    
    if fill:
        color = lambda i: cmap(norm(i))
    else:
        color = lambda i: 'white'
    
    for i in range(num_boxes):
        zmin = bbmin[i][2]
        zmax = bbmax[i][2]
        box = pv.Box([bbmin[i][0], bbmax[i][0], bbmin[i][1], bbmax[i][1], zmin, zmax])
        plotter.add_mesh(box, color=color(i), opacity=0.5, edge_color=cmap(norm(i)), show_edges=True)

        # Add label id
        label = pv.Label (str(i), position=(bbmin[i][0], bbmin[i][1], zmin), size=10)
        plotter.add_actor(label)

def count_leafs (leftChild, ropeLeaf):
    nb_nodes = len(leftChild)
    nb_leafs = 0
    current_node = leftChild[0]
    while current_node != S:
        if current_node < nb_nodes:
            nb_leafs += 1
            current_node = ropeLeaf[current_node]
        else:
            current_node = leftChild[current_node - nb_nodes]
    
    return nb_leafs


internalNodes = Nodes()
internalNodes.nb_nodes = n - 1
internalNodes.left = left[n:2*n].get()
internalNodes.rope = rope[n:2*n].get()
leafNodes = Nodes()
leafNodes.nb_nodes = n
leafNodes.rope = rope[:n].get()

l, r = rope_to_right(internalNodes, leafNodes, leafNodes.nb_nodes)
# visualize_tree(l, r, leafNodes.nb_nodes)

# print(count_leafs(left_child, rope_leaf) == leafNodes.nb_nodes)
# print(l, r)

# 3d visualization
plotter = pv.Plotter()
internal_bbMin = bbMin[n:2*n].get()
internal_bbMax = bbMax[n:2*n].get()
leaf_bbMin = bbMin[:n].get()
leaf_bbMax = bbMax[:n].get()

# visualize_internal(plotter, internal_bbMin, internal_bbMax, 5, fill=True)
# visualize_leaf(plotter, leaf_bbMin, leaf_bbMax, 5, fill=True)

# plotter.show()
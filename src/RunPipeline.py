from deprecated.Optimizer import *
from Builder import *
from deprecated.Collapser import *
from MeshReader import *
from SceneManager import *
from BVHTree import BVHTree
from CudaPipeline import CudaPipeline
from deprecated.Render import Render
from MeshGenerator import GenerateRandomPoints
# Measure time
import time


def test_run():
	n = (100, 100, 100)
	x = y = z = (-1, 1)

	pipeline = CudaPipeline([
		"/home/lt0649/Dev/PyBVH/src", 
		"/opt/cuda/include/"
		])
	
	# # Measure time
	start = time.time()
	# meshReader = MeshReader(StlReader("triangle2.stl"))
	# meshReader = MeshReader(StlReader("cube.stl"))
	# meshReader = MeshReader(StlReader("monkey.stl"))
	# # meshReader = MeshReader(StlReader("255.stl"))
	# meshReader = MeshReader(StlReader("big_monkey.stl"))
	# meshReader = MeshReader(StlReader("surface.stl"))
	# meshReader = MeshReader(StlReader("sphere.stl"))
	# meshReader = MeshReader(StlReader("phantom.stl"))
	meshReader = MeshReader(StlReader("torus.stl"))
	end = time.time()
	print ("MeshReader: ", end - start)

	# print ("Nb triangles: ", meshReader.nbTriangles)

	treeBuilder = Builder(LBVHBuilder(pipeline))
	vertices = meshReader.vertices
	nbTriangles = meshReader.nbTriangles

	sceneManager = SceneManager(pipeline, nbTriangles, vertices)

	treeBuilder.build(sceneManager.deviceScene)

	tree = treeBuilder.getTree()
	# treeBuilder.showLeafBoxes()
	# treeBuilder.showBoxesPerLevel(list(range(0, 10)))

	# print(tree.child_left)
	# print(tree.child_right)

	tree.root = tree.child_left[-1]
	print("root found", treeBuilder.rootIndex)
	print("implicit root", tree.child_left[-1])


	# optimizer = Optimizer(AgglomerativeTreeletOptimizer())
	# optimizer.optimize(tree)

	# collapser = Collapser(GPUCollapser())
	# collapser.collapse(tree)

	renderer = Render(pipeline, sceneManager.deviceScene, tree)
	levels = renderer.descendTree()
	renderer.projectParallelGeometry(1000, 1000)

	sum = 0
	for key, value in levels.items():
		print (key, ":", len(value))
		sum += len(value)
	print ("Total: ", sum)
	print ("Nb nodes: ", 2*nbTriangles-1)

	# renderer.showTreeGraph()

	if sum != 2*nbTriangles-1:
		print ("Error")

if __name__ == "__main__":
	test_run()
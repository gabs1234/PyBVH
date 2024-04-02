from Optimizer import *
from Builder import *
from Collapser import *
from MeshReader import *
from SceneManager import *
from BVHTree import BVHTree
from CudaPipeline import CudaPipeline
from Render import Render
from MeshGenerator import GenerateRandomPoints
# Measure time
import time

def find_close_floats(array, epsilon=1e-5):
	diffs = np.abs(array[:, None] - array)

	# Check if any difference is less than epsilon, excluding comparisons with the same element
	close_pairs_exist = np.any(np.isclose(diffs, 0, atol=epsilon) & (diffs > 0))

	return close_pairs_exist

def test_run():
	n = (100, 100, 100)
	x = y = z = (-1, 1)

	pipeline = CudaPipeline([
		"/home/lt0649/Dev/PyBVH/src", 
		"/opt/cuda/include/"
		])
	
	# # Measure time
	# start = time.time()
	# meshReader = MeshReader(StlReader("triangle2.stl"))
	# # meshReader = MeshReader(StlReader("cube.stl"))
	# # meshReader = MeshReader(StlReader("monkey.stl"))
	# # meshReader = MeshReader(StlReader("255.stl"))
	# # meshReader = MeshReader(StlReader("big_monkey.stl"))
	# # meshReader = MeshReader(StlReader("surface.stl"))
	# # meshReader = MeshReader(StlReader("sphere.stl"))
	# # meshReader = MeshReader(StlReader("phantom.stl"))
	# # meshReader = MeshReader(StlReader("torus.stl"))
	# end = time.time()
	# print ("MeshReader: ", end - start)

	# print ("Nb triangles: ", meshReader.nbTriangles)
	counter = 0
	max_iter = 100

	treeBuilder = Builder(LBVHBuilder(pipeline))

	nbTriangles = 80
	nbPoints = nbTriangles*3

	while True:
		meshGenerator = GenerateRandomPoints(nbPoints)
		meshGenerator.build()
		vertices = meshGenerator.vertices

		sceneManager = SceneManager(pipeline, nbTriangles, vertices)

		treeBuilder.build(sceneManager.deviceScene)

		tree = treeBuilder.getTree()

		tree.root = tree.child_left[-1]
		print("root found", treeBuilder.rootIndex)
		print("implicit root", tree.child_left[-1])

		# optimizer = Optimizer(AgglomerativeTreeletOptimizer())
		# optimizer.optimize(tree)

		# collapser = Collapser(GPUCollapser())
		# collapser.collapse(tree)

		renderer = Render(pipeline, sceneManager.deviceScene, tree)
		levels = renderer.descendTree()

		sum = 0
		for key, value in levels.items():
			print (key, ":", len(value))
			sum += len(value)
		print ("Total: ", sum)
		print ("Nb nodes: ", 2*nbTriangles-1)

		# renderer.showTreeGraph()

		if sum != 2*nbTriangles-1:
			print ("Error")
			break
		counter += 1
		if counter == max_iter:
			break

if __name__ == "__main__":
	test_run()
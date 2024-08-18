from Scene import Scene
from BVHTree import BVHTree
from CudaTimer import CudaTimer
from CudaPipeline import CudaPipeline
import cupy as cp
from collections import deque

import sys
import numpy as np

from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor
from PyQt5.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
import pyqtgraph as pg

import matplotlib.pyplot as plt

import graphviz

IS_LEAF = 2147483648

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


class ImageTracker(QWidget):
    def __init__(self, image, imageUpdater=None):
        super().__init__()

        self.image = image
        self.imageUpdater = imageUpdater

        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 500, 500)
        self.setWindowTitle('Image Tracker')

        # Create a QLabel to display the image
        self.label = QLabel(self)
        self.label.setGeometry(10, 10, self.image.shape[1], self.image.shape[0])

        self.updateImage()  # Update the initial image
        self.show()

    def updateImage(self):
        # Normalize the floating-point values to the range [0, 1]
        normalized_image = (self.image - np.min(self.image)) / (np.max(self.image) - np.min(self.image))

        # Map the normalized values to the range [0, 255]
        scaled_image = (normalized_image * 255).astype(np.uint8)

        # Convert the NumPy array to a QImage
        image = scaled_image.get()
        height, width = image.shape
        qimage = QImage(image.data, width, height, QImage.Format_Grayscale8)

        # Convert the QImage to a QPixmap and set it to the QLabel
        pixmap = QPixmap.fromImage(qimage)
        self.label.setPixmap(pixmap)

    def mouseMoveEvent(self, event):
        # Update the image where the mouse is moved
        x = event.x()
        y = event.y()

        # Update the image using the updater function
        self.image = self.imageUpdater(self.image, x, y)

        self.updateImage()

class Render:
    def __init__(self, pipeline : CudaPipeline, scene : Scene, tree : BVHTree, parallelGeometry=False):
        self._pipeline = pipeline
        self._scene = scene
        self._tree = tree
        self.rootIndex = self._tree.getRoot()
        self.parallelGeometry = parallelGeometry

        self.timer = CudaTimer(cp.cuda.get_current_stream())

        self.projector = self._pipeline.getKernelFromModule("LBVH", "projectRayGridKernel")

    def deviceProjectRayGrid(self, image : cp.ndarray, x, y) -> cp.ndarray:
        ny, nx  = image.shape[0], image.shape[1]

        start = self.timer.start()

        self.projector(self.gridSize, self.blockSize, [self._tree.deviceTreeStructurePtr, nx, ny, x, y, image])

        stop = self.timer.stop()

        print("Time to project ray grid: ", self.timer.elapsedTime())
        self.timer.reset()
        return image
    
    def projectParallelGeometry(self, nx, ny) -> cp.ndarray:
        self.parallelGeometry = True
        image = cp.zeros((ny, nx), dtype=cp.float32)

        self.blockSize = (32, 32, 1)
        self.gridSize = ((nx + self.blockSize[0] - 1) // self.blockSize[0] , (ny + self.blockSize[1] - 1) // self.blockSize[1], 1)

        image = self.deviceProjectRayGrid(image, 0, 0)

        plt.imshow(image.get(), cmap='copper')
        plt.show()

    def render(self, nx, ny):
        app = QApplication(sys.argv)
        
        # Set the block and grid sizes for the CUDA kernel     
        self.blockSize = (32, 32, 1)
        self.gridSize = ((nx + self.blockSize[0] - 1) // self.blockSize[0] , (ny + self.blockSize[1] - 1) // self.blockSize[1], 1)

        # Create a 2D array to store the image
        image = cp.zeros((ny, nx), dtype=cp.float32)

        # Create an ImageTracker to display the image
        tracker = ImageTracker(image, self.deviceProjectRayGrid)
        sys.exit(app.exec_())
    
    def showTreeGraph(self):
        root = self._tree.root
        print ("python", root)

        child_left = self._tree.child_left
        child_right = self._tree.child_right

        queue = deque([root])

        dot = graphviz.Digraph("bvh")
        dot.attr(dpi='300') # Set the DPI to 300

        while queue:
            node = queue.popleft()
            if node != -1:
                if node & IS_LEAF:
                    continue
                else:
                    # print (node)
                    parent = node
                    left = child_left[node]
                    right = child_right[node]
                    queue.append(left)
                    queue.append(right)
                    dot.node(str(parent), str(parent))
                    dot.node(str(left), str(left))
                    dot.node(str(right), str(right))
                    dot.edge(str(parent), str(left))
                    dot.edge(str(parent), str(right))               

            else:
                continue
        
        dot.render('graphviz/bvh.png',format='png')
            

    def descendTree(self) -> dict:
        root = self._tree.root
        # root = self._tree.child_left[-1]

        child_left = self._tree.child_left
        child_right = self._tree.child_right

        MAX_LEVEL = 30  # Maximum level to descend to

        level_nodes = {}  # Dictionary to store nodes per level
        queue = deque([(root, 0)])  # Initialize a queue with root node and its level


        while queue:
            v, level = queue.popleft()  # Dequeue a vertex and its level

            # Check if the level is within the maximum level
            if level >= MAX_LEVEL:
                print ("Maximum level reached")
                break

            if level not in level_nodes:
                level_nodes[level] = [v]  # Initialize list for the level
            else:
                level_nodes[level].append(v)  # Append node to the level's list

            # Check if v is a leaf node
            if v & IS_LEAF:
                continue  # Skip processing children if it's a leaf

            # Check the left child
            if child_left[v] is not None:
                queue.append((child_left[v], level + 1))

            # Check the right child
            if child_right[v] is not None:
                queue.append((child_right[v], level + 1))

        return level_nodes
    
    # def showBoxesPerLevel(self, levels=[]) -> None:
    #     self.level_nodes = self.descendTree()
    #     bboxMinLeaf = self.bboxMinLeaf.get()
    #     bboxMaxLeaf = self.bboxMaxLeaf.get()
    #     bboxMinInternal = self.bboxMinInternal.get()
    #     bboxMaxInternal = self.bboxMaxInternal.get()

    #     # Make a color map
    #     nbLevels = len(self.level_nodes)

    #     cmap = plt.get_cmap('turbo', nbLevels)

    #     # Plot the bounding boxes
    #     for level, nodes in self.level_nodes.items():
    #         if levels and level not in levels:
    #             continue

    #         color = cmap(level / nbLevels)

    #         for node in nodes:
    #             if self.isLeaf(node):
    #                 dimensions = (bboxMaxLeaf[node ^ IS_LEAF] - bboxMinLeaf[node ^ IS_LEAF])[0:3] 
    #                 position = bboxMinLeaf[node ^ IS_LEAF][0:3]
    #                 plot_3d_bounding_box(self.ax,dimensions, position, color=color)

    #             else:
    #                 dimensions = (bboxMaxInternal[node] - bboxMinInternal[node])[0:3]
    #                 position = bboxMinInternal[node][0:3]
    #                 plot_3d_bounding_box(self.ax,dimensions, position, color=color)

    #     # Set the aspect ratio to equal
    #     self.ax.set_box_aspect([1,1,1])

    #     # Display the plot
    #     plt.show()
    
    # def showLeafBoxes (self) -> None:
    #     bboxMinLeaf = self.bboxMinLeaf.get()
    #     bboxMaxLeaf = self.bboxMaxLeaf.get()

    #     for node in range(len(bboxMinLeaf)):
    #         dimensions = (bboxMaxLeaf[node] - bboxMinLeaf[node])[0:3]
    #         position = bboxMinLeaf[node][0:3]
    #         plot_3d_bounding_box(self.ax, dimensions, position)




import numpy as np
import cupy as cp
from dataclasses import dataclass


class SceneDelegate(object):
    numberOfTriangles = None
    vertices = None
    bboxMin = None
    bboxMax = None

@dataclass
class HostScene(SceneDelegate):
    numberOfTriangles : np.uint32
    vertices : np.ndarray
    bboxMin : np.ndarray
    bboxMax : np.ndarray

@dataclass
class DeviceScene(SceneDelegate):
    numberOfTriangles : cp.uint32
    vertices : cp.ndarray
    bboxMin : cp.ndarray
    bboxMax : cp.ndarray

@dataclass
class Scene():
    def __init__(self, _sceneDelegate : SceneDelegate):
        self._delegate = _sceneDelegate
    
    def __getattr__(self, name):
        return getattr(self._delegate, name)

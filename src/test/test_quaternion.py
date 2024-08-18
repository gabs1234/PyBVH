import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from CudaPipeline import CudaPipeline
from CudaTimer import CudaTimer

pipeline = CudaPipeline(["/home/lt0649/Dev/PyBVH/src/"])

source_files = ["Quaternion.cu", "RotationQuaternion.cu", "test_quaternion.cu"]
module_name = "test"
pipeline.readModuleFromFiles(source_files, module_name, backend="nvcc",
                             options=[],  jitify=False)

rotate = pipeline.getKernelFromModule(module_name, "rotateVector")

inputVector = cp.array([1.0, 0.0, 0.0, 0.0], dtype=cp.float32)
outputVector = cp.zeros(4, dtype=cp.float32)

angle = 2 *cp.pi 
axis = cp.array([0.0, 0.0, 1.0, 0], dtype=cp.float32)

rotate((1,), (1,), (inputVector, outputVector, axis, angle))

print (f"inputVector: {inputVector}")
print (f"outputVector: {outputVector}")

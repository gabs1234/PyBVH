import cupy as cp

good_raw_code = """
extern "C" __global__ void add() {
    return;
}
"""


bad_raw_code = """
#include <cub/cub.cuh>
#include "RayTracer.cuh"
#include "Ray.cuh"
#include "tree_prokopenko.cuh"
#include "Commons.cuh"
extern "C" __global__ void add() {
    return;
}
"""

block_size = (256, 1, 1)
grid_size = (1, 1, 1)

try:
    good_module = cp.RawModule(code=good_raw_code, backend="nvcc", options=('-D__CUDA_NO_HALF_CONVERSIONS__',))
    good_add = good_module.get_function("add")
    good_add(grid_size, block_size, ())
    print("Good code worked")
except Exception as e:
    print("Good code failed")
    print(e)

print ("-------------------")

try:
    bad_module = cp.RawModule(code=bad_raw_code, backend="nvcc", options=('-D__CUDA_NO_HALF_CONVERSIONS__',"-I/home/lt0649/Dev/PyBVH/src"))
    bad_add = bad_module.get_function("add")
    bad_add(grid_size, block_size, ())
    print("Bad code worked")
except Exception as e:
    print("Bad code failed")
    print(e)
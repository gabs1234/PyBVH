import cupy as cp
import numpy as np
from CudaTypes import make_float4

module = cp.RawModule(code = r'''
#include "Commons.cuh"
#include "Scene.h"
                      
#include <cstdio>
extern "C" {
// Calculate the centroid of the triangle AABB
__device__ float4 getTriangleCentroid(float4 vertex1, float4 vertex2, float4 vertex3)
{
    float4 boundingBoxMin;
    float4 boundingBoxMax;

    boundingBoxMin.x = min(vertex1.x, vertex2.x);
    boundingBoxMin.x = min(boundingBoxMin.x, vertex3.x);
    boundingBoxMax.x = max(vertex1.x, vertex2.x);
    boundingBoxMax.x = max(boundingBoxMax.x, vertex3.x);

    boundingBoxMin.y = min(vertex1.y, vertex2.y);
    boundingBoxMin.y = min(boundingBoxMin.y, vertex3.y);
    boundingBoxMax.y = max(vertex1.y, vertex2.y);
    boundingBoxMax.y = max(boundingBoxMax.y, vertex3.y);

    boundingBoxMin.z = min(vertex1.z, vertex2.z);
    boundingBoxMin.z = min(boundingBoxMin.z, vertex3.z);
    boundingBoxMax.z = max(vertex1.z, vertex2.z);
    boundingBoxMax.z = max(boundingBoxMax.z, vertex3.z);

    float4 centroid;
    centroid.x = (boundingBoxMax.x + boundingBoxMin.x) * 0.5f;
    centroid.y = (boundingBoxMax.y + boundingBoxMin.y) * 0.5f;
    centroid.z = (boundingBoxMax.z + boundingBoxMin.z) * 0.5f;

    return centroid;
}

__device__ float4 getBoundingBoxCentroid(float4 bboxMin, float4 bboxMax)
{
    float4 centroid;

    centroid.x = (bboxMin.x + bboxMax.x) / 2.0f;
    centroid.y = (bboxMin.y + bboxMax.y) / 2.0f;
    centroid.z = (bboxMin.z + bboxMax.z) / 2.0f;

    return centroid;
}
__inline__ void print_float4(float4 v) {
    printf("[%f %f %f %f] ", v.x, v.y, v.z, v.w);
}
__global__ void generateMortonCodesKernel3D(int numberOfTriangles, float4 *vertices, float4 bboxMin, float4 bboxMax, unsigned int *mortonCodes, unsigned int *sortIndices)
{
    const int globalId = threadIdx.x + blockIdx.x * blockDim.x;

    // Check for valid threads
    if (globalId >= numberOfTriangles) {
        return;
    }

    // Load vertices into shared memory
    int globalTriangleId = globalId * 3;
    float4 v1 = vertices[globalTriangleId];
    float4 v2 = vertices[globalTriangleId + 1];
    float4 v3 = vertices[globalTriangleId + 2];

    sortIndices[globalId] = globalId;

    if (globalId == 0) {
        print_float4(v1);
        print_float4(v2);
        print_float4(v3);
        print_float4(bboxMin);
        print_float4(bboxMax);
    }              

    float4 centroid = getTriangleCentroid(v1, v2, v3);
                      
    float4 normalizedCentroid = normalize(centroid, bboxMin, bboxMax);

    unsigned int mortonCode = calculateMortonCode(normalizedCentroid);

    mortonCodes[globalId] = mortonCode;
}

}

''', options=('-I /home/lt0649/Dev/PyBVH/src',
              '-I /usr/local/include/'))

generateMortonCodes = module.get_function('generateMortonCodesKernel3D')

nbTriangles = 512
extents = cp.array([50, 50, 50, 50], dtype=cp.float32)
# Array of float4 using CudaTypes
vertices = cp.random.rand(nbTriangles * 3, 4).astype(cp.float32)
print (vertices)
mortonCodes = cp.zeros(nbTriangles, dtype=cp.uint32)
sortIndices = cp.zeros(nbTriangles, dtype=cp.uint32)

bboxMin = make_float4 (0, 0, 0, 0)
bboxMax = make_float4 (50, 50, 50, 50)

generateMortonCodes(
    (nbTriangles,), (512,), 
    (nbTriangles, vertices, bboxMin, bboxMax,
     mortonCodes, sortIndices))

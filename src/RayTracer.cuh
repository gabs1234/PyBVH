
#pragma once
#include "tree_prokopenko.cuh"
#include "Commons.cuh"

class BVHTree;

class RayTracer {
public:
    __device__ RayTracer(BVHTree *tree, float4 *vertices, unsigned int nbVertices, bool parallelGeometry);

    __device__ float4 sphericalToCartesian(float theta, float phi, float r);
    __device__ float4 computePointInPlane(float2 &I, float4 &O, float4 &R);
    __device__ bool computeRayAABB(float4 &O, float4 &min, float4 &max);
    __device__ float4 computeRayParametric(float t);
    __device__ bool hasParallelGeometry() { return this->parallelGeometry; };

    __device__ float traceRayParallel(float2 P, float theta, float phi, float r, float4 origin, CollisionList &collisions);

    __device__ void printRayTracer() {
        printf("Hello from RayTracer\n");
        printf("raySet: %d\n", raySet);
        printf("parallelGeometry: %d\n", parallelGeometry);
        printf("tree: %p\n", tree);
    }

private:
    BVHTree *tree;
    float4 *vertices;
    unsigned int nbVertices;
    bool parallelGeometry;
    int raySet;
    Ray ray;
    float4 tail;
    float4 reference_direction;
    float4 reference_origin;
};
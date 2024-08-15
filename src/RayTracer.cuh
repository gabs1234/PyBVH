
#pragma once
#include "tree_prokopenko.cuh"
#include "Commons.cuh"

class BVHTree;

class RayTracer {
public:
    __device__ RayTracer(BVHTree *tree, float4 origin, float4 *vertices, unsigned int nbVertices, bool parallelGeometry);

    __device__ float4 sphericalToCartesian(float theta, float phi, float r);
    __device__ void rotateBasis (float4 &u1, float4 &u2, float4 &u3, float theta, float phi);
    __device__ Basis makeProjectionBasis (Basis &MeshBasis, float4 &spherical, float4 &euler);
    __device__ bool computeRayAABB(float4 &O, float4 &min, float4 &max);
    __device__ float4 computeRayParametric(float t);
    __device__ bool hasParallelGeometry() { return this->parallelGeometry; };

    __device__ float traceRayParallel(Ray &ray);

    __device__ void printRayTracer() {
        printf("Hello from RayTracer\n");
        printf("raySet: %d\n", raySet);
        printf("parallelGeometry: %d\n", parallelGeometry);
        printf("tree: %p\n", tree);
    }

private:
    BVHTree *tree;
    float4 origin;
    float4 *vertices;
    unsigned int nbVertices;
    bool parallelGeometry;
    int raySet;
    Ray ray;
    float4 tail;
    float4 reference_direction;
    float4 reference_origin;
};
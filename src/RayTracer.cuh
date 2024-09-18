#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#pragma once
#include "tree_prokopenko.cuh"
#include "Ray.cuh"
#include "Basis.cuh"
#include "Commons.cuh"

class BVHTree;

class RayTracer {
public:
    __host__ __device__ RayTracer(BVHTree *tree, float4 *vertices, unsigned int nb_vertices);
    __host__ __device__ RayTracer(BVHTree *tree, float4 origin, float4 *vertices, unsigned int nb_vertices, bool parallelGeometry);

    __host__ __device__ float4 sphericalToCartesian(float theta, float phi, float r);
    __host__ __device__ BasisNamespace::Basis makeProjectionBasis (BasisNamespace::Basis &MeshBasis, float4 &spherical, float4 &euler);
    __host__ __device__ bool computeRayAABB(float4 &O, float4 &min, float4 &max);
    __host__ __device__ float4 computeRayParametric(float t);
    __host__ __device__ bool hasParallelGeometry() { return this->parallelGeometry; };
    __host__ __device__ float4* getVertices() { return this->vertices; };

    __host__ __device__ float4 phi (int i, int j, float2 D, uint2 N) {
        float delta_x = D.x / (N.x-1);
        float delta_y = D.y / (N.y-1);
        float Dx2 = D.x / 2;
        float Dy2 = D.y / 2;

        float x = -Dx2 + i * delta_x;
        float y = -Dy2 + j * delta_y;
        return make_float4(x, y, 0.0, 0);
    }

    __device__ float traceRayParallel(Ray &ray);

    __host__ __device__ void testSingleRay(Ray ray, CollisionList *collisions);

    __host__ __device__ void printRayTracer() {
        printf("Hello from RayTracer\n");
        printf("raySet: %d\n", raySet);
        printf("parallelGeometry: %d\n", parallelGeometry);
        printf("tree: %p\n", tree);
    }

    __host__ __device__ void getSceneBB(float4 &bbMin, float4 &bbMax) {
        bbMin = this->tree->getSceneBBMin();
        bbMax = this->tree->getSceneBBMax();
    }

    __host__ __device__ int getNbKeys() {
        return this->tree->getNbKeys();
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

__global__ void projectPlaneRaysKernel (
    RayTracer *tracer, float *image, uint2 N, float2 D,
    BasisNamespace::Basis projectionPlaneBasis);
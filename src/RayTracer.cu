#include "RayTracer.cuh"
#include "Ray.cuh"
#include "tree_prokopenko.cuh"
#include "Commons.cuh"

#include <thrust/sort.h>

__device__ RayTracer::RayTracer(BVHTree *tree, float4 origin, float4 *vertices, unsigned int nbVertices, bool parallel) {
    this->tree = tree;
    this->origin = origin;
    this->vertices = vertices;
    this->nbVertices = nbVertices;
    this->raySet = 0;
    this->parallelGeometry = parallel;
}

__device__ float4 RayTracer::sphericalToCartesian(float theta, float phi, float r) {
    float const x = r * sin(theta) * cos(phi);
    float const y = r * sin(theta) * sin(phi);
    float const z = r * cos(theta);

    return make_float4(x, y, z, 1.0);
}



__device__ bool RayTracer::computeRayAABB(float4 &O, float4 &min, float4 &max) {
    float4 sceneBBMin = this->tree->getSceneBBMin();
    float4 sceneBBMax = this->tree->getSceneBBMax();
    float4 rayBBmin, rayBBmax;
    float2 t;
    
    if (this->ray.intersects(sceneBBMin, sceneBBMax, t)) {
        min = this->ray.computeParametric(t.x);
        max = this->ray.computeParametric(t.y);
        return true;
    }
    return false;
}

__device__ void RayTracer::rotateBasis (float4 &u1, float4 &u2, float4 &u3, float theta, float phi) {
    RotationQuaternion rot1(-theta, u1);
    RotationQuaternion rot2(phi, u3);

    float4 u1p = rot1.rotate(u1);
    float4 u2p = rot1.rotate(u2);
    float4 u3p = rot1.rotate(u3);

    u1p = rot2.rotate(u1p);
    u2p = rot2.rotate(u2p);
    u3p = rot2.rotate(u3p);

    u1 = u1p;
    u2 = u2p;
    u3 = u3p;
}

__device__ float computeThickness(CollisionList &tvalues) {
    

    int i, j;
    float result = 0.0;

    i = 0;
    while (i < tvalues.count) {
        j = i + 1;
        while (j < tvalues.count && fabsf (tvalues.collisions[j] - tvalues.collisions[i]) < 0.00001) {
            j++;
        }
        if (i < tvalues.count && j < tvalues.count) {
            result += fabsf (tvalues.collisions[j] - tvalues.collisions[i]);
        }
        i = j + 1;
    }

    return result;
}

__device__ float sumTvalues (CollisionList &t_values) {
    float thickness = 0;
    for (int i = 0; i < t_values.count; i++) {
        thickness += t_values.collisions[i];
    }
    return thickness;
}

__device__ float RayTracer::traceRayParallel(Ray &ray) {
    

    CollisionList candidates;
    candidates.count = 0;
    memset(candidates.collisions, 0, MAX_COLLISIONS * sizeof(float));

    // This is where the acceleration structure (BVH) is actually usefull
    this->tree->query(ray, candidates);

    // for (int i = 0; i < candidates.count; i++) {
    //     candidates.collisions[i] += 1;
    // }

    // Test the candidates for actual intersections
    for (int i = 0; i < candidates.count; i++) {
        int primIndex = candidates.collisions[i]*3;

        if (primIndex + 2>= this->nbVertices || primIndex < 0) {
            printf ("Invalid index %d\n", primIndex);
        }
        
        // printf("Collision at %d\n", primIndex);
        float4 V1 = this->vertices[primIndex];
        float4 V2 = this->vertices[primIndex + 1];
        float4 V3 = this->vertices[primIndex + 2];

        float t;
        if (ray.intersects(V1, V2, V3, t)) {
            // printf("real Collision at %d, %f\n", primIndex, t);
            candidates.collisions[i] = t;
        }
    }

    // Print the t_values


    // // Sort the t_values
    // thrust::sort(thrust::device, candidates.collisions, candidates.collisions + candidates.count);

    // printf ("t_values count = %d\n", t_values.count);
    // for (int i = 0; i < t_values.count; i++) {
    //     printf ("t_values[%d] = %f\n", i, t_values.collisions[i]);
    // }

    // compute the thickness
    return candidates.count;
}
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#pragma once

#include "tree_prokopenko.cuh"
#include "RayTracer.cuh"
#include "Basis.cuh"
#include "Commons.cuh"
#include "Ray.cuh"

typedef struct {
    unsigned int nb_keys;
    float4 *vertices;
    float4 *bbMinLeaf;
    float4 *bbMaxLeaf;
    float4 bbMinScene;
    float4 bbMaxScene;
} Scene;

// __global__ void printTreeKernel (BVHTree *tree);

class SceneManager {
public:
    SceneManager (unsigned int N);
    SceneManager (float4 *vertices, unsigned int N, float4 bbMinScene, float4 bbMaxScene);
    SceneManager (Scene &scene);
    SceneManager (
        unsigned int N, float4 *bbMin, float4 *bbMax, 
        float4 bbMinScene, float4 bbMaxScene, float4 *vertices);

    ~SceneManager();

    BVHTree *getDeviceTree() { return device_tree; };
    RayTracer *getDeviceRayTracer() { return device_ray_tracer; };

    void deviceToHost ();
    void printNodes ();
    void getTreeStructure ();

    void calculateTriangleBoundingBoxes (Scene &scene);

    void setupAccelerationStructure ();

    bool sanityCheck ();
    
    CollisionList getCollisionList (unsigned int index);
    float* projectPlaneRays (uint2 &N, float2 &D, float4 &spherical, float4 &euler, float4 &meshOrigin);

private:
    unsigned int nb_keys, nb_nodes;
    Scene scene, device_scene;

    // Device variables
    BVHTree *device_tree;
    float4 device_bbMinScene, device_bbMaxScene;
    morton_t *device_keys;
    unsigned int *device_sorted_indices;
    float4 *device_bbMinLeaf, *device_bbMaxLeaf;
    float4 *device_bbMinInternal, *device_bbMaxInternal;
    int *device_left_child, *device_entered, *device_rope_leaf, *device_rope_internal;
    Nodes device_leaf_nodes, device_internal_nodes;

    // Ray tracer variables
    float4 *device_vertices;
    RayTracer *device_ray_tracer;

    // int *host_left_range, *device_left_range;
    // int *host_right_range, *device_right_range;
    // int *host_right_child, *device_right_child;
};
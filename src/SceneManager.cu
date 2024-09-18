#include "SceneManager.cuh"
// #define THRUST_IGNORE_CUB_VERSION_CHECK
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

// Functor to find the minimum float4 element-wise
struct float4_min {
    __host__ __device__
    float4 operator()(const float4& a, const float4& b) const {
        float4 result;
        result.x = fminf(a.x, b.x);
        result.y = fminf(a.y, b.y);
        result.z = fminf(a.z, b.z);
        result.w = fminf(a.w, b.w);
        return result;
    }
};

struct float4_max {
    __host__ __device__
    float4 operator()(const float4& a, const float4& b) const {
        float4 result;
        result.x = fmaxf(a.x, b.x);
        result.y = fmaxf(a.y, b.y);
        result.z = fmaxf(a.z, b.z);
        return result;
    }
};

// __global__ void initializeTreeStructureKernel (
//     BVHTree *tree, unsigned int nb_keys, float4 *vertices,
//     float4 bbMinScene, float4 bbMaxScene,
//     morton_t *device_keys, unsigned int *sorted_indices,
//     Nodes internal_nodes, Nodes leaf_nodes) {
//     unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= 1) {
//         return;
//     }
    
//     new (tree) BVHTree(nb_keys, device_keys);
//     tree->setSortedIndices(sorted_indices);
//     tree->setInternalNodes(internal_nodes);
//     tree->setLeafNodes(leaf_nodes);
//     tree->setSceneBB(bbMinScene, bbMaxScene);
// }

__global__ void calculateBbBoxKernel (float4 *vertices, float4 *bbMin, float4 *bbMax, unsigned int nb_keys) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < nb_keys) {
        float4 V1 = vertices[tid * 3];
        float4 V2 = vertices[tid * 3 + 1];
        float4 V3 = vertices[tid * 3 + 2];
        calculateTriangleBoundingBox (V1, V2, V3, &(bbMin[tid]), &(bbMax[tid]));

        tid += blockDim.x * gridDim.x;
    }
    
}

__global__ void projectKeys(
        Scene scene, morton_t *keys, unsigned *sorted_indices,
        Nodes inodes, Nodes lnodes) {

    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

    while (index < scene.nb_keys) {
        // Calculate the centroid of the AABB
        float4 centroid = getBoundingBoxCentroid(scene.bbMinLeaf[index], scene.bbMaxLeaf[index]);
        
        float4 normalizedCentroid = normalize(centroid, scene.bbMinScene, scene.bbMaxScene);

        // Calculate the morton code of the triangle
        morton_t mortonCode = calculateMortonCode(normalizedCentroid);

        // Store the morton code
        keys[index] = mortonCode;

        // Setup the sorted indices + Node structure
        sorted_indices[index] = index;
        inodes.entered[index + scene.nb_keys] = INVALID;
        inodes.left_child[index + scene.nb_keys] = SENTINEL;
        inodes.rope[index + scene.nb_keys] = SENTINEL;
        lnodes.rope[index] = SENTINEL;

        index += blockDim.x * gridDim.x;
    }
}

__global__ void growTreeKernel (BVHTree *deviceTree) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    while (index < deviceTree->getNbKeys()) {
        deviceTree->updateParents(index);
        index += blockDim.x * gridDim.x;
    }
}

__global__ void printTreeKernel (BVHTree *tree) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // printf("Hello from printTreeKernel\n");
    if (tid != 0) {
        return;
    }
    tree->printTree();
}

__global__ void testSingleRayKernel (RayTracer *ray_tracer, unsigned int primitive_index, CollisionList *collisions) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) {
        return;
    }

    float4 scene_bbMin, scene_bbMax;
    ray_tracer->getSceneBB(scene_bbMin, scene_bbMax);

    int nb_keys = ray_tracer->getNbKeys();

    float dx = (scene_bbMax.x - scene_bbMin.x) / (nb_keys );
    float dy = (scene_bbMax.y - scene_bbMin.y) / (nb_keys );

    float4* vertices = ray_tracer->getVertices();

    // // print the vertices
    // for (int i = 0; i < 3; i++) {
    //     float4 V1 = vertices[primitive_index + i];
        // printf("V1[%d] = (%f, %f, %f, %f)\n", i, V1.x, V1.y, V1.z, V1.w);
    // }

    float4 V1 = vertices[primitive_index+1];
    V1.z = -5;
    V1.x += dx/4;
    V1.y += dy/4;

    printf ("ray origin = (%f, %f, %f, %f)\n", V1.x, V1.y, V1.z, V1.w);

    float4 direction = make_float4(0, 0, 1, 0);

    Ray ray = Ray(V1, direction);

    ray_tracer->testSingleRay(ray, collisions);
}

// __global__ void descendTreeKernel (BVHTree *tree, int *left_child, int *right_child) {
//     unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid != 0) {
//         return;
//     }

//     tree->traverse(left_child, right_child);
// }

SceneManager::SceneManager (unsigned int N) : nb_keys(N) {
    // Allocate the geometry
    cudaCheckError (cudaMallocManaged(&device_vertices, nb_keys * 3 * sizeof(float4)));
    cudaCheckError (cudaMallocManaged(&device_keys, nb_keys * sizeof(morton_t)));
    cudaCheckError (cudaMallocManaged(&device_sorted_indices, nb_keys * sizeof(unsigned int)));

    // Allocate the internal nodes
    cudaCheckError (cudaMallocManaged(&device_internal_nodes.bbMin, (2 * nb_keys) * sizeof(float4)));
    cudaCheckError (cudaMallocManaged(&device_internal_nodes.bbMax, (2 * nb_keys) * sizeof(float4)));
    cudaCheckError (cudaMallocManaged(&device_internal_nodes.rope, (2 * nb_keys) * sizeof(int)));
    cudaCheckError (cudaMallocManaged(&device_internal_nodes.left_child, (2 * nb_keys) * sizeof(int)));
    cudaCheckError (cudaMallocManaged(&device_internal_nodes.entered, (2 * nb_keys) * sizeof(int)));

    // Allocate the leaf nodes
    // cudaCheckError (cudaMallocManaged(&device_leaf_nodes.rope, nb_keys * sizeof(int)));
    // cudaCheckError (cudaMallocManaged(&device_leaf_nodes.bbMin, nb_keys * sizeof(float4)));
    // cudaCheckError (cudaMallocManaged(&device_leaf_nodes.bbMax, nb_keys * sizeof(float4)));
    device_leaf_nodes.rope = device_internal_nodes.rope;
    device_leaf_nodes.bbMin = device_internal_nodes.bbMin;
    device_leaf_nodes.bbMax = device_internal_nodes.bbMax;

    // Allocate the tree and the ray tracer
    cudaCheckError (cudaMallocManaged(&device_tree, sizeof(BVHTree)));
    cudaCheckError (cudaMallocManaged(&device_ray_tracer, sizeof(RayTracer)));

    // std::cout << "SceneManager created with " << nb_keys << " keys" << std::endl;
}

SceneManager::SceneManager (float4 *vertices, unsigned int N, float4 bbMinScene, float4 bbMaxScene) : SceneManager(N) {
    int blockSize = 256;
    int numBlocks = (nb_keys + blockSize - 1) / blockSize;

    // Print the number of keys
    // std::cout << "Number of keys = " << nb_keys << std::endl;
    
    // Copy the vertices to the device
    // std::cout << "Copying the vertices to the device" << std::endl;
    for (size_t i = 0; i < nb_keys * 3; i++) {
        device_vertices[i] = vertices[i];
    }
    
    // Calculate the bounding boxes
    // std::cout << "Calculating the bounding boxes" << std::endl;
    calculateBbBoxKernel<<<numBlocks, blockSize>>>(device_vertices, device_leaf_nodes.bbMin, device_leaf_nodes.bbMax, nb_keys);
    cudaDeviceSynchronize();
    CUDA_KERNEL_LAUNCH_CHECK();

    device_bbMinScene = bbMinScene;
    device_bbMaxScene = bbMaxScene;

    this->device_scene.nb_keys = nb_keys;
    this->device_scene.vertices = device_vertices;
    this->device_scene.bbMinLeaf = device_leaf_nodes.bbMin;
    this->device_scene.bbMaxLeaf = device_leaf_nodes.bbMax;
    this->device_scene.bbMinScene = device_bbMinScene;
    this->device_scene.bbMaxScene = device_bbMaxScene;
}

SceneManager::SceneManager (Scene &scene) : SceneManager(scene.nb_keys) {
    this->scene = scene;

    // std::cout << "Copying the scene to the device with " << nb_keys << " keys" << std::endl;
    for (size_t i = 0; i < nb_keys; i++) {
        device_leaf_nodes.bbMin[i] = scene.bbMinLeaf[i];
        device_leaf_nodes.bbMax[i] = scene.bbMaxLeaf[i];
    }
    for (size_t i = 0; i < nb_keys * 3; i++) {
        device_vertices[i] = scene.vertices[i];
    }

    device_bbMinScene = scene.bbMinScene;
    device_bbMaxScene = scene.bbMaxScene;

    this->device_scene.nb_keys = nb_keys;
    this->device_scene.vertices = device_vertices;
    this->device_scene.bbMinLeaf = device_leaf_nodes.bbMin;
    this->device_scene.bbMaxLeaf = device_leaf_nodes.bbMax;
    this->device_scene.bbMinScene = device_bbMinScene;
    this->device_scene.bbMaxScene = device_bbMaxScene;

    // std::cout << "Scene bounding box min = (" << device_bbMinScene.x << ", " << device_bbMinScene.y << ", " << device_bbMinScene.z << ", " << device_bbMinScene.w << ")" << std::endl;
    // std::cout << "Scene bounding box max = (" << device_bbMaxScene.x << ", " << device_bbMaxScene.y << ", " << device_bbMaxScene.z << ", " << device_bbMaxScene.w << ")" << std::endl;
}

void SceneManager::setupAccelerationStructure () {
    unsigned blockSize = 256;
    unsigned numBlocks = (nb_keys + blockSize - 1) / blockSize;
    float milliseconds1 = 0, milliseconds2 = 0, milliseconds3 = 0, milliseconds4 = 0, milliseconds5 = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // Project the keys
    cudaEventRecord(start, 0);
    projectKeys<<<numBlocks, blockSize>>>(
        this->device_scene, device_keys, device_sorted_indices,
        device_internal_nodes, device_leaf_nodes
    );
    CUDA_KERNEL_LAUNCH_CHECK();
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds1, start, stop);
    // printf("Projecting the keys took %f milliseconds\n", milliseconds);

    unsigned int *d_keys_in = device_keys;
    unsigned int *d_values_in = device_sorted_indices;
    unsigned int *d_keys_out;
    unsigned int *d_values_out;
    void *tmp_storage = nullptr;
    size_t tmp_storage_bytes = 0;
    cudaMallocManaged(&d_keys_out, nb_keys * sizeof(morton_t));
    cudaMallocManaged(&d_values_out, nb_keys * sizeof(unsigned int));

    // Sort the keys
    // printf ("Sorting the keys\n");
    cudaEventRecord(start, 0);
    cub::DeviceRadixSort::SortPairs(
        tmp_storage, tmp_storage_bytes,
        d_keys_in, d_keys_out,  // Input keys and values (indices)
        d_values_in, d_values_out,  // Output keys and values (sorted keys and indices)
        nb_keys
    );
    cudaMalloc(&tmp_storage, tmp_storage_bytes);

    cub::DeviceRadixSort::SortPairs(
        tmp_storage, tmp_storage_bytes,
        d_keys_in, d_keys_out,  // Input keys and values (indices)
        d_values_in, d_values_out,  // Output keys and values (sorted keys and indices)
        nb_keys
    );
    CUDA_KERNEL_LAUNCH_CHECK();
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds2, start, stop);
    // printf("Sorting the keys took %f milliseconds\n", milliseconds);

    // print out the sorted indices
    for (int i = 0; i < nb_keys; i++) {
        device_sorted_indices[i] = d_values_out[i];
        device_keys[i] = d_keys_out[i];
    }

    cudaFree(tmp_storage);
    cudaFree(d_keys_out);
    cudaFree(d_values_out);

    // Initialize the tree structure
    // printf ("Initializing the tree structure\n");
    new (device_tree) BVHTree (
        nb_keys, device_keys, device_sorted_indices,
        device_internal_nodes, device_leaf_nodes,
        device_bbMinScene, device_bbMaxScene
    );

    // Update the parents
    // printf ("Growing the tree\n");
    cudaEventRecord(start, 0);
    growTreeKernel<<<numBlocks, blockSize>>>(device_tree);
    cudaDeviceSynchronize();
    CUDA_KERNEL_LAUNCH_CHECK();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds3, start, stop);
    // printf("Growing the tree took %f milliseconds\n", milliseconds);

    // Initialize the ray tracer
    // printf ("Initializing the ray tracer\n");
    new (device_ray_tracer) RayTracer(device_tree, device_vertices, 3 * nb_keys);

    // printf ("Acceleration structure setup complete\n");
    printf("%f, %f, %f", milliseconds1, milliseconds2, milliseconds3);
}

SceneManager::~SceneManager() {
    cudaCheckError (cudaFree(device_keys));
    cudaCheckError (cudaFree(device_sorted_indices));

    cudaCheckError (cudaFree(device_internal_nodes.bbMin));
    cudaCheckError (cudaFree(device_internal_nodes.bbMax));
    cudaCheckError (cudaFree(device_internal_nodes.rope));
    cudaCheckError (cudaFree(device_internal_nodes.left_child));
    cudaCheckError (cudaFree(device_internal_nodes.entered));

    // cudaCheckError (cudaFree(device_leaf_nodes.bbMin));
    // cudaCheckError (cudaFree(device_leaf_nodes.bbMax));
    // cudaCheckError (cudaFree(device_leaf_nodes.rope));

    cudaCheckError (cudaFree(device_tree));
    cudaCheckError (cudaFree(device_ray_tracer));
}

void SceneManager::getTreeStructure () {
    // Print the sorted indices
    for (int i = 0; i < nb_keys; i++) {
        // printf("sorted_indices[%d] = %d\n", i, device_sorted_indices[i]);
    }
    // Print the keys
    for (int i = 0; i < nb_keys; i++) {
        // printf("Key sorted %d = %d\n", i, device_keys[i]);
    }
    // print the internal nodes left child and rope
    // printf ("left_child = [");
    for (int i = 0; i < nb_keys; i++) {
        // printf("%d, ", device_internal_nodes.left_child[i]);
    }
    // printf("]\n");
    // printf ("rope_internal = [");
    for (int i = 0; i < nb_keys; i++) {
        // printf("%d, ", device_internal_nodes.rope[i]);
    }
    // printf("]\n");
    // print the leaf nodes rope
    // printf ("rope_leaf = [");
    for (int i = 0; i < nb_keys; i++) {
        // printf("%d, ", device_leaf_nodes.rope[i]);
    }
    // printf("]\n");
    // Print the bbmin and bbmax of internal nodes
    // printf ("bbMin_internal = [");
    for (int i = 0; i < nb_keys - 1; i++) {
        // printf("(%f, %f, %f, %f), ", device_internal_nodes.bbMin[i].x, device_internal_nodes.bbMin[i].y, device_internal_nodes.bbMin[i].z, device_internal_nodes.bbMin[i].w);
    }
    // printf("]\n");
    // printf ("bbMax_internal = [");
    for (int i = 0; i < nb_keys - 1; i++) {
        // printf("(%f, %f, %f, %f), ", device_internal_nodes.bbMax[i].x, device_internal_nodes.bbMax[i].y, device_internal_nodes.bbMax[i].z, device_internal_nodes.bbMax[i].w);
    }
    // printf("]\n");
    // Print the bbmin and bbmax of leaf nodes
    // printf ("bbMin_leaf = [");
    for (int i = 0; i < nb_keys; i++) {
        // printf("(%f, %f, %f, %f), ", device_leaf_nodes.bbMin[i].x, device_leaf_nodes.bbMin[i].y, device_leaf_nodes.bbMin[i].z, device_leaf_nodes.bbMin[i].w);
    }
    // printf("]\n");
    // printf ("bbMax_leaf = [");
    for (int i = 0; i < nb_keys; i++) {
        // printf("(%f, %f, %f, %f), ", device_leaf_nodes.bbMax[i].x, device_leaf_nodes.bbMax[i].y, device_leaf_nodes.bbMax[i].z, device_leaf_nodes.bbMax[i].w);
    }
    // printf("]\n");
}

float* SceneManager::projectPlaneRays (
    uint2 &N, float2 &D, float4 &spherical, float4 &euler, float4 &meshOrigin) {
    // 2D grid
    dim3 blockSize(16,16,1);
    dim3 numBlocks((N.x + blockSize.x - 1) / blockSize.x, (N.y + blockSize.y - 1) / blockSize.y, 1);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // printf ("nb_keys = %d\n", nb_keys);
    float* host_image = new float[N.x * N.y];
    float* device_image;
    cudaMallocManaged(&device_image, N.x * N.y * sizeof(float));

    // // Print spherical and euler
    // printf ("spherical = (%f, %f, %f, %f)\n", spherical.x, spherical.y, spherical.z, spherical.w);
    // printf ("euler = (%f, %f, %f, %f)\n", euler.x, euler.y, euler.z, euler.w);

    // Project the rays
    // printf ("Projecting the rays\n");

    BasisNamespace::Basis meshBasis = BasisNamespace::Basis(
        meshOrigin,
        make_float4(1, 0, 0, 0),
        make_float4(0, 1, 0, 0),
        make_float4(0, 0, 1, 0));

    BasisNamespace::Basis projectionPlaneBasis = device_ray_tracer->makeProjectionBasis(meshBasis, spherical, euler);

    cudaEventRecord(start, 0);
    projectPlaneRaysKernel<<<numBlocks, blockSize>>>(
        device_ray_tracer, device_image, N, D, projectionPlaneBasis);
    cudaDeviceSynchronize();
    CUDA_KERNEL_LAUNCH_CHECK();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf ("% f\n", milliseconds);
    // printf("Projecting the rays took %f milliseconds\n", milliseconds);

    // Copy the image to the host
    for (int i = 0; i < N.x * N.y; i++) {
        host_image[i] = device_image[i];
    }

    cudaFree(device_image);

    return host_image;
}

CollisionList SceneManager::getCollisionList (unsigned int index) {
    
    CollisionList *device_candidates;
    cudaMallocManaged(&device_candidates, sizeof(CollisionList));
    device_candidates->count = 0;
    for (int i=0; i < MAX_COLLISIONS; i++) {
        device_candidates->collisions[i] = 0;
    }

    testSingleRayKernel <<<1, 1>>>(device_ray_tracer, index, device_candidates);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }

    // Copy the collision list to the host
    CollisionList host_candidates;
    for (int i = 0; i < device_candidates->count; i++) {
        host_candidates.collisions[i] = device_candidates->collisions[i];
    }
    host_candidates.count = device_candidates->count;

    cudaFree(device_candidates);

    return host_candidates;
}

bool SceneManager::sanityCheck () {
    int id = 0;
    bool ret = device_tree->sanityCheck(id);

    if (ret) {
        return true;
    }

    if (0 <= id && id < nb_keys) {
        // print the vertices of the returned node
        float4 V1 = device_vertices[id * 3];
        float4 V2 = device_vertices[id * 3 + 1];
        float4 V3 = device_vertices[id * 3 + 2];
        
        // printf("V1 = (%f, %f, %f, %f)\n", V1.x, V1.y, V1.z, V1.w);
        // printf("V2 = (%f, %f, %f, %f)\n", V2.x, V2.y, V2.z, V2.w);
        // printf("V3 = (%f, %f, %f, %f)\n", V3.x, V3.y, V3.z, V3.w);
    }

    return false;
}
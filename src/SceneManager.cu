#include "SceneManager.cuh"
// include thrust sort
// #define THRUST_IGNORE_CUB_VERSION_CHECK
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <numeric>
// #include <thrust/device_vector.h>
// #include <thrust/execution_policy.h>
// #include <thrust/device_ptr.h>
// #include <thrust/sort.h>

template<typename T>
std::vector<size_t> argsort(const std::vector<T> &array) {
    std::vector<size_t> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&array](int left, int right) -> bool {
                  // sort indices according to corresponding array element
                  return array[left] < array[right];
              });

    return indices;
}

__global__ void initializeTreeKernel (
    BVHTree *tree, unsigned int nb_keys, float4 *vertices,
    morton_t *keys, unsigned int *sorted_indices,
    float4 bbMinScene, float4 bbMaxScene,
    Nodes internal_nodes, Nodes leaf_nodes) {
    
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) {
        return;
    }

    // print the bbmaxscene
    printf("bbMaxScene = (%f, %f, %f, %f)\n", bbMaxScene.x, bbMaxScene.y, bbMaxScene.z, bbMaxScene.w);
    printf("bbMinScene = (%f, %f, %f, %f)\n", bbMinScene.x, bbMinScene.y, bbMinScene.z, bbMinScene.w);
    // Set the entered and left child to -1
    for (int i = 0; i < nb_keys - 1; i++) {
        internal_nodes.entered[i] = -1;
        internal_nodes.left_child[i] = -1;
        internal_nodes.rope[i] = -1;
        leaf_nodes.rope[i] = -1;
    }

    new (tree) BVHTree(nb_keys, keys);
    tree->setSceneBB(bbMinScene, bbMaxScene);
    tree->setSortedIndices(sorted_indices);
    tree->setInternalNodes(internal_nodes);
    tree->setLeafNodes(leaf_nodes);

    // // Print tree
    // tree->printTree();
}

__global__ void initializeRayTracerKernel (RayTracer *ray_tracer, BVHTree *tree, float4 *vertices) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) {
        return;
    }

    new (ray_tracer) RayTracer(tree, vertices, tree->getNbKeys());
}

__global__ void printTreeKernel (BVHTree *tree) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from printTreeKernel\n");
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

    float4* vertices = ray_tracer->getVertices();
    float4 V1 = vertices[primitive_index];
    float4 direction = make_float4(0, 0, 1, 0);

    Ray ray = Ray(V1, direction);

    ray_tracer->testSingleRay(ray, collisions);
}


SceneManager::SceneManager (
    unsigned int N, float4 *bbMin, float4 *bbMax, 
    float4 bbMinScene, float4 bbMaxScene, float4 *vertices) : nb_keys(N) {

    // /** Setup the host variables **/
    // host_keys = new morton_t[nb_keys];
    // host_sorted_indices = new unsigned int[nb_keys];
    // host_bbMinLeaf = bbMin;
    // host_bbMaxLeaf = bbMax;

    // host_internal_nodes.nb_nodes = nb_keys;
    // host_internal_nodes.bbMin = new float4[nb_keys];
    // host_internal_nodes.bbMax = new float4[nb_keys];
    // host_internal_nodes.rope = new int[nb_keys];
    // host_internal_nodes.left_child = new int[nb_keys];
    // host_internal_nodes.entered = new int[nb_keys];

    // host_leaf_nodes.nb_nodes = nb_keys;
    // host_leaf_nodes.bbMin = new float4[nb_keys];
    // host_leaf_nodes.bbMax = new float4[nb_keys];
    // host_leaf_nodes.rope = new int[nb_keys];

    // new (host_tree) BVHTree(nb_keys, host_keys);
    // host_tree->setSceneBB(bbMinScene, bbMaxScene);
    // host_tree->setSortedIndices(host_sorted_indices);
    // host_tree->setInternalNodes(host_internal_nodes);
    // host_tree->setLeafNodes(host_leaf_nodes);
    
    /** Setup the device variables **/
    cudaCheckError (cudaMallocManaged(&device_keys, nb_keys * sizeof(morton_t)));
    cudaCheckError (cudaMallocManaged(&device_sorted_indices, nb_keys * sizeof(unsigned int)));

    Nodes tmp_internal_nodes;
    cudaCheckError (cudaMallocManaged(&device_internal_nodes.bbMin, (nb_keys) * sizeof(float4)));
    cudaCheckError (cudaMallocManaged(&device_internal_nodes.bbMax, (nb_keys) * sizeof(float4)));
    cudaCheckError (cudaMallocManaged(&device_internal_nodes.rope, (nb_keys) * sizeof(int)));
    cudaCheckError (cudaMallocManaged(&device_internal_nodes.left_child, (nb_keys) * sizeof(int)));
    cudaCheckError (cudaMallocManaged(&device_internal_nodes.entered, (nb_keys) * sizeof(int)));
    
    tmp_internal_nodes.nb_nodes = nb_keys;
    tmp_internal_nodes.bbMin = device_internal_nodes.bbMin;
    tmp_internal_nodes.bbMax = device_internal_nodes.bbMax;
    tmp_internal_nodes.rope = device_internal_nodes.rope;
    tmp_internal_nodes.left_child = device_internal_nodes.left_child;
    tmp_internal_nodes.entered = device_internal_nodes.entered;
    
    Nodes tmp_leaf_nodes;
    cudaCheckError (cudaMallocManaged(&device_leaf_nodes.bbMin, nb_keys * sizeof(float4)));
    cudaCheckError (cudaMemcpy((void*)device_leaf_nodes.bbMin, (void*)bbMin, nb_keys * sizeof(float4), cudaMemcpyHostToDevice));
    cudaCheckError (cudaMallocManaged(&device_leaf_nodes.bbMax, nb_keys * sizeof(float4)));
    cudaCheckError (cudaMemcpy((void*)device_leaf_nodes.bbMax, (void*)bbMax, nb_keys * sizeof(float4), cudaMemcpyHostToDevice));
    cudaCheckError (cudaMallocManaged(&device_leaf_nodes.rope, nb_keys * sizeof(int)));
    tmp_leaf_nodes.nb_nodes = nb_keys;
    tmp_leaf_nodes.bbMin = device_leaf_nodes.bbMin;
    tmp_leaf_nodes.bbMax = device_leaf_nodes.bbMax;
    tmp_leaf_nodes.rope = device_leaf_nodes.rope;

    // Tree structure
    cudaCheckError (cudaMallocManaged (&device_tree, sizeof(BVHTree)));
    initializeTreeKernel<<<1, 1>>>(device_tree, nb_keys, vertices, device_keys, device_sorted_indices, bbMinScene, bbMaxScene, tmp_internal_nodes, tmp_leaf_nodes);
    cudaCheckError (cudaDeviceSynchronize());

    // Setup the ray tracer
    cudaCheckError (cudaMallocManaged(&device_ray_tracer, sizeof(RayTracer)));
    cudaCheckError (cudaMallocManaged(&device_vertices, nb_keys * 3 * sizeof(float4)));
    cudaCheckError (cudaMemcpy(device_vertices, vertices, nb_keys * 3 * sizeof(float4), cudaMemcpyHostToDevice));
    initializeRayTracerKernel<<<1, 1>>>(device_ray_tracer, device_tree, device_vertices);
    cudaCheckError (cudaDeviceSynchronize());
}

SceneManager::~SceneManager() {
    // delete[] host_keys;
    // delete[] host_sorted_indices;
    // delete[] host_bbMinLeaf;
    // delete[] host_bbMaxLeaf;
    // delete[] host_internal_nodes.bbMin;
    // delete[] host_internal_nodes.bbMax;
    // delete[] host_internal_nodes.rope;
    // delete[] host_internal_nodes.left_child;
    // delete[] host_internal_nodes.entered;
    // delete[] host_leaf_nodes.bbMin;
    // delete[] host_leaf_nodes.bbMax;
    // delete[] host_leaf_nodes.rope;

    cudaFree(device_keys);
    cudaFree(device_sorted_indices);
    cudaFree(device_internal_nodes.bbMin);
    cudaFree(device_internal_nodes.bbMax);
    cudaFree(device_internal_nodes.rope);
    cudaFree(device_internal_nodes.left_child);
    cudaFree(device_leaf_nodes.bbMin);
    cudaFree(device_leaf_nodes.bbMax);
    cudaFree(device_leaf_nodes.rope);
    cudaFree(device_tree);
    cudaFree(device_ray_tracer);
}

void SceneManager::deviceToHost () {
    // Simply copy the Nodes structure
    cudaMemcpy(host_internal_nodes.bbMin, device_internal_nodes.bbMin, (nb_keys - 1) * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_internal_nodes.bbMax, device_internal_nodes.bbMax, (nb_keys - 1) * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_internal_nodes.rope, device_internal_nodes.rope, (nb_keys - 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_internal_nodes.left_child, device_internal_nodes.left_child, (nb_keys - 1) * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(host_leaf_nodes.bbMin, device_leaf_nodes.bbMin, nb_keys * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_leaf_nodes.bbMax, device_leaf_nodes.bbMax, nb_keys * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_leaf_nodes.rope, device_leaf_nodes.rope, nb_keys * sizeof(int), cudaMemcpyDeviceToHost);
}

void SceneManager::printNodes () {
    for (int i = 0; i < nb_keys - 1; i++) {
        printf("Internal node %d\n", i);
        printf("bbMin = (%f, %f, %f, %f)\n", host_internal_nodes.bbMin[i].x, host_internal_nodes.bbMin[i].y, host_internal_nodes.bbMin[i].z, host_internal_nodes.bbMin[i].w);
        printf("bbMax = (%f, %f, %f, %f)\n", host_internal_nodes.bbMax[i].x, host_internal_nodes.bbMax[i].y, host_internal_nodes.bbMax[i].z, host_internal_nodes.bbMax[i].w);
        printf("rope = %d\n", host_internal_nodes.rope[i]);
        printf("left_child = %d\n", host_internal_nodes.left_child[i]);
    }

    for (int i = 0; i < nb_keys; i++) {
        printf("Leaf node %d\n", i);
        printf("bbMin = (%f, %f, %f, %f)\n", host_leaf_nodes.bbMin[i].x, host_leaf_nodes.bbMin[i].y, host_leaf_nodes.bbMin[i].z, host_leaf_nodes.bbMin[i].w);
        printf("bbMax = (%f, %f, %f, %f)\n", host_leaf_nodes.bbMax[i].x, host_leaf_nodes.bbMax[i].y, host_leaf_nodes.bbMax[i].z, host_leaf_nodes.bbMax[i].w);
        printf("rope = %d\n", host_leaf_nodes.rope[i]);
    }
}

void SceneManager::setupAccelerationStructure () {
    int blockSize = 256;
    int numBlocks = (nb_keys + blockSize - 1) / blockSize;

    // Generate the morton keys
    projectKeysKernel<<<numBlocks, blockSize>>>(device_tree);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "yo Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }

    // // Sort the keys
    // cub::DeviceRadixSort::SortPairs(NULL, err, device_keys, device_keys, device_sorted_indices, device_sorted_indices, nb_keys);

    // // Print the tree
    // printTreeKernel<<<1, 1>>>(device_tree);
    // err = cudaDeviceSynchronize();
    // if (err != cudaSuccess) {
    //     std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    // }
    // Sort the keys
    // No need for cudaMemcpy, because of managed memory
    std::vector<morton_t> host_keys(device_keys, device_keys + nb_keys);
    std::vector<size_t> sorted_indices = argsort(host_keys);
    cudaMemcpy(device_sorted_indices, sorted_indices.data(), nb_keys * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // // Print the tree
    // printTreeKernel<<<1, 1>>>(device_tree);
    // err = cudaDeviceSynchronize();
    // if (err != cudaSuccess) {
    //     std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    // }

    // Update the parents
    growTreeKernel<<<numBlocks, blockSize>>>(device_tree);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }

    // // Print the device nodes
    // for (int i = 0; i < nb_keys - 1; i++) {
    //     printf("Internal node %d\n", i);
    //     printf("bbMin = (%f, %f, %f, %f)\n", device_internal_nodes.bbMin[i].x, device_internal_nodes.bbMin[i].y, device_internal_nodes.bbMin[i].z, device_internal_nodes.bbMin[i].w);
    //     printf("bbMax = (%f, %f, %f, %f)\n", device_internal_nodes.bbMax[i].x, device_internal_nodes.bbMax[i].y, device_internal_nodes.bbMax[i].z, device_internal_nodes.bbMax[i].w);
    //     printf("rope = %d\n", device_internal_nodes.rope[i]);
    //     printf("left_child = %d\n", device_internal_nodes.left_child[i]);
    // }

    // Print the tree
    // printTreeKernel<<<1, 1>>>(device_tree);
}

float* SceneManager::projectPlaneRays (
    uint2 &N, float2 &D, float4 &spherical, float4 &euler, float4 &meshOrigin) {
    int blockSize = 256;
    int numBlocks = (nb_keys + blockSize - 1) / blockSize;

    // printf ("nb_keys = %d\n", nb_keys);
    float* host_image = new float[N.x * N.y];
    float* device_image;
    cudaMallocManaged(&device_image, N.x * N.y * sizeof(float));

    projectPlaneRaysKernel<<<numBlocks, blockSize>>>(
        device_ray_tracer, device_image, N, D, spherical, euler, meshOrigin);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }

    // Copy the image to the host
    cudaMemcpy(host_image, device_image, N.x * N.y * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_image);

    return host_image;
}

CollisionList SceneManager::getCollisionList (unsigned int index) {
    
    CollisionList *device_candidates;
    cudaMallocManaged(&device_candidates, sizeof(CollisionList));
    device_candidates->count = 0;
    for (int i=0; i < nb_keys; i++) {
        device_candidates->collisions[i] = 0;
    }

    testSingleRayKernel <<<1, 1>>>(device_ray_tracer, index, device_candidates);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }

    // Copy the collision list to the host
    CollisionList host_candidates;
    for (int i = 0; i < nb_keys; i++) {
        host_candidates.collisions[i] = device_candidates->collisions[i];
    }
    host_candidates.count = device_candidates->count;

    cudaFree(device_candidates);

    return host_candidates;
}
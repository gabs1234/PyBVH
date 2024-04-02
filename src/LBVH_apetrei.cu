#include "Defines.h"
#include <cuda_runtime.h>
// #include <device_launch_parameters.h>

#include "Commons.cuh"
#include "Scene.h"

typedef unsigned int morton_t;

extern "C"
{

#define EPSILON 0.00001f
#define MAX_STACK_PTRS 128
#define MAX_CANDIDATES 64
#define MAX_COLLISIONS 64
#define IS_LEAF 2147483648

typedef struct Ray
{
    float4 origin;
    float4 direction;
    float4 start, end;
} Ray;

typedef struct
{
    int *ids;
    unsigned int *child_left;
    unsigned int *child_right;
    unsigned int *left_range;
    unsigned int *right_range;
    unsigned int *entered;
    float4 *bboxMin;
    float4 *bboxMax;
} InternalNodes;

typedef struct
{
    morton_t *morton_keys;
    unsigned int *sortedIndices;
    float4 *bboxMin;
    float4 *bboxMax;
    unsigned int *entered;
} LeafNodes;

typedef struct NodeStack
{
    int top;
    int capacity;
    unsigned int nodes[MAX_STACK_PTRS];
} NodeStack;

typedef struct _BVHTree
{
    unsigned int number_of_triangles;
    InternalNodes internal_nodes;
    LeafNodes leaf_nodes;
    float4 *vertices;
    float4 scene_bbMin, scene_bbMax;
    int root_index;
    float *area;
} BVHTree;

typedef struct CandidateStruct
{
    unsigned int hits[MAX_CANDIDATES];
    unsigned int count;
} CandidateStruct;

typedef struct CollisionStruct
{
    float hits[MAX_COLLISIONS];
    unsigned int count;
} CollisionStruct;

// Calculate the centroid of the triangle AABB
__device__ float4 getTriangleCentroid(float4 v1, float4 v2, float4 v3)
{
    float4 boundingBoxMin;
    float4 boundingBoxMax;

    boundingBoxMin.x = min(v1.x, v2.x);
    boundingBoxMin.x = min(boundingBoxMin.x, v3.x);
    boundingBoxMax.x = max(v1.x, v2.x);
    boundingBoxMax.x = max(boundingBoxMax.x, v3.x);

    boundingBoxMin.y = min(v1.y, v2.y);
    boundingBoxMin.y = min(boundingBoxMin.y, v3.y);
    boundingBoxMax.y = max(v1.y, v2.y);
    boundingBoxMax.y = max(boundingBoxMax.y, v3.y);

    boundingBoxMin.z = min(v1.z, v2.z);
    boundingBoxMin.z = min(boundingBoxMin.z, v3.z);
    boundingBoxMax.z = max(v1.z, v2.z);
    boundingBoxMax.z = max(boundingBoxMax.z, v3.z);

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

__device__ __forceinline__ void calculateLeafBoundingBox(
    float4 vertex1, float4 vertex2, float4 vertex3,
    float4 *bbMin, float4 *bbMax)
{
    bbMin->x = min(vertex1.x, vertex2.x);
    bbMin->x = min(bbMin->x, vertex3.x);
    bbMin->y = min(vertex1.y, vertex2.y);
    bbMin->y = min(bbMin->y, vertex3.y);
    bbMin->z = min(vertex1.z, vertex2.z);
    bbMin->z = min(bbMin->z, vertex3.z);

    bbMax->x = max(vertex1.x, vertex2.x);
    bbMax->x = max(bbMax->x, vertex3.x);
    bbMax->y = max(vertex1.y, vertex2.y);
    bbMax->y = max(bbMax->y, vertex3.y);
    bbMax->z = max(vertex1.z, vertex2.z);
    bbMax->z = max(bbMax->z, vertex3.z);
}

__device__ __forceinline__ void calculateNodeBoundingBox(float4 *bbMin, float4 *bbMax,
                                                            float4 *leftBbMin, float4 *leftBbMax, float4 *rightBbMin, float4 *rightBbMax)
{
    float4 bboxMin;
    bboxMin.x = min(leftBbMin->x, rightBbMin->x);
    bboxMin.y = min(leftBbMin->y, rightBbMin->y);
    bboxMin.z = min(leftBbMin->z, rightBbMin->z);

    float4 bboxMax;
    bboxMax.x = max(leftBbMax->x, rightBbMax->x);
    bboxMax.y = max(leftBbMax->y, rightBbMax->y);
    bboxMax.z = max(leftBbMax->z, rightBbMax->z);

    *bbMin = bboxMin;
    *bbMax = bboxMax;
}

__device__ __forceinline__ void calculateRayBoundingBox(float4 *ray_start, float4 *ray_end,
                                                        float4 *bbMin, float4 *bbMax)
{
    float4 bboxMin, bboxMax;

    bboxMin.x = min(ray_start->x, ray_end->x);
    bboxMin.y = min(ray_start->y, ray_end->y);
    bboxMin.z = min(ray_start->z, ray_end->z);

    bboxMax.x = max(ray_start->x, ray_end->x);
    bboxMax.y = max(ray_start->y, ray_end->y);
    bboxMax.z = max(ray_start->z, ray_end->z);

    *bbMin = bboxMin;
    *bbMax = bboxMax;
}

__global__ void getTreeStructureSize(unsigned int *size)
{
    size[0] = sizeof(BVHTree);
}

__device__ bool pushNode(unsigned int node, NodeStack *stack)
{
    if (stack->top == stack->capacity)
    {
        return false;
    }
    stack->top++;
    stack->nodes[stack->top] = node;
    // printf("Pushed node %d\n", node->id);
    return true;
}

__device__ bool popNode(unsigned int *node, NodeStack *stack)
{
    if (stack->top == -1)
    {
        return false;
    }
    *node = stack->nodes[stack->top];
    stack->top--;
    // printf("Popped node %d\n", (*node)->id);
    return true;
}

__global__ void initializeTreeStructureKernel(
    unsigned int *BVHTreePtr, unsigned int numberOfTriangles, float4 *vertices,
    float4 *bbMinScene, float4 *bbMaxScene, float4 *bbMinLeaf, float4 *bbMaxLeaf, float4 *bbMinInternal, float4 *bbMaxInternal,
    morton_t *morton_codes, unsigned int *sortedIndices, unsigned int *enteredLeaf, unsigned int *enteredInternal,
    unsigned int *child_left, unsigned int *child_right, unsigned int *left_range, unsigned int *right_range)
{

    unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalId >= 1)
    {
        return;
    }

    BVHTree *tree = (BVHTree *)BVHTreePtr;
    tree->number_of_triangles = numberOfTriangles;
    tree->vertices = vertices;
    tree->scene_bbMin = bbMinScene[0];
    tree->scene_bbMax = bbMaxScene[0];

    // printf("Scene BBMin: %f %f %f\n", tree->scene_bbMin.x, tree->scene_bbMin.y, tree->scene_bbMin.z);
    // printf("Scene BBMax: %f %f %f\n", tree->scene_bbMax.x, tree->scene_bbMax.y, tree->scene_bbMax.z);

    tree->internal_nodes.child_left = child_left;
    tree->internal_nodes.child_right = child_right;
    tree->internal_nodes.left_range = left_range;
    tree->internal_nodes.right_range = right_range;
    tree->internal_nodes.entered = enteredInternal;
    tree->internal_nodes.bboxMin = bbMinInternal;
    tree->internal_nodes.bboxMax = bbMaxInternal;

    tree->leaf_nodes.morton_keys = morton_codes;
    tree->leaf_nodes.sortedIndices = sortedIndices;
    tree->leaf_nodes.bboxMin = bbMinLeaf;
    tree->leaf_nodes.bboxMax = bbMaxLeaf;
    tree->leaf_nodes.entered = enteredLeaf;
}

__global__ void getRootIndexKernel(unsigned int *BVHTreePtr, unsigned int *rootIndex)
{
    BVHTree *tree = (BVHTree *)BVHTreePtr;
    rootIndex[0] = tree->root_index;
}

__global__ void printTreeStructureKernel(unsigned int *BVHTreePtr)
{
    // BVHTree *tree = (BVHTree *)BVHTreePtr;
    // BVHNode *internal_nodes = tree->internal_nodes;
    // printf("Pi|LChi|RChii|LRge|RRge|BBi|root|leaf|int\n");
    // printf("Root: %d\n", tree->root_index);
    // for (int i = 0; i < tree->number_of_triangles-1; i++) {
    //     printf("Node %d: %d %d %d\n", i,
    //         internal_nodes[i].left_range, internal_nodes[i].right_range,
    //         internal_nodes[i].type);
    // }
    // // Print the scene bounding box
    // printf("Scene BBMin: %f %f %f\n", tree->scene_bbMin.x, tree->scene_bbMin.y, tree->scene_bbMin.z);
    // printf("Scene BBMax: %f %f %f\n", tree->scene_bbMax.x, tree->scene_bbMax.y, tree->scene_bbMax.z);
}

__global__ void generateMortonCodesKernel3D(
    int numberOfTriangles, float4 *vertices, morton_t *mortonCodes,
    float4 *bboxMinLeaf, float4 *bboxMaxLeaf, float4 *bboxMinScene, float4 *bboxMaxScene)
{
    int keyId = threadIdx.x + blockIdx.x * blockDim.x;

    while (keyId < numberOfTriangles)
    {

        // Load vertices into shared memory
        int globalTriangleId = keyId * 3;
        float4 v1, v2, v3;
        v1 = reinterpret_cast<float4 *>(vertices)[globalTriangleId];
        v2 = reinterpret_cast<float4 *>(vertices)[globalTriangleId + 1];
        v3 = reinterpret_cast<float4 *>(vertices)[globalTriangleId + 2];

        // Calculate the bounding box of the triangle
        calculateLeafBoundingBox(v1, v2, v3, &bboxMinLeaf[keyId], &bboxMaxLeaf[keyId]);

        // Calculate the centroid of the triangle
        float4 centroid = getBoundingBoxCentroid(bboxMinLeaf[keyId], bboxMaxLeaf[keyId]);
        float4 normalizedCentroid = normalize(centroid, bboxMinScene[0], bboxMaxScene[0]);

        // Calculate the morton code of the triangle
        morton_t mortonCode = calculateMortonCode(normalizedCentroid);

        mortonCodes[keyId] = mortonCode;

        keyId += blockDim.x * gridDim.x;
    }
}

__device__ unsigned int delta(unsigned int id, LeafNodes *lnodes)
{
    // unsigned int id1 = lnodes->sortedIndices[id];
    // unsigned int id2 = lnodes->sortedIndices[id + 1];
    unsigned int a = lnodes->morton_keys[id];
    unsigned int b = lnodes->morton_keys[id+1];
    return a ^ b;
}

__device__ unsigned int chooseParent(
    unsigned int N, unsigned int current_node, int child_left, int child_right,
    InternalNodes *inodes, LeafNodes *lnodes)
{
    unsigned int parent;
    if (child_left == -1 ||((child_right < N-1) &&
        (delta(child_right, lnodes) <= delta(child_left, lnodes))))
    {
        parent = child_right;
        inodes->child_left[parent] = current_node;
        inodes->left_range[parent] = child_left;
        return parent;
    }
    else
    {
        parent = child_left;
        inodes->child_right[parent] = current_node;
        inodes->right_range[parent] = child_right;
        return parent;
    }
}

__device__ void growBbox (unsigned int current_node, InternalNodes *inodes, LeafNodes *lnodes) {
    float4 *bbMinLeft;
    float4 *bbMaxLeft;
    float4 *bbMinRight;
    float4 *bbMaxRight;

    // unsigned int *sortedIndices = lnodes->sortedIndices;

    if (inodes->child_left[current_node] & IS_LEAF)
    {
        unsigned int left_id = inodes->child_left[current_node] ^ IS_LEAF;
        bbMinLeft = &lnodes->bboxMin[left_id];
        bbMaxLeft = &lnodes->bboxMax[left_id];
    }
    else
    {
        unsigned int left_id = inodes->child_left[current_node];
        bbMinLeft = &inodes->bboxMin[left_id];
        bbMaxLeft = &inodes->bboxMax[left_id];
    }

    if (inodes->child_right[current_node] & IS_LEAF)
    {
        unsigned int right_id = inodes->child_right[current_node] ^ IS_LEAF;
        bbMinRight = &lnodes->bboxMin[right_id];
        bbMaxRight = &lnodes->bboxMax[right_id];
    }
    else
    {
        unsigned int right_id = inodes->child_right[current_node];
        bbMinRight = &inodes->bboxMin[right_id];
        bbMaxRight = &inodes->bboxMax[right_id];
    }

    float4 *bbMinNode = &inodes->bboxMin[current_node];
    float4 *bbMaxNode = &inodes->bboxMax[current_node];
    calculateNodeBoundingBox(bbMinNode, bbMaxNode, bbMinLeft, bbMaxLeft, bbMinRight, bbMaxRight);
}

__device__ void updateParentNodes (unsigned int current_node, unsigned int N, InternalNodes *inodes, LeafNodes *lnodes) {
    unsigned int parent = current_node;
    while (atomicXor(&inodes->entered[parent], 1))
    {
        growBbox(parent, inodes, lnodes);
        parent = chooseParent(N, parent, inodes->left_range[parent], inodes->right_range[parent], inodes, lnodes);
    }
}

/**
 * Here we use Apetrei's method to build the BVH tree
 *
 */
__global__ void buildLVBHApetrei(unsigned int *tree_ptr)
{
    BVHTree *tree = (BVHTree *)tree_ptr;
    int leaf_index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int nb_keys = tree->number_of_triangles;
    // unsigned int *sortedIndices = tree->leaf_nodes.sortedIndices;

    LeafNodes *lnodes = &tree->leaf_nodes;
    InternalNodes *inodes = &tree->internal_nodes;

    // leaf_index = sortedIndices[leaf_index];

    while (leaf_index < nb_keys) {

        unsigned int current_node = leaf_index | IS_LEAF;
        current_node = chooseParent(nb_keys, current_node, leaf_index - 1, leaf_index, inodes, lnodes);

        updateParentNodes(current_node, nb_keys, inodes, lnodes);

        leaf_index += blockDim.x * gridDim.x;
    }
}

__device__ int longestCommonPrefix(morton_t *keys, unsigned int *sortedIndices, unsigned int N,
                                   int index1, int index2, unsigned int key1)
{
    // No need to check the upper bound, since i+1 will be at most numberOfElements - 1 (one
    // thread per internal node)
    if (index2 < 0 || index2 >= N)
    {
        return 0;
    }

    unsigned int key2 = keys[sortedIndices[index2]];

    if (key1 == key2)
    {
        return 32 + __clz(index1 ^ index2);
    }

    return __clz(key1 ^ key2);
}

__device__ int sgn(int number)
{
    return (0 < number) - (0 > number);
}


__global__ void buildTreeKernel(unsigned int *tree_ptr)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    BVHTree *tree = (BVHTree *)tree_ptr;
    unsigned int N = tree->number_of_triangles;

    if (i >= (N - 1))
    {
        return;
    }

    LeafNodes *lnodes = &tree->leaf_nodes;
    InternalNodes *inodes = &tree->internal_nodes;
    morton_t *keys = lnodes->morton_keys;
    unsigned int *sortIndices = lnodes->sortedIndices;

    morton_t key1 = keys[sortIndices[i]];

    const int lcp1 = longestCommonPrefix(keys, sortIndices, N, i, i + 1, key1);
    const int lcp2 = longestCommonPrefix(keys, sortIndices, N, i, i - 1, key1);

    const int direction = sgn((lcp1 - lcp2));

    // Compute upper bound for the length of the range
    const int minLcp = longestCommonPrefix(keys, sortIndices, N, i, i - direction, key1);
    int lMax = 128;
    while (longestCommonPrefix(keys, sortIndices, N, i, i + lMax * direction, key1) > 
            minLcp)
    {
        lMax *= 4;
    }

    // Find other end using binary search
    int l = 0;
    int t = lMax;
    while (t > 1)
    {
        t = t / 2;
        if (longestCommonPrefix(keys, sortIndices, N, i, i + (l + t) * direction, key1) >
                minLcp)
        {
            l += t;
        }
    }
    const int j = i + l * direction;

    // Find the split position using binary search
    const int nodeLcp = longestCommonPrefix(keys, sortIndices, N, i, j, key1);
    int s = 0;
    int divisor = 2;
    t = l;
    const int maxDivisor = 1 << (32 - __clz(l));
    while (divisor <= maxDivisor)
    {
        t = (l + divisor - 1) / divisor;
        if (longestCommonPrefix(keys, sortIndices, N, i, i + (s + t) * direction, key1) >
            nodeLcp)
        {
            s += t;
        }
        divisor *= 2;
    }
    const int splitPosition = i + s * direction + min(direction, 0);

    int leftIndex;
    int rightIndex;

    // Update left child pointer
    if (min(i, j) == splitPosition)
    {
        // Children is a leaf, add the number of internal nodes to the index
        int leafIndex = splitPosition | IS_LEAF;
        leftIndex = leafIndex;
        lnodes->entered[leafIndex] = leafIndex;

        // Set the leaf data index
        // tree->SetDataIndex(leafIndex, sortIndices[splitPosition]);
        // lnodes->entered[leafIndex] = 1;
    }
    else
    {
        leftIndex = splitPosition;
    }

    // Update right child pointer
    if (max(i, j) == (splitPosition + 1))
    {
        // Children is a leaf, add the number of internal nodes to the index
        int leafIndex = splitPosition + 1 | IS_LEAF;
        rightIndex = leafIndex;

        lnodes->entered[leafIndex] = leafIndex;

        // Set the leaf data index
        // tree->SetDataIndex(leafIndex, sortIndices[splitPosition + 1]);
    }
    else
    {
        rightIndex = splitPosition + 1;
    }

    // Update children indices
    // tree->SetRightIndex(i, rightIndex);
    // tree->SetLeftIndex(i, leftIndex);
    inodes->child_left[i] = leftIndex;
    inodes->child_right[i] = rightIndex;

    // Set parent using entered flag
    inodes->entered[leftIndex] = i;
    inodes->entered[rightIndex] = i;

    if (i == 0)
    {
        tree->root_index = 0;
    }
}


// __global__ void calculateNodeBoundingBoxesKernel(unsigned int *BVHTreePtr) {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     BVHTree *tree = (BVHTree *)BVHTreePtr;
//     unsigned int N = tree->number_of_triangles;
//     if (i >= N-1) {
//         return;
    

// }

__device__ bool overlap(
    const float4 *bbMinQuery, const float4 *bbMaxQuery,
    const float4 *bbMinNode, const float4 *bbMaxNode)
{
    if (bbMinQuery->x > bbMaxNode->x || bbMaxQuery->x < bbMinNode->x)
    {
        return false;
    }
    if (bbMinQuery->y > bbMaxNode->y || bbMaxQuery->y < bbMinNode->y)
    {
        return false;
    }
    if (bbMinQuery->z > bbMaxNode->z || bbMaxQuery->z < bbMinNode->z)
    {
        return false;
    }
    return true;
}

/**
 * S. Woop et al. 2013, "Watertight Ray/Triangle Intersection"
 */
__device__ int maxDimIndex(const float4 &D)
{
    if (D.x > D.y)
    {
        if (D.x > D.z)
        {
            return 0;
        }
        else
        {
            return 2;
        }
    }
    else
    {
        if (D.y > D.z)
        {
            return 1;
        }
        else
        {
            return 2;
        }
    }
}

/* TODO doc*/
inline float xor_signmask(float x, int y)
{
    return (float)(int(x) ^ y);
}

inline float4 sub(float4 a, float4 b)
{
    float4 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    c.z = a.z - b.z;
    c.w = a.w - b.w;
    return c;
}

inline float4 abs4(float4 a)
{
    float4 c;
    c.x = fabs(a.x);
    c.y = fabs(a.y);
    c.z = fabs(a.z);
    c.w = fabs(a.w);
    return c;
}

__device__ float4 permuteVectorAlongMaxDim(float4 v, unsigned int shift)
{
    float4 c = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    switch (shift)
    {
    case 0:
        c.x = v.y;
        c.y = v.z;
        c.z = v.x;
        c.w = 0;
        break;
    case 1:
        c.x = v.z;
        c.y = v.x;
        c.z = v.y;
        c.w = 0;
        break;
    case 2:
        c.x = v.x;
        c.y = v.y;
        c.z = v.z;
        c.w = 0;
        break;
    }

    // printf("Permuted: %f %f %f\n", c.x, c.y, c.z);

    return c;
}

__device__ float intersects(
    float4 V1, float4 V2, float4 V3,
    float4 O, float4 D)
{
    // calculate dimension where the ray direction is maximal
    const float4 D_abs = abs4(D);
    const int shift = maxDimIndex(D_abs);
    // printf ("Shift: %d\n", shift);
    float4 D_perm = permuteVectorAlongMaxDim(D, shift);
    // printf ("D_perm: %f %f %f\n", D_perm.x, D_perm.y, D_perm.z);

    // swap kx and ky dimensions to preserve winding direction of triangles
    if (D_perm.z < 0.0f)
    {
        const float temp = D_perm.x;
        D_perm.x = D_perm.y;
        D_perm.y = temp;
    }

    /* calculate shear constants */
    float Sx = D_perm.x / D_perm.z;
    float Sy = D_perm.y / D_perm.z;
    float Sz = 1.0f / D_perm.z;

    // printf("D_perm: %f %f %f\n", D_perm.x, D_perm.y, D_perm.z);

    /* calculate vertices relative to ray origin */
    float4 A = sub(V1, O);
    float4 B = sub(V2, O);
    float4 C = sub(V3, O);

    A = permuteVectorAlongMaxDim(A, shift);
    B = permuteVectorAlongMaxDim(B, shift);
    C = permuteVectorAlongMaxDim(C, shift);

    // printf ("A: %f %f %f\n", A.x, A.y, A.z);
    // printf ("B: %f %f %f\n", B.x, B.y, B.z);
    // printf ("C: %f %f %f\n", C.x, C.y, C.z);

    /* perform shear and scale of vertices */
    const float Ax = A.x - Sx * A.z;
    const float Ay = A.y - Sy * A.z;
    const float Bx = B.x - Sx * B.z;
    const float By = B.y - Sy * B.z;
    const float Cx = C.x - Sx * C.z;
    const float Cy = C.y - Sy * C.z;

    // calculate scaled barycentric coordinates
    float U = Cx * By - Cy * Bx;
    float V = Ax * Cy - Ay * Cx;
    float W = Bx * Ay - By * Ax;

    /* fallback to test against edges using double precision  (if float is indeed float) */
    if (U == (float)0.0 || V == (float)0.0 || W == (float)0.0)
    {
        double CxBy = (double)Cx * (double)By;
        double CyBx = (double)Cy * (double)Bx;
        U = (float)(CxBy - CyBx);
        double AxCy = (double)Ax * (double)Cy;
        double AyCx = (double)Ay * (double)Cx;
        V = (float)(AxCy - AyCx);
        double BxAy = (double)Bx * (double)Ay;
        double ByAx = (double)By * (double)Ax;
        W = (float)(BxAy - ByAx);
    }

    if ((U < (float)0.0 || V < (float)0.0 || W < (float)0.0) &&
        (U > (float)0.0 || V > (float)0.0 || W > (float)0.0))
        return -1;

    /* calculate determinant */
    float det = U + V + W;
    if (det == (float)0.0)
        return -1;

    // printf("Det: %f\n", det);

    /* Calculates scaled z-coordinate of vertices and uses them to calculate the hit distance. */
    const float Az = Sz * A.z;
    const float Bz = Sz * B.z;
    const float Cz = Sz * C.z;
    const float T = U * Az + V * Bz + W * Cz;

    const int det_sign_mask = (int(det) & 0x80000000);
    const float xort_t = xor_signmask(T, det_sign_mask);
    if (xort_t < 0.0f)
        return -1;

    // normalize U, V, W, and T
    const float rcpDet = 1.0f / det;
    // *u = U*rcpDet;
    // *v = V*rcpDet;
    // *w = W*rcpDet;

    return T * rcpDet;
}

__global__ void testRayTriangleIntersectionKernel (
    float4 *vertices, float4 *origin, float4 *direction, float *t, int nb_rays) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= nb_rays)
    {
        return;
    }

    float4 v1, v2, v3;
    v1.x = vertices[0].x;
    v1.y = vertices[0].y;
    v1.z = vertices[0].z;
    v1.w = 0.0f;
    v2.x = vertices[1].x;
    v2.y = vertices[1].y;
    v2.z = vertices[1].z;
    v2.w = 0.0f;
    v3.x = vertices[2].x;
    v3.y = vertices[2].y;
    v3.z = vertices[2].z;
    v3.w = 0.0f;

    printf("V1: %f %f %f\n", v1.x, v1.y, v1.z);

    float4 o, d;
    o.x = origin[idx].x;
    o.y = origin[idx].y;
    o.z = origin[idx].z;
    o.w = 0.0f;
    d.x = direction[idx].x;
    d.y = direction[idx].y;
    d.z = direction[idx].z;
    d.w = 0.0f;

    t[idx] = intersects(v1, v2, v3, o, d);
}

/**
 * Traverses the BVH tree to find the candidate triangles
 */
__device__ void bvhTraverse(
    const float4 *bbMinQuery, const float4 *bbMaxQuery, BVHTree *tree,
    NodeStack *toSearchStack, CandidateStruct *candidates)
{
    unsigned int N = tree->number_of_triangles;
    unsigned int root = tree->internal_nodes.child_left[N - 1];
    unsigned int current_node = root;
    unsigned int child_left, child_right;

    float4 *bbMinNodeLeft, *bbMaxNodeLeft;
    float4 *bbMinNodeRight, *bbMaxNodeRight;

    bool overlap_left, overlap_right;

    InternalNodes *inodes = &tree->internal_nodes;
    LeafNodes *lnodes = &tree->leaf_nodes;

    // Push the root node to the stack
    if (!pushNode(root, toSearchStack))
    {
        printf("Stack overflow: root\n");
        return;
    }

    while (popNode(&current_node, toSearchStack))
    {
        child_left = inodes->child_left[current_node];
        child_right = inodes->child_right[current_node];

        if (child_left & IS_LEAF)
        {
            unsigned int leaf_id = child_left ^ IS_LEAF;
            bbMinNodeLeft = &lnodes->bboxMin[leaf_id];
            bbMaxNodeLeft = &lnodes->bboxMax[leaf_id];

            // If the leaf overlaps with the query bounding box, add it to the candidates
            if (overlap(bbMinQuery, bbMaxQuery, bbMinNodeLeft, bbMaxNodeLeft))
            {
                candidates->hits[candidates->count++] = leaf_id;
                if (candidates->count >= MAX_CANDIDATES)
                {
                    return;
                }
            }
        }
        else
        {
            bbMinNodeLeft = &inodes->bboxMin[child_left];
            bbMaxNodeLeft = &inodes->bboxMax[child_left];

            // If the internal node overlaps with the query bounding box, add its children to the stack
            if (overlap(bbMinQuery, bbMaxQuery, bbMinNodeLeft, bbMaxNodeLeft))
            {
                if (!pushNode(child_left, toSearchStack))
                {
                    printf("Stack overflow: LEFT\n");
                    return;
                }
            }
        }

        if (child_right & IS_LEAF)
        {
            unsigned int leaf_id = child_right ^ IS_LEAF;
            bbMinNodeRight = &lnodes->bboxMin[leaf_id];
            bbMaxNodeRight = &lnodes->bboxMax[leaf_id];

            // If the leaf overlaps with the query bounding box, add it to the candidates
            if (overlap(bbMinQuery, bbMaxQuery, bbMinNodeRight, bbMaxNodeRight))
            {
                candidates->hits[candidates->count++] = leaf_id;
                if (candidates->count >= MAX_CANDIDATES)
                {
                    return;
                }
            }
        }
        else
        {
            bbMinNodeRight = &inodes->bboxMin[child_right];
            bbMaxNodeRight = &inodes->bboxMax[child_right];

            // If the internal node overlaps with the query bounding box, add its children to the stack
            if (overlap(bbMinQuery, bbMaxQuery, bbMinNodeRight, bbMaxNodeRight))
            {
                if (!pushNode(child_right, toSearchStack))
                {
                    printf("Stack overflow: RIGHT\n");
                    return;
                }
            }
        }
    }
}

// __global__ void bvhTraverseStackless(
//     const float4 *bbMinQuery, const float4 *bbMaxQuery, BVHTree *tree, CandidateStruct *candidates)
// {
// 	uint i = threadIdx.x + blockIdx.x * blockDim.x;

//     unsigned int N = tree->number_of_triangles;
//     unsigned int root = tree->internal_nodes.child_left[N - 1];
//     unsigned int current_node = root;
//     unsigned int child_left, child_right;

//     float4 *bbMinNodeLeft, *bbMaxNodeLeft;
//     float4 *bbMinNodeRight, *bbMaxNodeRight;

//     bool overlap_left, overlap_right;

//     InternalNodes *inodes = &tree->internal_nodes;
//     LeafNodes *lnodes = &tree->leaf_nodes;
// 	while (i < size)
// 	{
// 		uint currLeaf = i;
// 		DeviceAABB& queryAABB = bvh->LeafNodesAABBs[currLeaf];
// 		bool collides;
// 		bool traverseRightChild = true;

// 		uint curr = bvh->getDfsNextNode(currLeaf ^ IS_LEAF);
// 		// Start the collision detection
// 		while (curr < size - 1)
// 		{
// 			curr = (traverseRightChild) ? bvh->InternalNodesChildren[curr].y : bvh->InternalNodesChildren[curr].x;

// 			collides = queryAABB.Collide(bvh->GetNodeAABB(curr));

// 			if (collides)
// 			{
// 				if (curr & IS_LEAF)
// 				{
// 					uint index = atomicInc(listSize, listCapacity);
// 					list[index].x = bvh->GetBaseObjectIdx(currLeaf);
// 					list[index].y = bvh->GetBaseObjectIdx(curr ^ IS_LEAF);

// 				}
// 				else
// 				{
// 					traverseRightChild = false;
// 					continue;
// 				}
// 			}

// 			curr = bvh->getDfsNextNode(curr);
// 			traverseRightChild = true;
// 		}


// 		i += blockDim.x * gridDim.x;
// 	}
// }

__global__ void testTraversal(BVHTree *tree_ptr)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx > 0)
    {
        return;
    }

    BVHTree *tree = tree_ptr;

    float4 bbMinQuery, bbMaxQuery;
    bbMinQuery.x = 0.0f;
    bbMinQuery.y = 0.0f;
    bbMinQuery.z = -1;

    bbMaxQuery.x = 0.f;
    bbMaxQuery.y = 0.f;
    bbMaxQuery.z = 1.0f;

    NodeStack toSearchStack;
    toSearchStack.top = -1;
    toSearchStack.capacity = MAX_STACK_PTRS;

    CandidateStruct candidates;
    candidates.count = 0;

    bvhTraverse(&bbMinQuery, &bbMaxQuery, tree, &toSearchStack, &candidates);

    // printf("Candidates: %d\n", candidates.count);
}

__device__ void bvhFindCandidates(
    BVHTree *bvhTreePtr, CandidateStruct *candidates,
    float4 *ray_start, float4 *ray_end)
{

    BVHTree *tree = bvhTreePtr;

    float4 bbMinQuery, bbMaxQuery;
    calculateRayBoundingBox(ray_start, ray_end, &bbMinQuery, &bbMaxQuery);

    NodeStack toSearchStack;
    toSearchStack.top = -1;
    toSearchStack.capacity = MAX_STACK_PTRS;

    bvhTraverse(&bbMinQuery, &bbMaxQuery, tree, &toSearchStack, candidates);
}

__device__ void bvhFindCollisions(
    BVHTree *bvhTreePtr, CollisionStruct *collisions,
    float4 *ray_start, float4 *ray_end)
{
    // Use Grid stride loop over the rays

    BVHTree *tree = bvhTreePtr;
    CandidateStruct candidates;
    candidates.count = 0;
    memset(candidates.hits, 0, MAX_CANDIDATES * sizeof(unsigned int));
    bvhFindCandidates(bvhTreePtr, &candidates, ray_start, ray_end);

    // printf("Candidates: %d\n", candidates.count);

    // Loop over the candidates and find actual collisions
    for (int j = 0; j < candidates.count; j++)
    {
        unsigned int triangle_index = candidates.hits[j];

        unsigned int id = triangle_index * 3;

        float4 v1, v2, v3;
        v1.x = tree->vertices[id].x;
        v1.y = tree->vertices[id].y;
        v1.z = tree->vertices[id].z;
        v1.w = 0.0f;
        v2.x = tree->vertices[id + 1].x;
        v2.y = tree->vertices[id + 1].y;
        v2.z = tree->vertices[id + 1].z;
        v2.w = 0.0f;
        v3.x = tree->vertices[id + 2].x;
        v3.y = tree->vertices[id + 2].y;
        v3.z = tree->vertices[id + 2].z;
        v3.w = 0.0f;

        float4 origin, direction;
        origin.x = ray_start->x;
        origin.y = ray_start->y;
        origin.z = ray_start->z;
        origin.w = 0.0f;
        direction.x = ray_end->x - ray_start->x;
        direction.y = ray_end->y - ray_start->y;
        direction.z = ray_end->z - ray_start->z;
        direction.w = 0.0f;
        
        // printf("Direction: %f %f %f\n", direction.x, direction.y, direction.z);

        float t = intersects(v1, v2, v3, origin, direction);
        if (t > -1) {
            collisions->hits[collisions->count++] = t;
            if (collisions->count >= MAX_COLLISIONS)
            {
                return;
            }
        }
    }
}

__device__ float computeThickness(CollisionStruct *collisions)
{
    // Compute the thickness of the object
    float thickness = 0.0f;
    if (collisions->count < 2)
    {
        return 0.0f;
    }
    unsigned int i = 0, j = 0;

    while (i < collisions->count)
    {
        j = i + 1;
        while (j < collisions->count && fabs(collisions->hits[j] - collisions->hits[i]) < EPSILON)
        {
            j++;
        }
        if (i < collisions->count && j < collisions->count)
        {
            thickness += fabs(collisions->hits[j] - collisions->hits[i]);
        }
        i = j + 1;
    }

    return thickness;
}

__device__ float sumOfCollisions(CollisionStruct *collisions)
{
    float sum = 0.0f;
    for (int i = 0; i < collisions->count; i++)
    {
        if (collisions->hits[i] >= 0)
        {
            sum += collisions->hits[i];
        }
    }
    return sum;
}

// __global__ void projectRayDistributionKernel (
//     unsigned int *bvhTreePtr, float4 *ray_start, float4 *ray_end, int nb_rays, float *image, int nx, int ny) {
//     // Compute the thickness of the object

//     if (nb_rays == 0) {
//         return;
//     }
//     __shared__ int stride_x, stride_y;
//     if (threadIdx.x == 0 && threadIdx.y == 0) {
//         stride_x = nx / blockDim.x;
//         stride_y = ny / blockDim.y;
//     }
//     __syncthreads();

//     int threadStartIdx = threadIdx.x + threadIdx.y * blockDim.x;
//     int threadStartIdy = blockIdx.x * stride_x + blockIdx.y * stride_y * nx;

//     CollisionStruct collisions;
//     collisions.count = 0;

//     for (int i = 0; i < nb_rays; i++) {
//         float4 *ray_start_i = &ray_start[i];
//         float4 *ray_end_i = &ray_end[i];
//         bvhFindCollisions(bvhTreePtr, &collisions, ray_start_i, ray_end_i);
//         for (int j = 0; j < collisions.count; j++) {
//             // Compute the thickness of the object
//         }
//     }
// }

__global__ void projectRayGridKernel(
    BVHTree *bvhTreePtr, unsigned int nx, unsigned int ny, float x_pos, float y_pos, float *image)
{
    // Compute the thickness of the object
    unsigned int nb_pixels = nx * ny;
    // Kernel is launched on a 2D grid
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int globalThreadNum = idx + idy * nx;

    while (globalThreadNum < nb_pixels)
    {
        float4 ray_start, ray_end;
        BVHTree *tree = (BVHTree *)bvhTreePtr;
        float4 bbMin = tree->scene_bbMin;
        float4 bbMax = tree->scene_bbMax;
        float pixel_width = (bbMax.x - bbMin.x) / nx;
        float pixel_height = (bbMax.y - bbMin.y) / ny;

        CollisionStruct collisions;
        collisions.count = 0;
        memset(collisions.hits, 0, MAX_COLLISIONS * sizeof(float));

        // Ray in the middle of the pixel
        ray_start.x = (idx + .5) * pixel_width + bbMin.x + x_pos;
        ray_start.y = (idy + .5) * pixel_height + bbMin.y + y_pos;
        ray_start.z = bbMin.z;
        ray_start.w = 0;
        ray_end.x = ray_start.x - 2 * x_pos;
        ray_end.y = ray_start.y - 2 * y_pos;
        ray_end.z = bbMax.z;
        ray_end.w = 0;

        // printf("Ray start: %f %f %f\n", ray_start.x, ray_start.y, ray_start.z);
        bvhFindCollisions(bvhTreePtr, &collisions, &ray_start, &ray_end);

        // printf("Collisions: %d\n", collisions.count);
        image[globalThreadNum] = computeThickness(&collisions);

        idx += blockDim.x * gridDim.x;
        idy += blockDim.y * gridDim.y;
        globalThreadNum = idx + idy * nx;
    }
}

} // extern "C"
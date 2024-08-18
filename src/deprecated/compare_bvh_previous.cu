#include "Ray.cuh"
#include "tree_prokopenko.cuh"
#include "Commons.cuh"

extern "C" {
__global__ void getTreeClassSize (unsigned int *size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > 0) {
        return;
    }
    printf ("Size of BVHTree: %lu\n", sizeof(BVHTree));
    size[0] = sizeof(BVHTree);
}

__global__ void initializeTreeProkopenko(
    BVHTree *tree, unsigned int nb_keys, morton_t *keys, unsigned int *sorted_indices,
    float4 *bbMinScene, float4 *bbMaxScene,
    float4 *bbMinLeafs, float4 *bbMaxLeafs,
    float4 *bbMinInternals, float4 *bbMaxInternals,
    int *left_range, int *right_range,
    int *left_child, int *right_child,
    int *entered, int *rope_leafs, int *rope_internals)
{
    *tree = BVHTree(nb_keys, keys, bbMinScene, bbMaxScene);

    Nodes leaf_nodes;
    leaf_nodes.nb_nodes = nb_keys;
    leaf_nodes.bbMin = bbMinLeafs;
    leaf_nodes.bbMax = bbMaxLeafs;
    leaf_nodes.rope = rope_leafs;

    Nodes internal_nodes;
    internal_nodes.nb_nodes = nb_keys - 1;
    internal_nodes.bbMin = bbMinInternals;
    internal_nodes.bbMax = bbMaxInternals;
    internal_nodes.left_range = left_range;
    internal_nodes.right_range = right_range;
    internal_nodes.left_child = left_child;
    internal_nodes.right_child = right_child;
    internal_nodes.entered = entered;
    internal_nodes.rope = rope_internals;
    
    tree->setLeafNodes(leaf_nodes);
    tree->setInternalNodes(internal_nodes);
    tree->setSortedIndices(sorted_indices);

    // tree->printTree();
}

__global__ void projectKeysProkopenko(BVHTree *tree, unsigned int nb_keys) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    while (index < nb_keys) {
        tree->projectKeys(index);

        index += blockDim.x * gridDim.x;
    }
}

__global__ void buildTreeProkopenko(BVHTree *tree, unsigned int nb_keys) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    while (index < nb_keys) {
        tree->updateParents(index);
        index += blockDim.x * gridDim.x;
    }
}

// __device__ int find_leftmost (float4 *vertices,
//                    int x_0,
//                    int x_1,
//                    float value)
// {
//     int i;

//     while (x_0 <= x_1) {
//         i = (x_0 + x_1) / 2;
//         float4 v1 = vertices[i];
//         float4 v2 = vertices[i+1];
//         float4 v3 = vertices[i+2];
//         if (v3.x == value) {
//             while (v3.x == value) {
//                 i--;
//             }
//             return i;
//         } else if (v3.x < value) {
//             x_0 = i + 1;
//         } else {
//             x_1 = i - 1;
//         }
//     }

//     return i;
// }

// __device__ int compute_intersections (float4 *vertices,
//                            const int num_triangles,
//                            float4 *O,
//                            float max_dx)
// {
//     int i, num_intersections = 0;
//     float current;
//     // Make margins 1 px left and right
//     float xp = O->x + 1;
//     float xm = O->x - 1;
//     float yp = O->y + 1;
//     float ym = O->y - 1;
//     float stop = xp + max_dx;

//     int nbCandidates = 0;

//     /* Find the index for which all the triangles have already ended in
//      * x-direction */
//     i = find_leftmost (vertices, 0, num_triangles, xm);
//     /* Continue the search until we reach max_dx which is the largest triangle
//      * in x-direction, this way we are sure that if we search until ray.x +
//      * max_dx we have searched all the triangles starting to the left from the ray */
//     float4 v1 = vertices[i];
//     float4 v2 = vertices[i+1];
//     float4 v3 = vertices[i+2];
//     while (i < num_triangles && v1.x <= stop && v2.x <= stop && v3.x <= stop) {
//         if (!((v1.x < xm && v2.x < xm && v3.x < xm) ||
//               (v1.x > xp && v2.x > xp && v3.x > xp) ||
//               (v1.y < ym && v2.y < ym && v3.y < ym) ||
//               (v1.y > yp && v2.y > yp && v3.y > yp))) {
//             nbCandidates++;
//         }
//         i++;
//     }

//     return nbCandidates;
// }

// __global__ void findCandidatesOld (float4 *vertices, unsigned int nb_vertices, unsigned int *count) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= 1) return;

//     float4 O = make_float4(0.5, 0.5, -1, 0);

//     int nb_triangles = nb_vertices /3;

//     count[0] = compute_intersections(vertices, nb_triangles, &O, 1);
// }

__global__ void findCandidates (
    BVHTree *tree, unsigned int nb_primitives, unsigned int *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1) return;

    Ray ray(make_float4(0.5, 0.5, -1, 0), make_float4(0, 0, 1, 0));

    CollisionList candidates;
    candidates.count = 0;

    tree->query(ray, candidates);

    count[0] = candidates.count;
    // for (int i = 0; i < candidates.count; i++) {
    //     printf("Candidate %d: %d\n", i, candidates.collisions[i]);
    // }
}

}
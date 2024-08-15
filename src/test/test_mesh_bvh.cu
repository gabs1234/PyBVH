#include <iostream>
#include <limits>
#include <math.h>

#include "../Ray.cuh"
#include "../tree_prokopenko.cuh"

using namespace std;

namespace Color {
    enum Code {
        FG_RED      = 31,
        FG_GREEN    = 32,
        FG_BLUE     = 34,
        FG_DEFAULT  = 39,
        BG_RED      = 41,
        BG_GREEN    = 42,
        BG_BLUE     = 44,
        BG_DEFAULT  = 49
    };
    class Modifier {
        Code code;
    public:
        Modifier(Code pCode) : code(pCode) {}
        friend std::ostream&
        operator<<(std::ostream& os, const Modifier& mod) {
            return os << "\033[" << mod.code << "m";
        }
    };
}

typedef struct {
    float3 v1, v2, v3;
} Triangle;

// void create_triangle_ray_couple (Ray &ray, Triangle &triangle) {
//     ray = Ray(
//         make_float3(0, 0, 0),
//         make_float3(1, 0, 0)
//     );
//     triangle = Triangle {
//         make_float3(0, 0, 0),
//         make_float3(1, 0, 0),
//         make_float3(0, 1, 0)
//     };
// }

__global__ void buildTreeProkopenko(BVHTree *tree, unsigned int nb_keys) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    while (index < nb_keys) {
        tree->updateParents(index);

        // printf ("INDEX: %d\n", index);

        index += blockDim.x * gridDim.x;
    }
}

__global__ void *create_bvh (
    BVHTree *tree, uint nb_keys, morton_t *keys, uint *sorted_indices,
    float4 *bbMinScene, float4 *bbMaxScene,
    float4 *bbMinLeaf, float4 *bbMaxLeaf,
    float4 *bbMinInternal, float4 *bbMaxInternal,
    int *left_range, int *right_range,
    int *left_child, int *right_child,
    int *entered, int *rope_leaf, int *rope_internal) {
    
    cudaMalloc(&tree, sizeof(BVHTree));
    cudaMalloc(&keys, nb_keys * sizeof(morton_t));
    cudaMalloc(&sorted_indices, nb_keys * sizeof(uint));
    cudaMalloc(&bbMinScene, sizeof(float4));
    cudaMalloc(&bbMaxScene, sizeof(float4));
    cudaMalloc(&bbMinLeaf, nb_keys * sizeof(float4));
    cudaMalloc(&bbMaxLeaf, nb_keys * sizeof(float4));
    cudaMalloc(&bbMinInternal, (nb_keys - 1) * sizeof(float4));
    cudaMalloc(&bbMaxInternal, (nb_keys - 1) * sizeof(float4));
    cudaMalloc(&left_range, (nb_keys - 1) * sizeof(int));
    cudaMalloc(&right_range, (nb_keys - 1) * sizeof(int));
    cudaMalloc(&left_child, (nb_keys - 1) * sizeof(int));
    cudaMalloc(&right_child, (nb_keys - 1) * sizeof(int));
    cudaMalloc(&entered, (nb_keys - 1) * sizeof(int));
    cudaMalloc(&rope_leaf, nb_keys * sizeof(int));
    cudaMalloc(&rope_internal, (nb_keys - 1) * sizeof(int));
    
    Nodes leaf_nodes;
    leaf_nodes.nb_nodes = nb_keys;
    leaf_nodes.bbMin = bbMinLeaf;
    leaf_nodes.bbMax = bbMaxLeaf;
    leaf_nodes.rope = rope_leaf;

    Nodes internal_nodes;
    internal_nodes.nb_nodes = nb_keys - 1;
    internal_nodes.bbMin = bbMinInternal;
    internal_nodes.bbMax = bbMaxInternal;
    internal_nodes.rope = rope_internal;
    internal_nodes.left_range = left_range;
    internal_nodes.right_range = right_range;
    internal_nodes.left_child = left_child;
    internal_nodes.right_child = right_child;
    internal_nodes.entered = entered;

    tree->setLeafNodes(leaf_nodes);
    tree->setInternalNodes(internal_nodes);
    tree->setSortedIndices(sorted_indices);
}
int main() {
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);

    bool output = false;

    if (output) {
        cout << green << "Test " << "TODO" << " passed" << endl;
    } else {
        cout << red << "Test " << "TODO" << " failed" << endl;
    }

    return 0;
}
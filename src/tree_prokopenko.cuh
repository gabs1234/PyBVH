#pragma once

#include "Commons.cuh"
#include "Ray.cuh"

class Ray;

#define SENTINEL -1
#define INVALID -1

#define MAX_INT 2147483647
#define MIN_INT -2147483648

typedef unsigned int morton_t;
typedef int delta_t;

typedef struct {
    int rope_id;
    int rope_value;
    int child_lef_id;
    int child_right_id;
    int child_left_value;
    int child_right_value;
} Node;

typedef struct {
    unsigned int nb_nodes;
    int *left_range;
    int *right_range;
    int *left_child;
    int *right_child;
    int *entered;
    int *rope; // points to right child
    float4 *bbMin;
    float4 *bbMax;
} Nodes;

class BVHTree {
public:
    __device__ BVHTree(unsigned int nb_keys, morton_t *keys);
    __device__ BVHTree(unsigned int nb_keys, morton_t *keys, 
        float4 *bbMinScene, float4 *bbMaxScene);
    __device__ BVHTree(unsigned int nb_keys, morton_t *keys, 
        float4 *bbMinScene, float4 *bbMaxScene, float4 *bbMinLeaf, float4 *bbMaxLeaf);

    __device__ void projectKeys(int index);
    __device__ void updateParents(int index);

    __device__ void growLeaf(int parent, int leaf);
    __device__ void growInternal(int parent, int child);
    __device__ void growBox(float4 &bbMinInput, float4 &bbMaxInput, float4 *bbMinOutput, float4 *bbMaxOutput);
    __device__ void setRope(Nodes *nodes, unsigned int skip_index, int range_right, delta_t delta_right);
    __device__ int getRope(int index, int range_right, delta_t delta_right);
    __device__ delta_t delta(int index);
    __device__ int internal_index(int index);
    __device__ int original_index(int index) { return sorted_indices[index]; };

    __device__ void traverse(float4 queryMin, float4 queryMax, CollisionList *candidates);
    __device__ void query(Ray &ray, CollisionList &candidates);
    __device__ void query (float4 &bbMinQuery, float4 &bbMaxQuery, CollisionList &candidates);

    // Setters
    __device__ void setRootIndex(int index) { root_index = index; }
    __device__ void setInternalNodes(Nodes nodes) { internal_nodes = nodes; }
    __device__ void setLeafNodes(Nodes nodes) { leaf_nodes = nodes; }
    __device__ void setSortedIndices(unsigned int *sorted_indices) { this->sorted_indices = sorted_indices; }
    __device__ void setKeys(morton_t *keys) { this->keys = keys; }
    __device__ void setNbKeys(unsigned int nb_keys) { this->nb_keys = nb_keys; }
    __device__ void setSceneBB(float4 bbMin, float4 bbMax) { this->bbMin = bbMin; this->bbMax = bbMax; }
    
    __device__ void setInternalNode(int parent, int left_child, int rope, float4 bbMin, float4 bbMax);
    __device__ void setLeafNode(int index, int rope);

    // Getters
    __device__ int getRootIndex() { return root_index; }
    __device__ Nodes getInternalNodes() { return internal_nodes; }
    __device__ Nodes getLeafNodes() { return leaf_nodes; }
    __device__ unsigned int *getSortedIndices() { return sorted_indices; }
    __device__ morton_t *getKeys() { return keys; }
    __device__ unsigned int getNbKeys() { return nb_keys; }
    __device__ float4 getSceneBBMin() { return bbMin; }
    __device__ float4 getSceneBBMax() { return bbMax; }

    // Helpers
    __device__ void printTree();
    __device__ bool isLeaf(int index);
private:
    // Scene properties
    unsigned int nb_keys;
    float4 *AABBs;
    morton_t *keys;
    unsigned int *sorted_indices;
    float4 bbMin;
    float4 bbMax;

    // Actual tree structure
    int root_index;
    Nodes internal_nodes;
    Nodes leaf_nodes;
};
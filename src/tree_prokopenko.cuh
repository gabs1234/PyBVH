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
    unsigned int nb_nodes;
    int *left_range;
    int *right_range;
    int *left_child;
    int *right_child;
    int *entered;
    int *rope;
    float4 *bbMin;
    float4 *bbMax;
} Nodes;

class BVHTree {
public:
    __host__ __device__ BVHTree(unsigned int nb_keys, morton_t *keys);
    __host__ __device__ BVHTree(unsigned int nb_keys, morton_t *keys, 
        float4 *bbMinScene, float4 *bbMaxScene);
    __host__ __device__ BVHTree(unsigned int nb_keys, morton_t *keys, 
        float4 *bbMinScene, float4 *bbMaxScene, float4 *bbMinLeaf, float4 *bbMaxLeaf);
    
    __host__ __device__ BVHTree(unsigned int nb_keys, morton_t *keys, unsigned int *sorted_indices,
        Nodes internal_nodes, Nodes leaf_nodes, float4 const bbMinScene, float4 const bbMaxScene) {
        this->setNbKeys(nb_keys);
        this->setKeys(keys);
        this->setSortedIndices(sorted_indices);
        this->setInternalNodes(internal_nodes);
        this->setLeafNodes(leaf_nodes);
        this->setSceneBB(bbMinScene, bbMaxScene);
    };

    __host__ __device__ unsigned toInternalRepresentation(unsigned index){
        return index + this->nb_keys;
    };

    __host__ __device__ unsigned toOriginalRepresentation(unsigned index){
        return index - this->nb_keys;
    }; 

    __host__ __device__ unsigned permuteIndex(unsigned index) {
        return sorted_indices[index]; 
    };

    __host__ __device__ void growBox(float4 &bbMinInput, float4 &bbMaxInput, float4 *bbMinOutput, float4 *bbMaxOutput);
    __host__ __device__ void setRope(Nodes *nodes, unsigned int skip_index, int range_right, delta_t delta_right);
    __host__ __device__ void setLeafRope(unsigned int skip_index, int range_right, delta_t delta_right);
    __host__ __device__ void setInternalRope(unsigned int skip_index, int range_right, delta_t delta_right);
    __host__ __device__ delta_t delta(int index);

    __device__ void updateParents(int index);

    __host__ __device__ void traverse (int *left_child, int *right_child); // return the left and right children
    __host__ __device__ void traverse(float4 queryMin, float4 queryMax, CandidateList *candidates);
    __host__ __device__ void query(Ray &ray, CandidateList &candidates);
    __host__ __device__ void query (float4 &bbMinQuery, float4 &bbMaxQuery, CandidateList &candidates);

    // Setters
    __host__ __device__ void setRootIndex(int index) { root_index = index; };
    __host__ __device__ void setInternalNodes(Nodes nodes) { internal_nodes = nodes; };
    __host__ __device__ void setLeafNodes(Nodes nodes) { leaf_nodes = nodes; };
    __host__ __device__ void setSortedIndices(unsigned int *sorted_indices) { this->sorted_indices = sorted_indices; };
    __host__ __device__ void setKeys(morton_t *keys) { this->keys = keys; };
    __host__ __device__ void setNbKeys(unsigned int nb_keys) { this->nb_keys = nb_keys; };
    __host__ __device__ void setSceneBB(float4 bbMin, float4 bbMax) { this->bbMin = bbMin; this->bbMax = bbMax; };
    __host__ __device__ void setInternalNode(int parent, int left_child, int rope, float4 bbMin, float4 bbMax);
    __host__ __device__ void setLeafNode(int index, int rope);
    __host__ __device__ void setLeftChild(unsigned parent, int left_child) {
        this->internal_nodes.left_child[parent] = left_child;
    };
    __host__ __device__ void setBBMinLeaf(unsigned const index, float4 const bbMin) {
        leaf_nodes.bbMin[index] = bbMin;
    };
    __host__ __device__ void setBBMaxLeaf(unsigned const index, float4 const bbMax) {
        leaf_nodes.bbMax[index] = bbMax;
    };
    __host__ __device__ void setBBMinInternal(unsigned const index, float4 const bbMin) {
        internal_nodes.bbMin[index] = bbMin;
    };
    __host__ __device__ void setBBMaxInternal(unsigned const index, float4 const bbMax) {
        internal_nodes.bbMax[index] = bbMax;
    };

    // Getters
    __host__ __device__ int getRootIndex() { return root_index; };
    __host__ __device__ int getLeftChild(unsigned const index) {
        return this->internal_nodes.left_child[index];
    };
    __host__ __device__ float4 getBBMaxLeaf(unsigned const index) {
        return leaf_nodes.bbMax[index];
    };
    __host__ __device__ float4 getBBMinLeaf(unsigned const index) {
        return leaf_nodes.bbMin[index];
    };
    __host__ __device__ float4 getBBMaxInternal(unsigned const index) {
        return internal_nodes.bbMax[index];
    };
    __host__ __device__ float4 getBBMinInternal(unsigned const index) {
        return internal_nodes.bbMin[index];
    };
    __host__ __device__ int getRopeLeaf(unsigned const index) {
        return leaf_nodes.rope[index];
    };
    __host__ __device__ int getRopeInternal(unsigned const index) {
        return internal_nodes.rope[index];
    };

    __host__ __device__ Nodes getInternalNodes() { return internal_nodes; };
    __host__ __device__ Nodes getLeafNodes() { return leaf_nodes; };
    __host__ __device__ unsigned int *getSortedIndices() { return sorted_indices; };
    __host__ __device__ morton_t *getKeys() { return keys; };
    __host__ __device__ unsigned int getNbKeys() { return nb_keys; };
    __host__ __device__ float4 getSceneBBMin() { return bbMin; };
    __host__ __device__ float4 getSceneBBMax() { return bbMax; };

    // Helpers
    __host__ __device__ void printTree();
    __host__ __device__ bool isLeaf(int index) {
        return (0 <= index) && (index < this->nb_keys);
    };

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

private:
};
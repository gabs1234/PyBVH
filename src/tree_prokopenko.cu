#include "tree_prokopenko.cuh"
#include "Commons.cuh"

#include <set>

__host__ __device__ BVHTree::BVHTree(unsigned int nb_keys, morton_t *keys) {
    this->nb_keys = nb_keys;
    this->keys = keys;
}

__host__ __device__ BVHTree::BVHTree(unsigned int nb_keys, morton_t *keys, float4 *bbMinScene, float4 *bbMaxScene) {
    this->nb_keys = nb_keys;
    this->keys = keys;

    this->bbMin = bbMinScene[0];
    this->bbMax = bbMaxScene[0];
}

__host__ __device__ BVHTree::BVHTree(unsigned int nb_keys, morton_t *keys, float4 *bbMinScene, float4 *bbMaxScene,
    float4 *bbMinLeafs, float4 *bbMaxLeafs) {
    this->nb_keys = nb_keys;
    this->keys = keys;

    this->bbMin = bbMinScene[0];
    this->bbMax = bbMaxScene[0];
    
    this->leaf_nodes.bbMin = bbMinLeafs;
    this->leaf_nodes.bbMax = bbMaxLeafs;
}

__host__ __device__ void BVHTree::printTree () {
    printf ("BVHTree: %d\n", this->nb_keys);
    printf ("bbMin: %f %f %f\n", this->bbMin.x, this->bbMin.y, this->bbMin.z);
    printf ("bbMax: %f %f %f\n", this->bbMax.x, this->bbMax.y, this->bbMax.z);
    printf ("Leaf Nodes: %d\n", this->leaf_nodes.nb_nodes);
    printf ("Internal Nodes: %d\n", this->internal_nodes.nb_nodes);

    for (int i = 0; i < this->nb_keys; i++) {
        printf ("Rope: %d, %d |\n", i, this->leaf_nodes.rope[i]);
    }
    for (int i = 0; i < this->nb_keys; i++) {
        printf ("Morton keys: %d, %d |\n", i, this->keys[i]);
    }
    for (int i = 0; i < this->nb_keys; i++) {
        printf ("Sorted indices: %d, %d |\n", i, this->sorted_indices[i]);
    }
    for (int i = 0; i < this->nb_keys; i++) {
        printf ("Leaf bbmin: %d, %f, %f, %f |\n", i, this->leaf_nodes.bbMin[i].x, this->leaf_nodes.bbMin[i].y, this->leaf_nodes.bbMin[i].z);
    }
    for (int i = 0; i < this->nb_keys; i++) {
        printf ("Leaf bbmax: %d, %f, %f, %f |\n", i, this->leaf_nodes.bbMax[i].x, this->leaf_nodes.bbMax[i].y, this->leaf_nodes.bbMax[i].z);
    }

    printf ("\n");
}

// __host__ __device__ void BVHTree::growLeaf(int parent, int leaf) {
//     float4 bbMin = this->leaf_nodes.bbMin[leaf];
//     float4 bbMax = this->leaf_nodes.bbMax[leaf];

//     float4 bbMinParent = this->internal_nodes.bbMin[parent];
//     float4 bbMaxParent = this->internal_nodes.bbMax[parent];

//     this->internal_nodes.bbMin[parent].x = fminf(bbMin.x, bbMinParent.x);
//     this->internal_nodes.bbMin[parent].y = fminf(bbMin.y, bbMinParent.y);
//     this->internal_nodes.bbMin[parent].z = fminf(bbMin.z, bbMinParent.z);

//     this->internal_nodes.bbMax[parent].x = fmaxf(bbMax.x, bbMaxParent.x);
//     this->internal_nodes.bbMax[parent].y = fmaxf(bbMax.y, bbMaxParent.y);
//     this->internal_nodes.bbMax[parent].z = fmaxf(bbMax.z, bbMaxParent.z);
// }

// __host__ __device__ void BVHTree::growInternal(int parent, int child) {
//     float4 bbMinChild = this->internal_nodes.bbMin[child];
//     float4 bbMaxChild = this->internal_nodes.bbMax[child];

//     float4 bbMinParent = this->internal_nodes.bbMin[parent];
//     float4 bbMaxParent = this->internal_nodes.bbMax[parent];

//     this->internal_nodes.bbMin[parent].x = fminf(bbMinChild.x, bbMinParent.x);
//     this->internal_nodes.bbMin[parent].y = fminf(bbMinChild.y, bbMinParent.y);
//     this->internal_nodes.bbMin[parent].z = fminf(bbMinChild.z, bbMinParent.z);

//     this->internal_nodes.bbMax[parent].x = fmaxf(bbMaxChild.x, bbMaxParent.x);
//     this->internal_nodes.bbMax[parent].y = fmaxf(bbMaxChild.y, bbMaxParent.y);
//     this->internal_nodes.bbMax[parent].z = fmaxf(bbMaxChild.z, bbMaxParent.z);
// }

__host__ __device__ void BVHTree::growBox(float4 &bbMinInput, float4 &bbMaxInput, float4 *bbMinOutput, float4 *bbMaxOutput) {
    bbMinOutput->x = fminf(bbMinInput.x, bbMinOutput->x);
    bbMinOutput->y = fminf(bbMinInput.y, bbMinOutput->y);
    bbMinOutput->z = fminf(bbMinInput.z, bbMinOutput->z);

    bbMaxOutput->x = fmaxf(bbMaxInput.x, bbMaxOutput->x);
    bbMaxOutput->y = fmaxf(bbMaxInput.y, bbMaxOutput->y);
    bbMaxOutput->z = fmaxf(bbMaxInput.z, bbMaxOutput->z);
}

__host__ __device__ int BVHTree::delta(int index) {

    if (index < 0 || index >= this->nb_keys - 1) {
        return INT_MAX;
    }
    
    // TODO: augment the function if the codes are the same
    unsigned int a = this->keys[index];
    unsigned int b = this->keys[index + 1];
    int x = a ^ b;
    return x + (!x) * (INT_MIN + (index ^ (index + 1))) - 1; // 
}

__host__ __device__ void BVHTree::setRope(Nodes *nodes, unsigned int skip_index, int range_right, delta_t delta_right) {
    int rope;

    if (range_right != this->nb_keys - 1) {
        int r = range_right + 1;
        rope = delta_right < this->delta(r) ? r : this->toInternalRepresentation(r);
    }
    else {
        rope = SENTINEL;
    }
    nodes->rope[skip_index] = rope;
}

__host__ __device__ void BVHTree::setInternalRope(unsigned int skip_index, int range_right, delta_t delta_right) {
    Nodes *nodes = &this->internal_nodes;
    this->setRope(nodes, skip_index, range_right, delta_right);
}

__host__ __device__ void BVHTree::setLeafRope(unsigned int skip_index, int range_right, delta_t delta_right) {
    Nodes *nodes = &this->leaf_nodes;
    this->setRope(nodes, skip_index, range_right, delta_right);
}

__device__ void BVHTree::updateParents(int i) {
    int range_left = i;
    int range_right = i;
    delta_t delta_left = this->delta(i - 1);
    delta_t delta_right = this->delta(i);

    // printf ("delta_left: %d, delta_right: %d\n", delta_left, delta_right);

    float4 bbMinCurrent = this->getBBMinLeaf (i);
    float4 bbMaxCurrent = this->getBBMaxLeaf (i);

    this->setLeafRope(i, range_right, delta_right);

    unsigned const root = this->toInternalRepresentation(0);

    do {
        int left_child;
        int right_child;
        if (delta_right < delta_left) {
            const int apetrei_parent = range_right;

            range_right = atomicCAS (&(this->internal_nodes.entered[this->toInternalRepresentation(apetrei_parent)]), INVALID, range_left);
            // printf ("Apetrei parent: %d\n", apetrei_parent);
            if (range_right == INVALID) {
                return;
            }
            delta_right = this->delta(range_right);

            left_child = i;

            right_child = apetrei_parent + 1;
            bool const right_is_leaf = (right_child == range_right);

            // Memory sync
            __threadfence();

            if (right_is_leaf) {
                float4 bbMinRight = this->getBBMinLeaf (right_child);
                float4 bbMaxRight = this->getBBMaxLeaf (right_child);
                this->growBox(bbMinRight, bbMaxRight, &bbMinCurrent, &bbMaxCurrent);
                // print the bbmin
                // printf ("in tree bbmin rl: %f %f %f\n", bbMinRight.x, bbMinRight.y, bbMinRight.z);
            }
            else {
                right_child = this->toInternalRepresentation(right_child);
                float4 bbMinRight = this->getBBMinInternal (right_child);
                float4 bbMaxRight = this->getBBMaxInternal (right_child);
                this->growBox(bbMinRight, bbMaxRight, &bbMinCurrent, &bbMaxCurrent);
                // printf ("in tree bbmin ll: %f %f %f\n", bbMinRight.x, bbMinRight.y, bbMinRight.z);

            }
        }
        else {
            int const apetrei_parent = range_left - 1;
            range_left = atomicCAS (&(this->internal_nodes.entered[this->toInternalRepresentation(apetrei_parent)]), INVALID, range_right);
            // printf ("Apetrei parent: %d\n", apetrei_parent);

            if (range_left == INVALID){
                return;
            }

            delta_left = this->delta(range_left - 1);

            left_child = apetrei_parent;
            bool const left_is_leaf = (left_child == range_left);

            // Memory sync
            __threadfence();
            
            if (left_is_leaf) {
                float4 bbMinLeft = this->getBBMinLeaf ((left_child));
                float4 bbMaxLeft = this->getBBMaxLeaf ((left_child));
                this->growBox(bbMinLeft, bbMaxLeft, &bbMinCurrent, &bbMaxCurrent);
            }
            else {
                left_child = this->toInternalRepresentation(left_child);
                float4 bbMinLeft = this->getBBMinInternal (left_child);
                float4 bbMaxLeft = this->getBBMaxInternal (left_child);
                this->growBox(bbMinLeft, bbMaxLeft, &bbMinCurrent, &bbMaxCurrent);

            }
        }

        int karras_parent = delta_right < delta_left ? range_right : range_left;
        karras_parent = this->toInternalRepresentation(karras_parent);
        // printf ("karras_parent: %d\n", karras_parent);
        // if (karras_parent == range_left) {
        //     left_child = i;
        // }
        // else {
        //     left_child = this->toInternalRepresentation(i);
        // }
        // __threadfence();

        if (left_child < 0) {
            printf ("Error: Left child is negative: %d\n", left_child);
        }

        if (karras_parent < 0) {
            printf ("Error: karras_parent is negative: %d\n", karras_parent);
        }

        this->setLeftChild(karras_parent, left_child);
        // this->internal_nodes.right_child[karras_parent] = right_child;
        this->setBBMinInternal(karras_parent, bbMinCurrent);
        this->setBBMaxInternal(karras_parent, bbMaxCurrent);
        this->setInternalRope(karras_parent, range_right, delta_right);

        // printf ("INDEX, %d, internal rope: %d\n", index, inodes->rope[karras_parent]);

        i = karras_parent;

        if (i < 0) {
            printf ("Error: i is negative: %d\n", i);
        }
    }
    while (i != root);
    
    return;
}

__host__ __device__ bool overlapingAABBs(float4 bbMin1, float4 bbMax1, float4 bbMin2, float4 bbMax2) {
    if (bbMin1.x > bbMax2.x || bbMax1.x < bbMin2.x) {
        return false;
    }
    if (bbMin1.y > bbMax2.y || bbMax1.y < bbMin2.y) {
        return false;
    }
    if (bbMin1.z > bbMax2.z || bbMax1.z < bbMin2.z) {
        return false;
    }
    return true;
}

// __device__ void BVHTree::traverse(float4 queryMin, float4 queryMax, CollisionList *candidates) {
//     int current_node = this->toInternalRepresentation(0);

//     do {
//         if (this->isLeaf(current_node)) {
//             float4 bbMin = this->leaf_nodes.bbMin[current_node];
//             float4 bbMax = this->leaf_nodes.bbMax[current_node];

//             if (overlapingAABBs(queryMin, queryMax, bbMin, bbMax)) {
//                 if (candidates->count == MAX_COLLISIONS) {
//                     return;
//                 }
//                 candidates->collisions[candidates->count++] = this->permuteIndex(current_node);
//             }
//             else {
//                 current_node = this->leaf_nodes.rope[current_node];
//             }
//         }
//         else {
//             current_node -= this->nb_keys;
//             float4 bbMin = this->internal_nodes.bbMin[current_node];
//             float4 bbMax = this->internal_nodes.bbMax[current_node];

//             if (overlapingAABBs(queryMin, queryMax, bbMin, bbMax)) {
//                 current_node = this->internal_nodes.left_child[current_node];
//             }
//             else {
//                 current_node = this->internal_nodes.rope[current_node];
//             }
//         }
//     }
//     while (current_node != SENTINEL);
// }

// __host__ __device__ void BVHTree::traverse (int *left_child, int *right_child) {
//     int current_node = this->toInternalRepresentation(0);

//     int left = -1;
//     int right = -1;
    
//     do {
//         if (this->isLeaf(current_node)) {
//             current_node = this->leaf_nodes.rope[current_node];
//             right = current_node;
            
//         }
//         else {
//             current_node -= this->nb_keys;
//             current_node =  this->internal_nodes.left_child[current_node];
//             left = current_node;
//         }

//     } while (current_node != SENTINEL);
// }

__host__ __device__ void BVHTree::query (Ray &ray, CandidateList &candidates) {
    int current_node = this->toInternalRepresentation(0);
    
    do {
        float4 bbMax = this->getBBMaxLeaf ((current_node));
        float4 bbMin = this->getBBMinLeaf ((current_node));
        bool intersects = ray.intersects(bbMin, bbMax);

        if (intersects) {
            if (this->isLeaf(current_node)) {
                if (candidates.count == MAX_COLLISIONS) {
                    return;
                }
                candidates.collisions[candidates.count++] = current_node;
                current_node = this->getRopeLeaf(current_node);
            }
            else {
                current_node = this->getLeftChild(current_node);
            }
        }
        else {
            current_node = this->getRopeLeaf(current_node);
        }
    }
    while (current_node != SENTINEL);
}


// __host__ __device__ void BVHTree::query (float4 &bbMinQuery, float4 &bbMaxQuery, CandidateList &candidates) {
//     int current_node = this->toInternalRepresentation(0);

//     // printf ("Current node init: %d\n", current_node);

//     // ray.print();

//     float4 bbMin;
//     float4 bbMax;

//     do {
//         bool isLeaf = this->isLeaf(current_node);
//         if (isLeaf) {
//             bbMin = this->getBBMinLeaf (current_node);
//             bbMax = this->getBBMaxLeaf (current_node);
//         }
//         else {
//             bbMin = this->getBBMinInternal (current_node);
//             bbMax = this->getBBMaxInternal (current_node);
//         }
        
//         float2 t;
//         if (overlapingAABBs (bbMinQuery, bbMaxQuery, bbMin, bbMax)) {
//             // printf ("Current node: %d\n", current_node);
//             if (isLeaf) {
//                 if (candidates.count == MAX_COLLISIONS) {
//                     return;
//                 }
//                 candidates.collisions[candidates.count++] = this->permuteIndex(current_node);
//                 if(current_node >= this->nb_keys || current_node < 0) {
//                     // printf ("2Error: Current node out of bounds: %d\n", current_node);
//                     return;
//                 }
//                 current_node = this->leaf_nodes.rope[current_node];
//             }
//             else {
//                  if(current_node >= this->nb_keys || current_node < 0) {
//                     // printf ("3Error: Current node out of bounds: %d\n", current_node);
//                     return;
//                 }
//                 current_node = this->internal_nodes.left_child[current_node];
//             }
//             // printf ("Current node: %d\n", current_node);
//         }
//         else {
//             if(current_node >= this->nb_keys || current_node < 0) {
//                 // printf ("Error: Current node out of bounds: %d\n", current_node);
//                 return;
//             }
//             current_node = this->internal_nodes.rope[current_node];
//         }
        
//     }
//     while (current_node != SENTINEL);

// }

/**
 * Descend the tree and explore all leafs
 * Assert the the number of leafs is equal to the number of keys
 * 
 * 
 */
__host__ bool BVHTree::sanityCheck(int &id) {
    int root = this->toInternalRepresentation(0);

    int nb_leafs = 0;

    std::set<int> leafs;
    std::pair<std::set<int>::iterator, bool> ret;

    int current_node = root;
    // printf ("Root: %d\n", root);

    do {
        // printf ("Current node: %d\n", current_node);

        if (current_node < 0) {
            id = current_node;
            return false;
        }

        if (this->isLeaf(current_node)) {
            nb_leafs++;
            
            ret = leafs.insert(current_node);

            if (ret.second == false) {
                id = current_node;
                return false;
            }

            current_node = this->getRopeLeaf(current_node);
        }
        else {
            current_node = this->getLeftChild(this->toOriginalRepresentation(current_node));
        }

        
    }
    while (current_node != SENTINEL);

    return nb_leafs == this->nb_keys;
}
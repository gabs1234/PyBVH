#include "tree_prokopenko.cuh"
#include "Commons.cuh"

__device__ BVHTree::BVHTree(unsigned int nb_keys, morton_t *keys) {
    this->nb_keys = nb_keys;
    this->keys = keys;
}

__device__ BVHTree::BVHTree(unsigned int nb_keys, morton_t *keys, float4 *bbMinScene, float4 *bbMaxScene) {
    this->nb_keys = nb_keys;
    this->keys = keys;

    this->bbMin = bbMinScene[0];
    this->bbMax = bbMaxScene[0];
}

__device__ BVHTree::BVHTree(unsigned int nb_keys, morton_t *keys, float4 *bbMinScene, float4 *bbMaxScene,
    float4 *bbMinLeafs, float4 *bbMaxLeafs) {
    this->nb_keys = nb_keys;
    this->keys = keys;

    this->bbMin = bbMinScene[0];
    this->bbMax = bbMaxScene[0];
    
    this->leaf_nodes.bbMin = bbMinLeafs;
    this->leaf_nodes.bbMax = bbMaxLeafs;
}

__device__ void BVHTree::printTree () {
    printf ("BVHTree: %d\n", this->nb_keys);
    printf ("bbMin: %f %f %f\n", this->bbMin.x, this->bbMin.y, this->bbMin.z);
    printf ("bbMax: %f %f %f\n", this->bbMax.x, this->bbMax.y, this->bbMax.z);
    printf ("Leaf Nodes: %d\n", this->leaf_nodes.nb_nodes);
    printf ("Internal Nodes: %d\n", this->internal_nodes.nb_nodes);

    for (int i = 0; i < this->nb_keys; i++) {
        printf ("Entered: %d, %d |\n", i, this->internal_nodes.entered[i]);
    }
    printf ("\n");

    for (int i = 0; i < this->nb_keys; i++) {
        printf ("Rope: %d, %d |\n", i, this->leaf_nodes.rope[i]);
    }
    printf ("\n");

    for (int i = 0; i < this->nb_keys; i++) {
        printf ("Morton keys: %d, %d |\n", i, this->keys[i]);
    }

    for (int i = 0; i < this->nb_keys; i++) {
        printf ("Sorted indices: %d, %d |\n", i, this->sorted_indices[i]);
    }
    printf ("\n");
}

__device__ int BVHTree::internal_index(int index) {
    return index + this->nb_keys;    
}

__device__ bool BVHTree::isLeaf(int index) {
    // printf ("Index: %d\n", index);
    return index < this->nb_keys;
}
__device__ void BVHTree::projectKeys(int index) {
    // Calculate the centroid of the AABB
    float4 centroid = getBoundingBoxCentroid(this->leaf_nodes.bbMin[index], this->leaf_nodes.bbMax[index]);
    
    float4 normalizedCentroid = normalize(centroid, this->bbMin, this->bbMax);

    // Calculate the morton code of the triangle
    morton_t mortonCode = calculateMortonCode(normalizedCentroid);

    // Store the morton code
    this->keys[index] = mortonCode;
}

__device__ void BVHTree::growLeaf(int parent, int leaf) {
    float4 bbMin = this->leaf_nodes.bbMin[leaf];
    float4 bbMax = this->leaf_nodes.bbMax[leaf];

    float4 bbMinParent = this->internal_nodes.bbMin[parent];
    float4 bbMaxParent = this->internal_nodes.bbMax[parent];

    this->internal_nodes.bbMin[parent].x = fminf(bbMin.x, bbMinParent.x);
    this->internal_nodes.bbMin[parent].y = fminf(bbMin.y, bbMinParent.y);
    this->internal_nodes.bbMin[parent].z = fminf(bbMin.z, bbMinParent.z);

    this->internal_nodes.bbMax[parent].x = fmaxf(bbMax.x, bbMaxParent.x);
    this->internal_nodes.bbMax[parent].y = fmaxf(bbMax.y, bbMaxParent.y);
    this->internal_nodes.bbMax[parent].z = fmaxf(bbMax.z, bbMaxParent.z);
}

__device__ void BVHTree::growInternal(int parent, int child) {
    float4 bbMinChild = this->internal_nodes.bbMin[child];
    float4 bbMaxChild = this->internal_nodes.bbMax[child];

    float4 bbMinParent = this->internal_nodes.bbMin[parent];
    float4 bbMaxParent = this->internal_nodes.bbMax[parent];

    this->internal_nodes.bbMin[parent].x = fminf(bbMinChild.x, bbMinParent.x);
    this->internal_nodes.bbMin[parent].y = fminf(bbMinChild.y, bbMinParent.y);
    this->internal_nodes.bbMin[parent].z = fminf(bbMinChild.z, bbMinParent.z);

    this->internal_nodes.bbMax[parent].x = fmaxf(bbMaxChild.x, bbMaxParent.x);
    this->internal_nodes.bbMax[parent].y = fmaxf(bbMaxChild.y, bbMaxParent.y);
    this->internal_nodes.bbMax[parent].z = fmaxf(bbMaxChild.z, bbMaxParent.z);
}

__device__ void BVHTree::growBox(float4 &bbMinInput, float4 &bbMaxInput, float4 *bbMinOutput, float4 *bbMaxOutput) {
    bbMinOutput->x = fminf(bbMinInput.x, bbMinOutput->x);
    bbMinOutput->y = fminf(bbMinInput.y, bbMinOutput->y);
    bbMinOutput->z = fminf(bbMinInput.z, bbMinOutput->z);

    bbMaxOutput->x = fmaxf(bbMaxInput.x, bbMaxOutput->x);
    bbMaxOutput->y = fmaxf(bbMaxInput.y, bbMaxOutput->y);
    bbMaxOutput->z = fmaxf(bbMaxInput.z, bbMaxOutput->z);
}

__device__ int BVHTree::delta(int index) {
    // int sorted_index = index;

    if (index < 0 || index >= this->nb_keys - 1) {
        // return max unsigned int
        return MAX_INT;
    }

    
    // TODO: augment the function if the codes are the same
    unsigned int a = this->keys[index];
    unsigned int b = this->keys[index+1];
    int x = a ^ b;
    return x + (!x) * (MIN_INT + (index ^ (index + 1))) - 1; // 
}

__device__ void BVHTree::setRope(Nodes *nodes, unsigned int skip_index, int range_right, delta_t delta_right) {
    int rope;

    if (range_right != this->nb_keys - 1) {
        int r = range_right + 1;
        rope = delta_right < this->delta(r) ? r : this->internal_index(r);
    }
    else {
        rope = SENTINEL;
    }
    nodes->rope[skip_index] = rope;
}

__device__ int BVHTree::getRope(int index, int range_right, delta_t delta_right) {
    int rope;

    if (range_right != this->nb_keys - 1) {
        int r = range_right + 1;
        rope = delta_right < this->delta(r) ? r : this->internal_index(r);
    }
    else {
        rope = SENTINEL;
    }
    return rope;
}

__device__ void BVHTree::setInternalNode(int parent, int child_left, int rope, float4 bbMin, float4 bbMax) {
    if (parent >= this->internal_nodes.nb_nodes) {
        printf ("Error: Parent index out of bounds: %d\n", parent);
        return;
    }
    this->internal_nodes.left_child[parent] = child_left;
    this->internal_nodes.rope[parent] = rope;
    this->internal_nodes.bbMin[parent] = bbMin;
    this->internal_nodes.bbMax[parent] = bbMax;
}

__device__ void BVHTree::setLeafNode(int index, int rope) {
    if (index >= this->leaf_nodes.nb_nodes) {
        printf ("Error: Leaf index out of bounds: %d\n", index);
        return;
    }
    this->leaf_nodes.rope[index] = rope;
}

__device__ void BVHTree::updateParents(int index) {
    Nodes *lnodes = &(this->leaf_nodes);
    Nodes *inodes = &(this->internal_nodes);

    int i = index;

    int range_left = index;
    int range_right = index;
    delta_t delta_left = this->delta(index - 1);
    delta_t delta_right = this->delta(index);

    // printf ("delta_left: %d, delta_right: %d\n", delta_left, delta_right);

    float4 bbMinCurrent = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 bbMaxCurrent = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float4 bbMin = this->leaf_nodes.bbMin[i];
    float4 bbMax = this->leaf_nodes.bbMax[i];

    this->growBox(bbMin, bbMax, &bbMinCurrent, &bbMaxCurrent);

    this->setRope(lnodes, i, range_right, delta_right);

    int const root = this->internal_index(0);

    do {
        int left_child;
        int right_child;
        if (delta_right < delta_left) {
            const int apetrei_parent = range_right;

            range_right = atomicCAS(&(this->internal_nodes.entered[apetrei_parent]), INVALID, range_left);
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
                float4 bbMinRight = lnodes->bbMin[right_child];
                float4 bbMaxRight = lnodes->bbMax[right_child];
                this->growBox(bbMinRight, bbMaxRight, &bbMinCurrent, &bbMaxCurrent);
            }
            else {
                float4 bbMinRight = inodes->bbMin[right_child];
                float4 bbMaxRight = inodes->bbMax[right_child];
                this->growBox(bbMinRight, bbMaxRight, &bbMinCurrent, &bbMaxCurrent);
                // right_child = this->internal_index(right_child);
            }
        }
        else {
            int const apetrei_parent = range_left - 1;
            range_left = atomicCAS(&(this->internal_nodes.entered[apetrei_parent]), INVALID, range_right);
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
                float4 bbMinLeft = lnodes->bbMin[left_child];
                float4 bbMaxLeft = lnodes->bbMax[left_child];
                this->growBox(bbMinLeft, bbMaxLeft, &bbMinCurrent, &bbMaxCurrent);
            }
            else {
                float4 bbMinLeft = inodes->bbMin[left_child];
                float4 bbMaxLeft = inodes->bbMax[left_child];
                this->growBox(bbMinLeft, bbMaxLeft, &bbMinCurrent, &bbMaxCurrent);
                left_child = this->internal_index(left_child);
            }
        }

        int const karras_parent = delta_right < delta_left ? range_right : range_left;
        // printf ("karras_parent: %d\n", karras_parent);
        // if (karras_parent == range_left) {
        //     left_child = i;
        // }
        // else {
        //     left_child = this->internal_index(i);
        // }
        // __threadfence();

        inodes->left_child[karras_parent] = left_child;
        // inodes->right_child[karras_parent] = right_child;
        inodes->bbMin[karras_parent] = bbMinCurrent;
        inodes->bbMax[karras_parent] = bbMaxCurrent;
        this->setRope(inodes, karras_parent, range_right, delta_right);

        // printf ("INDEX, %d, internal rope: %d\n", index, inodes->rope[karras_parent]);

        i = this->internal_index(karras_parent);
    }
    while (i != root);
    
    return;
}

__device__ bool overlapingAABBs(float4 bbMin1, float4 bbMax1, float4 bbMin2, float4 bbMax2) {
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
//     int current_node = this->internal_index(0);

//     do {
//         if (this->isLeaf(current_node)) {
//             float4 bbMin = this->leaf_nodes.bbMin[current_node];
//             float4 bbMax = this->leaf_nodes.bbMax[current_node];

//             if (overlapingAABBs(queryMin, queryMax, bbMin, bbMax)) {
//                 if (candidates->count == MAX_COLLISIONS) {
//                     return;
//                 }
//                 candidates->collisions[candidates->count++] = this->original_index(current_node);
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

__device__ void BVHTree::query (Ray &ray, CollisionList &candidates) {
    int current_node = this->internal_index(0);

    do {
        if (this->isLeaf(current_node)) {
            // printf ("Current node leaf: %d\n", current_node);

            float4 bbMin = this->leaf_nodes.bbMin[current_node];
            float4 bbMax = this->leaf_nodes.bbMax[current_node];
            float2 t;
            if (ray.intersects(bbMin, bbMax, t)) {
                if (candidates.count == MAX_COLLISIONS) {
                    printf ("Max collisions reached %d\n", current_node);
                    return;
                }
                candidates.collisions[candidates.count++] = this->original_index(current_node);
            }
            current_node = this->leaf_nodes.rope[current_node];
        }
        else {
            int ajdusted = current_node - this->nb_keys;
            float4 bbMin = this->internal_nodes.bbMin[ajdusted];
            float4 bbMax = this->internal_nodes.bbMax[ajdusted];
            // printf ("Current node internal: %d\n", current_node);
            float2 t;
            if (ray.intersects(bbMin, bbMax, t)) {
                // printf ("Current node inter: %d\n", current_node);
                current_node = this->internal_nodes.left_child[ajdusted];
            }
            else {
                // printf ("Current node descend: %d\n", current_node);
                current_node = this->internal_nodes.rope[ajdusted];
            }
        }
        // printf ("Current node: %d\n", current_node);
    }
    while (current_node != SENTINEL);
}

__device__ void BVHTree::query (float4 &bbMinQuery, float4 &bbMaxQuery, CollisionList &candidates) {
    int current_node = this->internal_index(0);

    // printf ("Current node init: %d\n", current_node);

    // ray.print();

    float4 bbMin;
    float4 bbMax;

    do {
        bool isLeaf = this->isLeaf(current_node);
        if (isLeaf) {
            if(current_node >= this->nb_keys || current_node < 0) {
                // printf ("0Error: Current node out of bounds: %d\n", current_node);
                return;
            }
            bbMin = this->leaf_nodes.bbMin[current_node];
            bbMax = this->leaf_nodes.bbMax[current_node];
        }
        else {
            current_node -= this->nb_keys;
            if(current_node >= 2*this->nb_keys || current_node < 0) {
                // printf ("1Error: Current node internal out of bounds: %d\n", current_node);
                return;
            }
            bbMin = this->internal_nodes.bbMin[current_node];
            bbMax = this->internal_nodes.bbMax[current_node];
        }
        
        float2 t;
        if (overlapingAABBs (bbMinQuery, bbMaxQuery, bbMin, bbMax)) {
            // printf ("Current node: %d\n", current_node);
            if (isLeaf) {
                if (candidates.count == MAX_COLLISIONS) {
                    return;
                }
                candidates.collisions[candidates.count++] = this->original_index(current_node);
                 if(current_node >= this->nb_keys || current_node < 0) {
                    // printf ("2Error: Current node out of bounds: %d\n", current_node);
                    return;
                }
                current_node = this->leaf_nodes.rope[current_node];
            }
            else {
                 if(current_node >= this->nb_keys || current_node < 0) {
                    // printf ("3Error: Current node out of bounds: %d\n", current_node);
                    return;
                }
                current_node = this->internal_nodes.left_child[current_node];
            }
            // printf ("Current node: %d\n", current_node);
        }
        else {
            if(current_node >= this->nb_keys || current_node < 0) {
                // printf ("Error: Current node out of bounds: %d\n", current_node);
                return;
            }
            current_node = this->internal_nodes.rope[current_node];
        }
        
    }
    while (current_node != SENTINEL);

}
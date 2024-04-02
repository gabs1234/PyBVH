#include "RayTracer.cuh"
#include "Ray.cuh"
#include "tree_prokopenko.cuh"
#include "Commons.cuh"

__device__ RayTracer::RayTracer(BVHTree *tree, float4 *vertices, unsigned int nbVertices, bool parallel) {
    this->tree = tree;
    this->vertices = vertices;
    this->nbVertices = nbVertices;
    this->raySet = 0;
    this->parallelGeometry = parallel;
}

__device__ float4 RayTracer::sphericalToCartesian(float theta, float phi, float r) {
    float const x = r * sin(theta) * cos(phi);
    float const y = r * sin(theta) * sin(phi);
    float const z = r * cos(theta);

    return make_float4(x, y, z, 1.0);
}

/**
 * Let S be the surface defined by the equation:
 * forall I in S, RI * RO = 0. 
 * Where:
 * - O is the origin of our frame of reference. 
 * - R the origin of the ray (i.e. RO the ray vector)
 * Let (Ix, Iy) be the coordinates of the a point I 
 * in the frame of reference of the image plane. We wish
 * to find the coordinates (x, y, z) of the point I in the
 * frame of reference of the surface S, such that:
 * (RI . RO) = 0 (i.e. I is in S)
 * 
*/
__device__ float4 RayTracer::computePointInPlane(float2 &I, float4 &O, float4 &R) {
    float4 RO = make_float4(O.x - R.x, O.y - R.y, O.z - R.z, 0.0);
    float2 IR = make_float2(R.x - I.x, R.y - I.y);
    float4 I_3d;
    I_3d.x = I.x;
    I_3d.y = I.y;
    I_3d.z = (IR.x * RO.x + IR.y * RO.y) / RO.z + R.z;

    return I_3d;
}

__device__ bool RayTracer::computeRayAABB(float4 &O, float4 &min, float4 &max) {
    float4 sceneBBMin = this->tree->getSceneBBMin();
    float4 sceneBBMax = this->tree->getSceneBBMax();
    float4 rayBBmin, rayBBmax;
    float2 t;
    
    if (this->ray.intersects(sceneBBMin, sceneBBMax, t)) {
        min = this->ray.computeParametric(t.x);
        max = this->ray.computeParametric(t.y);
        return true;
    }
    return false;
}

// __device__ void RayTracer::computeManifoldThickness (float *image, float *normals, CollisionList &t_values) {
    
// } 

__device__ float computeThickness(CollisionList &tvalues) {
    

    int i, j;
    float result = 0.0;

    i = 0;
    while (i < tvalues.count) {
        j = i + 1;
        while (j < tvalues.count && fabsf (tvalues.collisions[j] - tvalues.collisions[i]) < 0.00001) {
            j++;
        }
        if (i < tvalues.count && j < tvalues.count) {
            result += fabsf (tvalues.collisions[j] - tvalues.collisions[i]);
        }
        i = j + 1;
    }

    return result;
}

__device__ float sumTvalues (CollisionList &t_values) {
    float thickness = 0;
    for (int i = 0; i < t_values.count; i++) {
        thickness += t_values.collisions[i];
    }
    return thickness;
}

__device__ float RayTracer::traceRayParallel(float2 P, float theta, float phi, float r, float4 origin, CollisionList &t_values) {
    // int old_val;
    // if (atomicCAS(&this->raySet, 0, 1) == 0) {
    //     this->reference_origin = origin;
    //     this->tail = this->sphericalToCartesian(theta, phi, r);
    //     this->reference_direction.x = this->reference_origin.x - this->tail.x;
    //     this->reference_direction.y = this->reference_origin.y - this->tail.y;
    //     this->reference_direction.z = this->reference_origin.z - this->tail.z;
    //     float norm = sqrt(this->reference_direction.x * this->reference_direction.x + this->reference_direction.y * this->reference_direction.y + this->reference_direction.z * this->reference_direction.z);
    //     this->reference_direction.x /= norm;
    //     this->reference_direction.y /= norm;
    //     this->reference_direction.z /= norm;

    // }

    // __syncthreads();
    
    

    // printf ("Direction (%f, %f, %f)\n", direction.x, direction.y, direction.z);

    float4 tail = this->sphericalToCartesian(theta, phi, r);
    
    float4 direction;
    direction.x = origin.x - tail.x;
    direction.y = origin.y - tail.y;
    direction.z = origin.z - tail.z;
    float norm = sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    direction.x /= norm;
    direction.y /= norm;
    direction.z /= norm;
    
    // Set the the ray origin to the surface point
    float4 I = this->computePointInPlane(P, origin, tail);

    // printf ("Direction (%f, %f, %f)\n", this->reference_direction.x, this->reference_direction.y, this->reference_direction.z);

    Ray displaced_ray = Ray(I, direction);

    // displaced_ray.print();

    CollisionList candidates;
    candidates.count = 0;
    memset(candidates.collisions, 0, MAX_COLLISIONS * sizeof(float));

    // This is where the acceleration structure (BVH) is actually usefull
    this->tree->query(displaced_ray, candidates);

    // for (int i = 0; i < candidates.count; i++) {
    //     t_values.collisions[t_values.count++] += 1;
    // }

    // Test the candidates for actual intersections
    for (int i = 0; i < candidates.count; i++) {
        int primIndex = candidates.collisions[i]*3;

        if (primIndex + 2>= this->nbVertices || primIndex < 0) {
            printf ("Invalid index %d\n", primIndex);
        }
        
        // printf("Collision at %d\n", primIndex);
        float4 V1 = this->vertices[primIndex];
        float4 V2 = this->vertices[primIndex + 1];
        float4 V3 = this->vertices[primIndex + 2];

        float t;
        if (displaced_ray.intersects(V1, V2, V3, t)) {
            // printf("real Collision at %d, %f\n", primIndex, t);
            t_values.collisions[t_values.count++] = t;
        }
        else {
            // printf("No Collision at %d\n", primIndex);
        }
    }

    // Print the t_values


    // Sort the t_values
    thrust::sort(thrust::device, t_values.collisions, t_values.collisions + t_values.count);

    // printf ("t_values count = %d\n", t_values.count);
    // for (int i = 0; i < t_values.count; i++) {
    //     printf ("t_values[%d] = %f\n", i, t_values.collisions[i]);
    // }

    // compute the thickness
    return sumTvalues(t_values);
}
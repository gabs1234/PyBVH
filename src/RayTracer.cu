#include "RayTracer.cuh"

// #include <thrust/sort.h>

__host__ __device__ RayTracer::RayTracer(BVHTree *tree, float4 *vertices, unsigned int nb_vertices) {
    this->tree = tree;
    this->vertices = vertices;
    this->nbVertices = nb_vertices;
    this->raySet = 0;
}

__host__ __device__ RayTracer::RayTracer(BVHTree *tree, float4 origin, float4 *vertices, unsigned int nb_vertices, bool parallel) {
    this->tree = tree;
    this->origin = origin;
    this->vertices = vertices;
    this->nbVertices = nb_vertices;
    this->raySet = 0;
    this->parallelGeometry = parallel;
}

__host__ __device__ float4 RayTracer::sphericalToCartesian(float theta, float phi, float r) {
    float const x = r * sin(theta) * cos(phi);
    float const y = r * sin(theta) * sin(phi);
    float const z = r * cos(theta);

    return make_float4(x, y, z, 1.0);
}



// __host__ __device__ bool RayTracer::computeRayAABB(float4 &O, float4 &min, float4 &max) {
//     float4 sceneBBMin = this->tree->getSceneBBMin();
//     float4 sceneBBMax = this->tree->getSceneBBMax();

//     if (this->ray.intersects(sceneBBMin, sceneBBMax)) {
//         min = this->ray.computeParametric(t.x);
//         max = this->ray.computeParametric(t.y);
//         return true;
//     }
//     return false;
// }

__host__ __device__ BasisNamespace::Basis RayTracer::makeProjectionBasis (BasisNamespace::Basis &MeshBasis, float4 &spherical, float4 &euler) {
    // Precompute cos and sin values
    float cos_theta = cos(spherical.x);
    float cos_phi = cos(spherical.y);
    float sin_theta = sin(spherical.x);
    float sin_phi = sin(spherical.y);

    // Compute the BasisNamespace::Basisbasis vectors
    float4 w = make_float4(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta, 0.0);
    float4 u = make_float4(cos_phi * cos_theta, sin_phi * cos_theta, -sin_theta, 0.0);
    float4 v = make_float4(-sin_phi, cos_phi, 0.0, 0.0);

    // Get origin of the BasisNamespace::Basisbasis
    float4 meshOrigin = MeshBasis.getOrigin();
    float4 new_origin;

    new_origin.x = meshOrigin.x + spherical.z * w.x;
    new_origin.y = meshOrigin.y + spherical.z * w.y;
    new_origin.z = meshOrigin.z + spherical.z * w.z;
    new_origin.w = 0;

    w.x = -w.x;
    w.y = -w.y;
    w.z = -w.z;
    u.x = -u.x;
    u.y = -u.y;
    u.z = -u.z;
    v.x = -v.x;
    v.y = -v.y;
    v.z = -v.z;

    // Create the new BasisNamespace::Basisbasis
    BasisNamespace::Basis new_basis = BasisNamespace::Basis(new_origin, u, v, w);

    // Rotate the BasisNamespace::Basisbasis
    new_basis.rotate(euler);

    return new_basis;
}

// __host__ __device__ float computeThickness(CollisionList &tvalues) {
//     float result = 0.0;

//     float epsilon = 0.0000001f;
    
//     for (int i = 0; i < tvalues.count; i++) {
//         result += tvalues.collisions[i];
//     }

//     return result;
// }

__host__ __device__ float computeThickness(CollisionList &tvalues) {
    float result = 0.0;
    float epsilon = 1e-6;
    int i = 0, j = 1;
    if (tvalues.count == 0) {
        return 0.0;
    }
    
    float t1 = tvalues.collisions[i];
    while (j < tvalues.count) {
        float t2 = tvalues.collisions[j];
        float d = fabsf (t2 - t1);
        if (d > epsilon){
            result += d;
            t1 = t2;
        }
        j++;
    }

    return result;
}

// __host__ __device__ float computeThickness(CollisionList &tvalues) {
//     float result = 0.0;
//     for (int i = 0; i < tvalues.count / 2; i++) {
//         result += tvalues.collisions[i] - tvalues.collisions[tvalues.count - 1 - i];
//     }
//     return result;
// }

__host__ __device__ float sumTvalues (CollisionList &t_values) {
    float thickness = 0;
    for (int i = 0; i < t_values.count; i++) {
        thickness += t_values.collisions[i];
    }
    return thickness;
}

__device__ float RayTracer::traceRayParallel(Ray &ray) {
    

    CandidateList candidates;
    candidates.count = 0;
    memset(candidates.collisions, 0, MAX_COLLISIONS * sizeof(int));

    CollisionList tvalues;
    tvalues.count = 0;
    memset(tvalues.collisions, 0, MAX_COLLISIONS * sizeof(float));

    // This is where the acceleration structure (BVH) is actually usefull
    this->tree->query(ray, candidates);

    if (candidates.count == 0) {
        return 0.0;
    }

    // printf ("candidates count = %d\n", candidates.count);

    // Test the candidates for actual intersections
    for (int i = 0; i < candidates.count; i++) {
        int primIndex = candidates.collisions[i]*3;
        
        // printf("Collision at %d\n", primIndex);
        float4 V1 = this->vertices[primIndex];
        float4 V2 = this->vertices[primIndex + 1];
        float4 V3 = this->vertices[primIndex + 2];

        float t;
        if (ray.intersects(V1, V2, V3, t)) {
            tvalues.collisions[tvalues.count++] = t;
        }
    }

    // Print the t_values


    // Sort the t_values
    thrust::sort(thrust::device, tvalues.collisions, tvalues.collisions + tvalues.count);

    // printf ("t_values count = %d\n", t_values.count);
    // for (int i = 0; i < t_values.count; i++) {
    //     printf ("t_values[%d] = %f\n", i, t_values.collisions[i]);
    // }

    // compute the thickness
    float val = sumTvalues(tvalues);
    // printf ("thickness = %f\n", val);
    return val;
}

__host__ __device__ void RayTracer::testSingleRay(Ray ray, CollisionList *collisions) {
    // ray.print();
    CandidateList candidates;
    candidates.count = 0;
    memset(candidates.collisions, 0, MAX_COLLISIONS * sizeof(int));
    this->tree->query(ray, candidates);

    if (candidates.count == 0) {
        printf ("No collision\n");
        return;
    }

    // Test the candidates for actual intersections
    for (int i = 0; i < candidates.count; i++) {
        int primIndex = candidates.collisions[i] * 3;
        float4 V1 = this->vertices[primIndex];
        float4 V2 = this->vertices[primIndex + 1];
        float4 V3 = this->vertices[primIndex + 2];

        float t;
        if (ray.intersects(V1, V2, V3, t)) {
            collisions->collisions[collisions->count++] = t;
        }
    }

    thrust::sort(thrust::device, collisions->collisions, collisions->collisions + collisions->count);
}

__global__ void projectPlaneRaysKernel (
    RayTracer *tracer, float *image, uint2 N, float2 D,
    BasisNamespace::Basis projectionPlaneBasis) {
    int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int gid_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (N.x == 0 || N.y == 0) {
        return;
    }

    if (gid_x >= N.x || gid_y >= N.y) {
        return;
    }

    for (int i = gid_x; i < N.x; i += blockDim.x * gridDim.x) {
        for (int j = gid_y; j < N.y; j += blockDim.y * gridDim.y) {

            float4 point_local_basis = tracer->phi(i, j, D, N);
            float4 point_new_basis = projectionPlaneBasis.getPointInBasis(point_local_basis);

            float4 direction = projectionPlaneBasis.getVector(2);
            Ray ray = Ray(point_new_basis, direction);

            float thickness = tracer->traceRayParallel(ray);

            image[j * N.x + i] = thickness;
        }
    }
}
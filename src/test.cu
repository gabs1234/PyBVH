#include "Commons.cuh"
#include "RayTracer.cuh"
#include "Ray.cuh"
#include "tree_prokopenko.cuh"
#include "Basis.cuh"

// __device__ void findCollisions(float4 *vertices, Ray &ray, CollisionList &candidates, CollisionList &t) {
//     for (int i = 0; i < candidates.count; i++) {
//         float4 v1 = vertices[candidates.collisions[i]];
//         float4 v2 = vertices[candidates.collisions[i] + 1];
//         float4 v3 = vertices[candidates.collisions[i] + 2];

//         float t_value;
//         if (ray.intersects(v1, v2, v3, t_value)) {
//             printf ("t: %f\n", t_value);
//             t.collisions[t.count++] = t_value;
//         }
//     }

// }

// __device__ __forceinline__ void calculateRayBoundingBox(float4 *ray_start, float4 *ray_end,
//                                                         float4 *bbMin, float4 *bbMax)
// {
//     float4 bboxMin, bboxMax;

//     bboxMin.x = min(ray_start->x, ray_end->x);
//     bboxMin.y = min(ray_start->y, ray_end->y);
//     bboxMin.z = min(ray_start->z, ray_end->z);

//     bboxMax.x = max(ray_start->x, ray_end->x);
//     bboxMax.y = max(ray_start->y, ray_end->y);
//     bboxMax.z = max(ray_start->z, ray_end->z);

//     *bbMin = bboxMin;
//     *bbMax = bboxMax;
// }


// __device__ float intersects(
//     float4 V1, float4 V2, float4 V3,
//     float4 O, float4 D)
// {
//     // calculate dimension where the ray direction is maximal
//     const float4 D_abs = abs4(D);
//     const int shift = maxDimIndex(D_abs);
//     // printf ("Shift: %d\n", shift);
//     float4 D_perm = permuteVectorAlongMaxDim(D, shift);
//     // printf ("D_perm: %f %f %f\n", D_perm.x, D_perm.y, D_perm.z);

//     // swap kx and ky dimensions to preserve winding direction of triangles
//     if (D_perm.z < 0.0f)
//     {
//         const float temp = D_perm.x;
//         D_perm.x = D_perm.y;
//         D_perm.y = temp;
//     }

//     /* calculate shear constants */
//     float Sx = D_perm.x / D_perm.z;
//     float Sy = D_perm.y / D_perm.z;
//     float Sz = 1.0f / D_perm.z;

//     // printf("D_perm: %f %f %f\n", D_perm.x, D_perm.y, D_perm.z);

//     /* calculate vertices relative to ray origin */
//     float4 A = sub(V1, O);
//     float4 B = sub(V2, O);
//     float4 C = sub(V3, O);

//     A = permuteVectorAlongMaxDim(A, shift);
//     B = permuteVectorAlongMaxDim(B, shift);
//     C = permuteVectorAlongMaxDim(C, shift);

//     // printf ("A: %f %f %f\n", A.x, A.y, A.z);
//     // printf ("B: %f %f %f\n", B.x, B.y, B.z);
//     // printf ("C: %f %f %f\n", C.x, C.y, C.z);

//     /* perform shear and scale of vertices */
//     const float Ax = A.x - Sx * A.z;
//     const float Ay = A.y - Sy * A.z;
//     const float Bx = B.x - Sx * B.z;
//     const float By = B.y - Sy * B.z;
//     const float Cx = C.x - Sx * C.z;
//     const float Cy = C.y - Sy * C.z;

//     // calculate scaled barycentric coordinates
//     float U = Cx * By - Cy * Bx;
//     float V = Ax * Cy - Ay * Cx;
//     float W = Bx * Ay - By * Ax;

//     /* fallback to test against edges using double precision  (if float is indeed float) */
//     if (U == (float)0.0 || V == (float)0.0 || W == (float)0.0)
//     {
//         double CxBy = (double)Cx * (double)By;
//         double CyBx = (double)Cy * (double)Bx;
//         U = (float)(CxBy - CyBx);
//         double AxCy = (double)Ax * (double)Cy;
//         double AyCx = (double)Ay * (double)Cx;
//         V = (float)(AxCy - AyCx);
//         double BxAy = (double)Bx * (double)Ay;
//         double ByAx = (double)By * (double)Ax;
//         W = (float)(BxAy - ByAx);
//     }

//     if ((U < (float)0.0 || V < (float)0.0 || W < (float)0.0) &&
//         (U > (float)0.0 || V > (float)0.0 || W > (float)0.0))
//         return -1;

//     /* calculate determinant */
//     float det = U + V + W;
//     if (det == (float)0.0)
//         return -1;

//     // printf("Det: %f\n", det);

//     /* Calculates scaled z-coordinate of vertices and uses them to calculate the hit distance. */
//     const float Az = Sz * A.z;
//     const float Bz = Sz * B.z;
//     const float Cz = Sz * C.z;
//     const float T = U * Az + V * Bz + W * Cz;

//     const int det_sign_mask = (int(det) & 0x80000000);
//     const float xort_t = xor_signmask(T, det_sign_mask);
//     if (xort_t < 0.0f)
//         return -1;

//     // normalize U, V, W, and T
//     const float rcpDet = 1.0f / det;
//     // *u = U*rcpDet;
//     // *v = V*rcpDet;
//     // *w = W*rcpDet;

//     return T * rcpDet;
// }

// __device__ void bvhFindCollisions(
//     BVHTree *tree, CollisionList &collisions, CollisionList &candidates,
//     float4 *ray_start, float4 *ray_end, float4 *vertices)
// {
//     // Loop over the candidates and find actual collisions
//     for (int j = 0; j < candidates.count; j++)
//     {
//         unsigned int triangle_index = candidates.collisions[j];

//         unsigned int id = triangle_index * 3;

//         float4 v1, v2, v3;
//         v1.x = vertices[id].x;
//         v1.y = vertices[id].y;
//         v1.z = vertices[id].z;
//         v1.w = 0.0f;
//         v2.x = vertices[id + 1].x;
//         v2.y = vertices[id + 1].y;
//         v2.z = vertices[id + 1].z;
//         v2.w = 0.0f;
//         v3.x = vertices[id + 2].x;
//         v3.y = vertices[id + 2].y;
//         v3.z = vertices[id + 2].z;
//         v3.w = 0.0f;

//         float4 origin, direction;
//         origin.x = ray_start->x;
//         origin.y = ray_start->y;
//         origin.z = ray_start->z;
//         origin.w = 0.0f;
//         direction.x = ray_end->x - ray_start->x;
//         direction.y = ray_end->y - ray_start->y;
//         direction.z = ray_end->z - ray_start->z;
//         direction.w = 0.0f;
        
//         // printf("Direction: %f %f %f\n", direction.x, direction.y, direction.z);

//         float t = intersects(v1, v2, v3, origin, direction);
//         if (t > -1) {
//             collisions.collisions[collisions.count++] = t;
//             if (collisions.count >= MAX_COLLISIONS)
//             {
//                 return;
//             }
//         }
//     }
// }

extern "C" {


__global__ void getRayTracerSize(unsigned int *size) {
    *size = sizeof(RayTracer);
}

__global__ void initializeRayTracer(RayTracer *rayTracer, BVHTree *tree, float4 *origin, float4 *vertices, unsigned int nbVertices, bool parallel) {
    *rayTracer = RayTracer(tree, origin[0], vertices, nbVertices, parallel);

    rayTracer->printRayTracer();
}

__global__ void testRayBoxIntersection() {
    float4 bbMin = make_float4(-5, -5, 0, 1);
    float4 bbMax = make_float4(1, 1, 1, 1);

    float4 origin = make_float4(-5, -5, 1, 1);
    float4 direction = make_float4(0, 0, -1, 0);

    Ray ray = Ray(origin, direction);
    ray.setDirection(direction);

    ray.print();

    float2 t;
    if (ray.intersects(bbMin, bbMax, t)) {
        printf ("t: %f %f\n", t.x, t.y);
    } else {
        printf ("No intersection\n");
    }
}

// __global__ void testRayTriangleIntersection() {
//     float4 v1 = make_float4(0, 0, 0, 1);
//     float4 v2 = make_float4(1, 0, 0, 1);
//     float4 v3 = make_float4(0, 1, 0, 1);

//     float4 origin = make_float4(0.3, .3, -1, -1);
//     float4 direction = make_float4(0, 0, -1, 0);

//     Ray ray = Ray(origin, direction);
//     ray.setDirection(direction);

//     ray.print();

//     float t;
//     if (ray.intersects(v1, v2, v3, t)) {
//         printf ("t: %f\n", t);
//     } else {
//         printf ("No intersection\n");
//     }
// }

__global__ void projectPlaneRays(
        RayTracer *rayTracer, float *image, uint2 *N, float2 *D, 
        float2 *spherical, float4 *viewerOrigin) {
    // 2D grid of 1D blocks
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    int globalThreadNum = tx + ty * N->x;

    if (!rayTracer->hasParallelGeometry()) {
        printf ("Geometry is not parallel\n");
        return;
    }

    if (N->x == 0 || N->y == 0) {
        printf ("Nx or Ny is 0\n");
        return;
    }

    if (tx >= N->x || ty >= N->y){
        return;
    }

    // if (globalThreadNum > 0){
    //     return;
    // }

    // printf ("Nx: %d Ny: %d\n", Nx, Ny);

    float4 rorigin = *viewerOrigin;

    float delta_x = D->x / N->x;
    float delta_y = D->y / N->y;

    Basis rayBasis;

    rayBasis.translate(rorigin);
    rayBasis.rotate(*spherical); // theta, phi
    rayBasis.scale(delta_x , delta_y , 1);

    int Nx2 = N->x / 2;
    int Ny2 = N->y / 2;

    int x_i = -Nx2 + tx;
    int y_i = -Ny2 + ty;


    while(x_i < Nx2) {
        while(y_i < Ny2) {
            int4 local_coords = make_int4(x_i, y_i, 0, 0);

            float4 global_coords = rayBasis.getPointInBasis(local_coords);

            Ray ray(global_coords, rayBasis.getVector(2)); // Ray along z-axis

            // ray.print();

            image[globalThreadNum] = rayTracer->traceRayParallel(ray);

            y_i += blockDim.y * gridDim.y;
        }
        x_i += blockDim.x * gridDim.x;
    }
}

// __global__ void projectPlaneRaysSimple(
//         BVHTree *tree, float *image, float4 *vertices,
//         unsigned int Nx, unsigned int Ny,
//         float Dx, float Dy) {
//     // 2D grid of 1D blocks
//     int x_i = threadIdx.x + blockIdx.x * blockDim.x;
//     int y_i = threadIdx.y + blockIdx.y * blockDim.y;

//     int globalThreadNum = x_i + y_i * Nx;

//     if (Nx == 0 || Ny == 0) {
//         printf ("Nx or Ny is 0\n");
//         return;
//     }

//     if (x_i >= Nx || y_i >= Ny){
//         return;
//     }

//     float delta_x = Dx / Nx;
//     float delta_y = Dy / Ny;

//     float4 bbMinScene = tree->getSceneBBMin();
//     float4 bbMaxScene = tree->getSceneBBMax();

//     printf ("bbMinScene: %f %f %f\n", bbMinScene.x, bbMinScene.y, bbMinScene.z);
//     printf ("bbMaxScene: %f %f %f\n", bbMaxScene.x, bbMaxScene.y, bbMaxScene.z);

//     float pixel_width = (bbMaxScene.x - bbMinScene.x) / Nx;
//     float pixel_height = (bbMaxScene.y - bbMinScene.y) / Ny;

//     printf ("Pixel width: %f\n", pixel_width);
//     printf ("Pixel height: %f\n", pixel_height);

//     while( globalThreadNum < Nx * Ny) {
//         float4 ray_start, ray_end;

//         // Ray in the middle of the pixel
//         ray_start.x = (x_i + .5) * pixel_width + bbMinScene.x;
//         ray_start.y = (y_i + .5) * pixel_height + bbMinScene.y;
//         ray_start.z = bbMinScene.z;
//         ray_start.w = 0;
//         ray_end.x = ray_start.x ;
//         ray_end.y = ray_start.y ;
//         ray_end.z = bbMaxScene.z;
//         ray_end.w = 0;

//         float4 bbMinQuery, bbMaxQuery;
//         calculateRayBoundingBox(&ray_start, &ray_end, &bbMinQuery, &bbMaxQuery);

//         CollisionList candidates;
//         candidates.count = 0;
//         memset(candidates.collisions, 0, sizeof(float) * MAX_COLLISIONS);

//         CollisionList t_values;
//         t_values.count = 0;
//         memset(t_values.collisions, 0, sizeof(float) * MAX_COLLISIONS);

//         tree->query(bbMinQuery, bbMaxQuery, candidates);

//         bvhFindCollisions(tree, t_values, candidates, &ray_start, &ray_end, vertices);

//         image[globalThreadNum] = computeThickness(t_values);

//         idx += blockDim.x * gridDim.x;
//         idy += blockDim.y * gridDim.y;
//         globalThreadNum = idx + idy * nx;
//     }
// }

__global__ void printTreeProkopenko(BVHTree *tree) {
    tree->printTree();
}

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

        // printf ("INDEX: %d\n", index);

        index += blockDim.x * gridDim.x;
    }
}
    
}
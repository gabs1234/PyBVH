#include "Ray.cuh"

__host__ __device__ Ray::Ray(const Ray &ray) {
    this->tail = ray.tail;
    this->head = ray.head;
    this->direction = ray.direction;
    this->invDirection = ray.invDirection;
    this->sign[0] = ray.sign[0];
    this->sign[1] = ray.sign[1];
    this->sign[2] = ray.sign[2];
    // printf ("Signs: %d, %d, %d\n", this->sign[0], this->sign[1], this->sign[2]);
};
__host__ __device__ Ray::Ray(float4 tail, float4 direction){
    this->tail = tail;
    this->direction = direction;
    this->updateInvDirection();
    this->updateSign();
    // this->print();
};

__host__ __device__ void Ray::print() {
    printf ("Ray: %f %f %f -> %f %f %f\n", this->tail.x, this->tail.y, this->tail.z, this->direction.x, this->direction.y, this->direction.z);
    printf ("InvDirection: %f %f %f\n", this->invDirection.x, this->invDirection.y, this->invDirection.z);
    printf ("Signs: %d, %d, %d\n", this->sign[0], this->sign[1], this->sign[2]);
}

__host__ __device__ float Ray::normalize(float4 &v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ void Ray::updateDirection() {
    float norm = this->normalize(this->direction);
    this->direction.x = this->head.x - this->tail.x;
    this->direction.y = this->head.y - this->tail.y;
    this->direction.z = this->head.z - this->tail.z;
    
    this->direction.x /= norm;
    this->direction.y /= norm;
    this->direction.z /= norm;
}

__host__ __device__ void Ray::updateInvDirection() {
    this->invDirection = make_float4(1.0f / this->direction.x, 1.0f / this->direction.y, 1.0f / this->direction.z, 0.0f);
}

__host__ __device__ void Ray::updateSign() {
    this->sign[0] = (this->invDirection.x < 0);
    this->sign[1] = (this->invDirection.y < 0);
    this->sign[2] = (this->invDirection.z < 0);
}

__host__ __device__ float4 Ray::computeParametric(float t) {
    float4 P;
    P.x = this->tail.x + t * this->direction.x;
    P.y = this->tail.y + t * this->direction.y;
    P.z = this->tail.z + t * this->direction.z;
    P.w = 1.0f;
    return P;
}

__host__ __device__ void Ray::setTail(float4 &tail) {
    this->tail = tail;
}
__host__ __device__ void Ray::setHead(float4 &head) {
    this->head = head;
}

__host__ __device__ void Ray::setDirection(float4 &direction) {
    this->direction = direction;
    this->updateInvDirection();
    this->updateSign();
}

__host__ __device__ void Ray::setInvDirection(float4 &invDirection) {
    this->invDirection = invDirection;
}

__host__ __device__ bool Ray::intersects (float4 min, float4 max, float2 &t) {
    float4 bounds[2];
    bounds[0] = min;
    bounds[1] = max;

    float tymin, tymax, tzmin, tzmax;

    // printf ("sign: %d %d %d\n", this->sign[0], this->sign[1], this->sign[2]);3
    t.x = (bounds[this->sign[0]].x - this->tail.x) * this->invDirection.x;
    t.y = (bounds[1 - this->sign[0]].x - this->tail.x) * this->invDirection.x;
    tymin = (bounds[this->sign[1]].y - this->tail.y) * this->invDirection.y;
    tymax = (bounds[1 - this->sign[1]].y - this->tail.y) * this->invDirection.y;

    if ((t.x > tymax) || (tymin > t.y))
        return false;

    if (tymin > t.x)
        t.x = tymin;

    if (tymax < t.y)
        t.y = tymax;

    tzmin = (bounds[this->sign[2]].z - this->tail.z) * this->invDirection.z;
    tzmax = (bounds[1 - this->sign[2]].z - this->tail.z) * this->invDirection.z;

    if ((t.x > tzmax) || (tzmin > t.y))
        return false;

    if (tzmin > t.x)
        t.x = tzmin;

    if (tzmax < t.y)
        t.y = tzmax;

    return true;
}

// /**
//  * S. Woop et al. 2013, "Watertight Ray/Triangle Intersection"
//  */
// __device__ bool Ray::intersects(
//     float4 &V1, float4 &V2, float4 &V3, float &t)
// {
//     // calculate dimension where the ray direction is maximal
//     float4 dir = this->direction;
//     const float4 D_abs = abs4(dir);
//     const int shift = maxDimIndex(D_abs);
//     // printf ("Shift: %d\n", shift);
//     float4 D_perm = permuteVectorAlongMaxDim(dir, shift);
//     // printf ("D_perm: %f %f %f\n", D_perm.x, D_perm.y, D_perm.z);

//     // swap kx and ky dimensions to preserve winding direction of triangles
//     if (D_perm.z < 0.0f)
//     {
//         const float temp = D_perm.x;
//         D_perm.x = D_perm.y;
//         D_perm.y = temp;
//     }

//     /* calculate shear constants */
//     float Sz = 1.0f / D_perm.z;
//     float Sx = D_perm.x * Sz;
//     float Sy = D_perm.y * Sz;

//     // printf("D_perm: %f %f %f\n", D_perm.x, D_perm.y, D_perm.z);

//     /* calculate vertices relative to ray origin */
//     float4 tail = this->tail;
//     float4 A = sub4(V1, tail);
//     float4 B = sub4(V2, tail);
//     float4 C = sub4(V3, tail);

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
//         (U > (float)0.0 || V > (float)0.0 || W > (float)0.0)){
//         return false;
//     }

//     /* calculate determinant */
//     float det = U + V + W;
//     if (det == (float)0.0)
//         return false;

//     // printf("Det: %f\n", det);

//     /* Calculates scaled z-coordinate of vertices and uses them to calculate the hit distance. */
//     const float Az = Sz * A.z;
//     const float Bz = Sz * B.z;
//     const float Cz = Sz * C.z;
//     const float T = U * Az + V * Bz + W * Cz;

//     const int det_sign_mask = (int(det) & 0x80000000);
//     const float xort_t = xor_signmask(T, det_sign_mask);
//     if (xort_t < 0.0f)
//         return false;

//     // normalize U, V, W, and T
//     const float rcpDet = 1.0f / det;
//     // *u = U*rcpDet;
//     // *v = V*rcpDet;
//     // *w = W*rcpDet;

//     t = T * rcpDet;

//     return t > -1;
// }

// __device__ bool Ray::intersects(
//     float4 &V1, float4 &V2, float4 &V3, float &t)
// {
//     float4 e_1, e_2, P, Q, T;
//     float det, inv_det, u, v;

//     e_1 = sub4 (V2, V1);
//     e_2 = sub4 (V3, V1);
//     P = cross4 (this->direction, e_2);

//     det = dot4 (e_1, P);
    
//     if (det > -EPSILON && det < EPSILON) {
//         return false;
//     }

//     inv_det = 1 / det;
//     T = sub4(this->tail, V1);
//     u = dot4(T, P) * inv_det;
//     if (u < 0 || u > 1) {
//         return false;
//     }

//     Q = cross4 (T, e_1);
//     v = dot4 (this->direction, Q) * inv_det;
//     if (v < 0 || u + v > 1) {
//         return false;
//     }

//     t = dot4 (e_2, Q) * inv_det;

//     return true;
// }

// computes the difference of products a*b - c*d using 
// FMA instructions for improved numerical precision
__host__ __device__ inline float diff_product(float a, float b, float c, float d) 
{
    float cd = c * d;
    float diff = fmaf(a, b, -cd);
    float error = fmaf(-c, d, cd);

    return diff + error;
}

// http://jcgt.org/published/0002/01/05/
__host__ __device__ bool Ray::intersects(
    float4 &V1_e, float4 &V2_e, float4 &V3_e, float &t)
{
	// todo: precompute for ray
    float V1[3] = {V1_e.x, V1_e.y, V1_e.z};
    float V2[3] = {V2_e.x, V2_e.y, V2_e.z};
    float V3[3] = {V3_e.x, V3_e.y, V3_e.z};

	int kz = maxDimIndex(abs4(this->direction));
	int kx = kz+1; if (kx == 3) kx = 0;
	int ky = kx+1; if (ky == 3) ky = 0;

    // TODO . make this in ray
    float dir[3] = {this->direction.x, this->direction.y, this->direction.z};

	if (dir[kz] < 0.0f)
	{
		float tmp = kx;
		kx = ky;
		ky = tmp;
	}

	float Sx = dir[kx]/dir[kz];
	float Sy = dir[ky]/dir[kz];
	float Sz = 1.0f/dir[kz];

	// todo: end precompute
    const float O[3] = {this->tail.x, this->tail.y, this->tail.z};
	float A[3], B[3], C[3];
    for (int i = 0; i < 3; i++) {
        A[i] = V1[i] - O[i];
        B[i] = V2[i] - O[i];
        C[i] = V3[i] - O[i];
    }
	
	const float Ax = A[kx] - Sx*A[kz];
	const float Ay = A[ky] - Sy*A[kz];
	const float Bx = B[kx] - Sx*B[kz];
	const float By = B[ky] - Sy*B[kz];
	const float Cx = C[kx] - Sx*C[kz];
	const float Cy = C[ky] - Sy*C[kz];
		
    float U = diff_product(Cx, By, Cy, Bx);
    float V = diff_product(Ax, Cy, Ay, Cx);
    float W = diff_product(Bx, Ay, By, Ax);

	if (U == 0.0f || V == 0.0f || W == 0.0f) 
	{
		double CxBy = (double)Cx*(double)By;
		double CyBx = (double)Cy*(double)Bx;
		U = (float)(CxBy - CyBx);
		double AxCy = (double)Ax*(double)Cy;
		double AyCx = (double)Ay*(double)Cx;
		V = (float)(AxCy - AyCx);
		double BxAy = (double)Bx*(double)Ay;
		double ByAx = (double)By*(double)Ax;
		W = (float)(BxAy - ByAx);
	}

	if ((U<0.0f || V<0.0f || W<0.0f) &&	(U>0.0f || V>0.0f || W>0.0f)) 
    {
        return false;
    }

	float det = U+V+W;

	if (det == 0.0f) 
    {
		return false;
    }

	const float Az = Sz*A[kz];
	const float Bz = Sz*B[kz];
	const float Cz = Sz*C[kz];
	const float T = U*Az + V*Bz + W*Cz;

	int det_sign = sign_mask(det);
	if (xorf(T,det_sign) < 0.0f)// || xorf(T,det_sign) > hit.t * xorf(det, det_sign)) // early out if hit.t is specified
    {
		return false;
    }

	const float rcpDet = 1.0f/det;
	// u = U*rcpDet;
	// v = V*rcpDet;
	t = T*rcpDet;
	// sign = det;
	
	// // optionally write out normal (todo: this branch is a performance concern, should probably remove)
	// if (normal)
	// {
	// 	const vec3 ab = b-a;
	// 	const vec3 ac = c-a;

	// 	// calculate normal
	// 	*normal = cross(ab, ac); 
	// }

	return true;
}
#ifndef BASIS_CUH
#define BASIS_CUH

#include "RotationQuaternion.cuh"

class Basis {
public:
    __device__ Basis() {
        this->u1 = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
        this->u2 = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
        this->u3 = make_float4(0.0f, 0.0f, 1.0f, 0.0f);

        this->origin = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    };
    __device__ Basis(float4 u1, float4 u2, float4 u3) : u1(u1), u2(u2), u3(u3) {\
        this->origin = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    };
    __device__ Basis(float4 origin, float4 u1, float4 u2, float4 u3) : origin(origin), u1(u1), u2(u2), u3(u3) { };
    

    __device__ void translate(float4 translation);
    __device__ void translate(float x, float y, float z);
    __device__ void rotate(float theta, float phi);
    __device__ void rotate(float2 spherical);
    __device__ void rotate (float4 euler);
    __device__ void scale(float s1, float s2, float s3);
    __device__ void scale(float4 scale);

    __device__ float4 getPointInBasis(int4 c) const;
    __device__ float4 getPointInBasis(float4 c) const;

    __device__ float4 getVector(int i) const;

    __device__ float4 getOrigin() const { return this->origin; };
    __device__ float4 getU1() const { return this->u1; };
    __device__ float4 getU2() const { return this->u2; };
    __device__ float4 getU3() const { return this->u3; };

    __device__ void print () const {
        printf("origin: %f %f %f\n", this->origin.x, this->origin.y, this->origin.z);
        printf("u1: %f %f %f\n", this->u1.x, this->u1.y, this->u1.z);
        printf("u2: %f %f %f\n", this->u2.x, this->u2.y, this->u2.z);
        printf("u3: %f %f %f\n", this->u3.x, this->u3.y, this->u3.z);
    }

private:
    float4 origin;
    float4 u1, u2, u3;
};

#endif // BASIS_CUH
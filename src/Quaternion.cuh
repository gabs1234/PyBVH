#pragma once

#include <cuda_runtime.h>
#include <cstdio>

class Quaternion {
public:
    __host__ __device__ Quaternion();
    __host__ __device__ Quaternion(float4 q);
    __host__ __device__ Quaternion(float4 q, float norm);
    __host__ __device__ Quaternion(const Quaternion& q);

    __host__ __device__ float getReal() const;
    __host__ __device__ float4 getImaginary() const;
    __host__ __device__ float getNorm();
    __host__ __device__ float4 getQuaternion() const;
    __host__ __device__ Quaternion getConjugate() const;
    __host__ __device__ Quaternion getInverse();

    __host__ __device__ void conjugate();
    __host__ __device__ void normalize();
    __host__ __device__ void inverse();

    __host__ __device__ void print() const {
        printf("Quaternion: %f + %fi + %fj + %fk\n", q.x, q.y, q.z, q.w);}

    __host__ __device__ Quaternion operator*(const float& s);
    __host__ __device__ Quaternion operator/(const float& s);
    __host__ __device__ Quaternion operator+(const Quaternion& q);
    __host__ __device__ Quaternion operator-(const Quaternion& q);
    __host__ __device__ Quaternion operator*(const Quaternion& q);
    __host__ __device__ Quaternion operator-();

protected:
    float4 q;
    float norm;
    bool isNormalized;
};
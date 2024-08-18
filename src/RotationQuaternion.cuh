#pragma once
#include <cuda_runtime.h>
#include "Commons.cuh"
#include "Quaternion.cuh"

class RotationQuaternion : public Quaternion {
public:
    __host__ __device__ RotationQuaternion();
    // Convert a quaternion to a rotation quaternion
    __host__ __device__ RotationQuaternion(Quaternion &q) : Quaternion(q) { };
    __host__ __device__ RotationQuaternion(const RotationQuaternion &q);
    __host__ __device__ RotationQuaternion(float angle, float4 axis);

    __host__ __device__ void setAngle(float angle);
    __host__ __device__ void setAxis(float4 axis);
    __host__ __device__ float getAngle() const;
    __host__ __device__ float4 getAxis() const;

    __host__ __device__ float4 rotate(float4 vector);

private:
    float4 axis;
    float angle;
    float halfAngle;
    float cosHalfAngle;
    float sinHalfAngle;
};
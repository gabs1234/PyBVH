#ifndef ROTATIONQUATERNION_CUH
#define ROTATIONQUATERNION_CUH

#include "Quaternion.cuh"

class RotationQuaternion : public Quaternion {
public:
    __device__ RotationQuaternion();
    // Convert a quaternion to a rotation quaternion
    __device__ RotationQuaternion(Quaternion &q) : Quaternion(q) { };
    __device__ RotationQuaternion(const RotationQuaternion &q);
    __device__ RotationQuaternion(float angle, float4 axis);

    __device__ void setAngle(float angle);
    __device__ void setAxis(float4 axis);
    __device__ float getAngle() const;
    __device__ float4 getAxis() const;

    __device__ float4 rotate(float4 vector);

private:
    float4 axis;
    float angle;
    float halfAngle;
    float cosHalfAngle;
    float sinHalfAngle;
};

#endif // ROTATIONQUATERNION_CUH
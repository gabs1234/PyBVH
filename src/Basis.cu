#include "Basis.cuh"

__device__ void Basis::translate(float4 translation) {
    this->origin.x += translation.x;
    this->origin.y += translation.y;
    this->origin.z += translation.z;
}
__device__ void Basis::translate(float x, float y, float z) {
    this->origin.x += x;
    this->origin.y += y;
    this->origin.z += z;
}

/**
 * Use spherical coordinates as standard to transform the basis
*/
__device__ void Basis::rotate (float theta, float phi) {
    RotationQuaternion rot1(-theta, this->u1);
    RotationQuaternion rot2(phi, this->u3);

    this->u1 = rot1.rotate(this->u1);
    this->u2 = rot1.rotate(this->u2);
    this->u3 = rot1.rotate(this->u3);

    this->u1 = rot2.rotate(this->u1);
    this->u2 = rot2.rotate(this->u2);
    this->u3 = rot2.rotate(this->u3);
}

__device__ void Basis::rotate (float2 spherical) {
    RotationQuaternion rot1(-spherical.x, this->u1);
    RotationQuaternion rot2(spherical.y, this->u3);

    this->u1 = rot1.rotate(this->u1);
    this->u2 = rot1.rotate(this->u2);
    this->u3 = rot1.rotate(this->u3);

    this->u1 = rot2.rotate(this->u1);
    this->u2 = rot2.rotate(this->u2);
    this->u3 = rot2.rotate(this->u3);
}

__device__ void Basis::scale (float s1, float s2, float s3) {
    this->u1.x = this->u1.x * s1;
    this->u1.y = this->u1.y * s1;
    this->u1.z = this->u1.z * s1;
    
    this->u2.x = this->u2.x * s2;
    this->u2.y = this->u2.y * s2;
    this->u2.z = this->u2.z * s2;

    this->u3.x = this->u3.x * s3;
    this->u3.y = this->u3.y * s3;
    this->u3.z = this->u3.z * s3;
}

__device__ void Basis::scale (float4 scale) {
    this->u1.x = this->u1.x * scale.x;
    this->u1.y = this->u1.y * scale.x;
    this->u1.z = this->u1.z * scale.x;
    
    this->u2.x = this->u2.x * scale.y;
    this->u2.y = this->u2.y * scale.y;
    this->u2.z = this->u2.z * scale.y;

    this->u3.x = this->u3.x * scale.z;
    this->u3.y = this->u3.y * scale.z;
    this->u3.z = this->u3.z * scale.z;
}

__device__ float4 Basis::getPointInBasis(int4 c) const {
    float4 result;
    result.x = c.x * this->u1.x + c.y * this->u2.x + c.z * this->u3.x + this->origin.x;
    result.y = c.x * this->u1.y + c.y * this->u2.y + c.z * this->u3.y + this->origin.y;
    result.z = c.x * this->u1.z + c.y * this->u2.z + c.z * this->u3.z + this->origin.z;
    result.w = c.w;
    return result;
}

__device__ float4 Basis::getPointInBasis(float4 c) const {
    float4 result;
    result.x = c.x * this->u1.x + c.y * this->u2.x + c.z * this->u3.x + this->origin.x;
    result.y = c.x * this->u1.y + c.y * this->u2.y + c.z * this->u3.y + this->origin.y;
    result.z = c.x * this->u1.z + c.y * this->u2.z + c.z * this->u3.z + this->origin.z;
    result.w = c.w;
    return result;
}

__device__ float4 Basis::getVector(int i) const {
    switch (i) {
        case 0:
            return this->u1;
        case 1:
            return this->u2;
        case 2:
            return this->u3;
        default:
            return make_float4(0, 0, 0, 0);
    }
}
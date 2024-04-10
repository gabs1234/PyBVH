#include "Quaternion.cuh"
#include "RotationQuaternion.cuh"
#include "Commons.cuh"

__device__ RotationQuaternion::RotationQuaternion() : Quaternion() {
    this->angle = 0;
    this->axis = make_float4(0, 0, 1, 0);
}

//Copy constructor
__device__ RotationQuaternion::RotationQuaternion(const RotationQuaternion &rq) {
    this->angle = rq.angle;
    this->axis = rq.axis;
    this->halfAngle = rq.halfAngle;
    this->cosHalfAngle = rq.cosHalfAngle;
    this->sinHalfAngle = rq.sinHalfAngle;

    this->isNormalized = rq.isNormalized;
    this->q = rq.q;
    this->norm = rq.norm;
}

__device__ RotationQuaternion::RotationQuaternion(float angle, float4 axis) {
    this->angle = angle;
    this->halfAngle = this->angle / 2;
    this->cosHalfAngle = cosf(this->halfAngle);
    this->sinHalfAngle = sinf(this->halfAngle);

    this->axis = normalize4(axis);

    this->q.x = this->axis.x * this->sinHalfAngle;
    this->q.y = this->axis.y * this->sinHalfAngle;
    this->q.z = this->axis.z * this->sinHalfAngle;
    this->q.w = this->cosHalfAngle;

    this->isNormalized = true;
}

__device__ void RotationQuaternion::setAngle(float angle) {
    this->angle = angle;
}

__device__ void RotationQuaternion::setAxis(float4 axis) {
    this->axis = axis;
}

__device__ float RotationQuaternion::getAngle() const {
    return this->angle;
}

__device__ float4 RotationQuaternion::getAxis() const {
    return this->axis;
}

__device__ float4 RotationQuaternion::rotate(float4 v) {
    Quaternion p = Quaternion(v);
    // p.setReal(0);
    Quaternion left_hand = *this;
    Quaternion right_hand = this->isNormalized ? this->getConjugate() : this->getInverse();
    
    Quaternion result = left_hand * p * right_hand;
    return result.getImaginary();
}
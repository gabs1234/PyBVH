#include "Quaternion.cuh"
#include "Commons.cuh"

__device__ Quaternion::Quaternion() {
    this->q.w = 1;
    this->q.x = 0;
    this->q.y = 0;
    this->q.z = 0;

    this->norm = 1;
}

__device__ Quaternion::Quaternion(float4 q) {
    this->q.w = q.w;
    this->q.x = q.x;
    this->q.y = q.y;
    this->q.z = q.z;

    this->norm = this->getNorm();

    if (this->norm == 1)
        this->isNormalized = true;
    else
        this->isNormalized = false;
}

__device__ Quaternion::Quaternion(float4 q, float norm) {
    this->q.w = q.w;
    this->q.x = q.x;
    this->q.y = q.y;
    this->q.z = q.z;

    this->norm = norm;

    if (this->norm == 1)
        this->isNormalized = true;
    else
        this->isNormalized = false;
}

__device__ Quaternion::Quaternion(const Quaternion &p) {
    this->q.w = p.q.w;
    this->q.x = p.q.x;
    this->q.y = p.q.y;
    this->q.z = p.q.z;

    if (p.isNormalized) {
        this->isNormalized = true;
        this->norm = 1;
    }
    else {
        this->isNormalized = false;
        this->norm = p.norm;
    }
}

__device__ float Quaternion::getReal() const {
    return this->q.w;
}

__device__ float4 Quaternion::getImaginary() const {
    return make_float4(this->q.x, this->q.y, this->q.z, 0);
}

__device__ float Quaternion::getNorm() {
    if (this->isNormalized)
        return 1;
    else {
        this->norm = sqrt(this->q.w * this->q.w + this->q.x * this->q.x + this->q.y * this->q.y + this->q.z * this->q.z);
        if (this->norm == 1) {
            this->isNormalized = true;
        }
    }
    return this->norm;
}

__device__ void Quaternion::normalize() {
    if (!this->isNormalized) {
        this->getNorm();

        this->q.x = this->q.x / this->norm;
        this->q.y = this->q.y / this->norm;
        this->q.z = this->q.z / this->norm;
        this->q.w = this->q.w / this->norm;

        this->isNormalized = true;
    }
}

__device__ float4 Quaternion::getQuaternion () const {
    return this->q;
}

__device__ Quaternion Quaternion::getConjugate() const {
    float4 result;
    result.w = this->q.w;
    result.x = -this->q.x;
    result.y = -this->q.y;
    result.z = -this->q.z;

    return Quaternion(result, this->norm);
}

__device__ Quaternion Quaternion::getInverse() {
    float normSquared = this->getNorm();
    normSquared *= normSquared;
    return this->getConjugate() / normSquared;
}

__device__ Quaternion Quaternion::operator* (const float &s) {
    float4 result;
    result.w = this->q.w * s;
    result.x = this->q.x * s;
    result.y = this->q.y * s;
    result.z = this->q.z * s;
    return Quaternion(result);
}

__device__ Quaternion Quaternion::operator/ (const float &s) {
    float4 result;
    result.w = this->q.w / s;
    result.x = this->q.x / s;
    result.y = this->q.y / s;
    result.z = this->q.z / s;
    return Quaternion(result);
}

__device__ Quaternion Quaternion::operator+ (const Quaternion &p) {
    float4 result;
    result.w = this->q.w + p.q.w;
    result.x = this->q.x + p.q.x;
    result.y = this->q.y + p.q.y;
    result.z = this->q.z + p.q.z;
    return Quaternion(result);
}

__device__ Quaternion Quaternion::operator- (const Quaternion &p) {
    float4 result;
    result.w = this->q.w - p.q.w;
    result.x = this->q.x - p.q.x;
    result.y = this->q.y - p.q.y;
    result.z = this->q.z - p.q.z;
    return Quaternion(result);
}

__device__ Quaternion Quaternion::operator- () {
    float4 result;
    result.w = -this->q.w;
    result.x = -this->q.x;
    result.y = -this->q.y;
    result.z = -this->q.z;
    return Quaternion(result);
}

__device__ Quaternion Quaternion::operator* (const Quaternion &b) {
    // Stupid notation because following the book
    float4 q = b.q;
    float4 pf = this->q;
    float4 result; // qp

    result.w = pf.w * q.w - pf.x * q.x - pf.y * q.y - pf.z * q.z;
    result.x = pf.w * q.x + q.w * pf.x + pf.y * q.z - pf.z * q.y;
    result.y = pf.w * q.y + q.w * pf.y + pf.z * q.x - pf.x * q.z;
    result.z = pf.w * q.z + q.w * pf.z + pf.x * q.y - pf.y * q.x;
    return Quaternion(result);
}


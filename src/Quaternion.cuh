#ifndef QUATERION_CUH
#define QUATERION_CUH

class Quaternion {
public:
    __device__ Quaternion();
    __device__ Quaternion(float4 q);
    __device__ Quaternion(float4 q, float norm);
    __device__ Quaternion(const Quaternion& q);

    __device__ float getReal() const;
    __device__ float4 getImaginary() const;
    __device__ float getNorm();
    __device__ float4 getQuaternion() const;
    __device__ Quaternion getConjugate() const;
    __device__ Quaternion getInverse();

    __device__ void conjugate();
    __device__ void normalize();
    __device__ void inverse();

    __device__ void print() const {
        printf("Quaternion: %f + %fi + %fj + %fk\n", q.x, q.y, q.z, q.w);}

    __device__ Quaternion operator*(const float& s);
    __device__ Quaternion operator/(const float& s);
    __device__ Quaternion operator+(const Quaternion& q);
    __device__ Quaternion operator-(const Quaternion& q);
    __device__ Quaternion operator*(const Quaternion& q);
    __device__ Quaternion operator-();

protected:
    float4 q;
    float norm;
    bool isNormalized;
};

#endif // QUATERION_CUH
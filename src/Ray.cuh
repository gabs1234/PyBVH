#pragma once
#include "Commons.cuh"

class Ray {
public:
    __host__ __device__ Ray() {};

    __host__ __device__ Ray(const Ray &ray);
    __host__ __device__ Ray(float4 tail, float4 direction);

    __host__ __device__ float normalize(float4 &v);
    __host__ __device__ void updateDirection();
    __host__ __device__ void updateInvDirection();
    __host__ __device__ void updateSign();
    __host__ __device__ float4 computeParametric(float t);
    __host__ __device__ bool intersects (float4 min, float4 max, float2 &t); // AABB
    __host__ __device__ bool intersects (float4 &V1, float4 &V2, float4 &V3, float &t); // Triangle

    // setters
    __host__ __device__ void setTail(float4 &T);
    __host__ __device__ void setHead(float4 &H);
    __host__ __device__ void setDirection(float4 &D);
    __host__ __device__ void setInvDirection(float4 &ID);

    // getters
    __host__ __device__ float4 getTail() { return this->tail; };
    __host__ __device__ float4 getHead() { return this->head; };
    __host__ __device__ float4 getDirection() { return this->direction; };
    __host__ __device__ float4 getInvDirection() { return this->invDirection; };
    __host__ __device__ float4 getOppositeDirection() { return make_float4(-this->direction.x, -this->direction.y, -this->direction.z, 0.0f); };
    __host__ __device__ int getSign(int i) { return this->sign[i]; };


    __host__ __device__ void print();

private:
    float4 tail, head;

    float4 direction, invDirection;
    int sign[3];
};
#include "Quaternion.cuh"
#include "RotationQuaternion.cuh"

#define PI_F 3.141592654f

extern "C" {

__global__ void rotateVector(float4 *input, float4 *output, float4 *axis, float angle) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 1) {
        return;
    }
    float4 laxis = make_float4(0, 0.0f, 1.0f, 0.0f);
    float langle = PI_F;
    float4 vector = make_float4(1, 0, 0.0f, 0.0f);

    RotationQuaternion q(langle, laxis);
    float4 result = q.rotate(vector);

    printf("Result: %f %f %f %f\n", result.x, result.y, result.z, result.w);
}

}
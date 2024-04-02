#include <cub/cub.cuh>
#include <cuda_runtime_api.h>

//---------------------------------------------------------------------
// Kernels
//---------------------------------------------------------------------

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
/**
 * Simple kernel for performing a block-wide sorting over integers
 */



// In and out buffers may be swaped
// Original data is not kept
extern "C" {

#define KEY_TYPE unsigned int
#define VALUE_TYPE unsigned int

__global__ void deviceSort(
    unsigned int numberOfElements, 
    KEY_TYPE** keysIn, KEY_TYPE** keysOut,
    VALUE_TYPE** valuesIn, VALUE_TYPE** valuesOut)
{
    cub::DoubleBuffer<KEY_TYPE> keysBuffer(*keysIn, *keysOut);
    cub::DoubleBuffer<VALUE_TYPE> valuesBuffer(*valuesIn, *valuesOut);

    // Check how much temporary memory will be required
    void* tempStorage = nullptr;
    size_t storageSize = 0;
    // cub::DeviceRadixSort::SortPairs(tempStorage, storageSize, keysBuffer, valuesBuffer,
    // numberOfElements);
    cub::DeviceRadixSort::SortKeys(tempStorage, storageSize, keysBuffer, numberOfElements);

    // Allocate temporary memory
    cudaMalloc(&tempStorage, storageSize);

    // Sort
    cub::DeviceRadixSort::SortPairs(tempStorage, storageSize, keysBuffer, valuesBuffer,
                                    numberOfElements);

    // Free temporary memory
    cudaFree(tempStorage);

    // Update out buffers
    KEY_TYPE* current = keysBuffer.Current();
    keysOut = &current;
    unsigned int* current2 = valuesBuffer.Current();
    valuesOut = &current2;

    // Update in buffers
    current = keysBuffer.d_buffers[1 - keysBuffer.selector];
    keysIn = &current;
    current2 = valuesBuffer.d_buffers[1 - valuesBuffer.selector];
    valuesIn = &current2;
}

}
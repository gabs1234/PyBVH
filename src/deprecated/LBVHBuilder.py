from cuda import cuda, nvrtc
import numpy as np
import Scene

#include "BVHTreeInstanceManager.h"
#include "CubWrapper.h"
#include "CudaErrorCheck.h"
#include "CudaTimer.h"
#include "Defines.h"
#include "LBVH.h"

def catchException(err):
    if err != nvrtc.NVRTC_SUCCESS:
        print ("Error: ", nvrtc.nvrtcGetErrorString(err))
        return None

def compile_kernel(source, function_name, options=[]):
    # Create program
    err, prog = nvrtc.nvrtcCreateProgram(source, function_name, [], [])
    catchException (err)

    # Compile program
    opts = ["--gpu-architecture=compute_61", "--gpu-code=compute_80"]
    err, = nvrtc.nvrtcCompileProgram(prog, 2, opts)
    catchException (err)

    # Get PTX from compilation
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    catchException (err)

    ptx = b" " * ptxSize
    err, = nvrtc.nvrtcGetPTX(prog, ptx)
    catchException (err)

class LBVHBuilder():
    def __init__(self, use64BitMortonCode=False) -> None:
        self.use64BitMortonCode = use64BitMortonCode
        if use64BitMortonCode:
            self.mortonCodesType = np.uint64
            self.GenerateMortonCodes = self._GenerateMortonCodes64
        else:
            self.mortonCodesType = np.uint32
            self.GenerateMortonCodes = self._GenerateMortonCodes32

        # Load the kernel source code from LVBH.cu
        with open('LVBH.cu', 'rt') as kernelFile:
            kernelsSource = kernelFile.read()


            
        # Compile the kernel source code
        program = nvrtc.nvrtcCreateProgram(kernelsSource.encode(), 'LBVH.cu', [], [])



    def _GenerateMortonCodes32(self, scene) -> float:
        # Load kernel source code
        return 0.0

    def _BuildTreeNoTriangleSplitting(self, scene) -> float:
        """
        Build a BVH tree using the LBVH algorithm without triangle splitting.
        Return the time spent in milliseconds.
        """
        # Initialize CUDA events
        startEvent = cuda.CUevent()

        nbTriangles = scene.numberOfTriangles
        mortonCodesSize = nbTriangles * self.mortonCodesType.itemsize

        # Create Morton codes buffer
        err, deviceMortonCodes = cuda.cuMemAlloc(mortonCodesSize)

        # Create sorted keys buffer
        err, deviceSortedIndices = cuda.cuMemAlloc(mortonCodesSize)

        # Generate Morton codes
        blockSize = 256
        gridSize = (nbTriangles + blockSize - 1) // blockSize


        stopEvent = cuda.CUevent()


        CUResult, time = cuda.cuEventElapsedTime(self.startEvent, self.stopEvent)
        # conver CUresult to string
        res = cuda.cuGetErrorString(CUResult)
        print(res)
    
    def BuildTree(self, V1, V2, V3):
        return self._BuildTreeNoTriangleSplitting(V1, V2, V3)


test = LBVHBuilder()
test.BuildTree(1, 2, 3)

# template <typename T> BVHTree* LBVHBuilder::BuildTreeNoTriangleSplitting(
#         const SceneWrapper* sceneWrapper) const
# {
#     CudaTimer timer;
#     timer.Start();

#     float timeMortonCodes, timeSort, timeBuildTree, timeBoundingBoxes;

#     const Scene* deviceScene = sceneWrapper->DeviceScene();
#     unsigned int numberOfTriangles = sceneWrapper->HostScene()->numberOfTriangles;

#     size_t globalMemoryUsed = 0;

#     // Create Morton codes buffer
#     T* deviceMortonCodes;
#     checkCudaError(cudaMalloc(&deviceMortonCodes, numberOfTriangles * sizeof(T)));
#     globalMemoryUsed += numberOfTriangles * sizeof(T);

#     // Array of indices that will be used in the sort algorithm
#     unsigned int* deviceIndices;
#     checkCudaError(cudaMalloc(&deviceIndices, numberOfTriangles * sizeof(unsigned int)));
#     globalMemoryUsed += numberOfTriangles * sizeof(unsigned int);

#     // Generate Morton codes
#     timeMortonCodes = DeviceGenerateMortonCodes(numberOfTriangles, deviceScene, deviceMortonCodes,
#             deviceIndices);

#     // Create sorted keys buffer
#     T* deviceSortedMortonCodes;
#     checkCudaError(cudaMalloc(&deviceSortedMortonCodes, numberOfTriangles * sizeof(T)));
#     globalMemoryUsed += numberOfTriangles * sizeof(T);

#     // Create sorted indices buffer
#     unsigned int* deviceSortedIndices;
#     checkCudaError(cudaMalloc(&deviceSortedIndices, numberOfTriangles * sizeof(unsigned int)));
#     globalMemoryUsed += numberOfTriangles * sizeof(unsigned int);

#     // Sort Morton codes
#     timeSort = DeviceSort(numberOfTriangles, &deviceMortonCodes, &deviceSortedMortonCodes,
#             &deviceIndices, &deviceSortedIndices);

#     // Free device memory
#     checkCudaError(cudaFree(deviceMortonCodes));
#     checkCudaError(cudaFree(deviceIndices));

#     // Create BVH instance
#     BVHTreeInstanceManager factory;
#     SoABVHTree* deviceTree = factory.CreateDeviceTree(numberOfTriangles);

#     // Create BVH
#     timeBuildTree = DeviceBuildTree(numberOfTriangles, deviceSortedMortonCodes, 
#             deviceSortedIndices, deviceTree);

#     // Create atomic counters buffer
#     unsigned int* deviceCounters;
#     checkCudaError(cudaMalloc(&deviceCounters, (numberOfTriangles - 1) * sizeof(unsigned int)));
#     checkCudaError(cudaMemset(deviceCounters, 0xFF, 
#             (numberOfTriangles - 1) * sizeof(unsigned int)));
#     globalMemoryUsed += (numberOfTriangles - 1) * sizeof(unsigned int);

#     // Calculate BVH nodes bounding boxes
#     timeBoundingBoxes = DeviceCalculateNodeBoundingBoxes(numberOfTriangles, deviceScene, 
#             deviceTree, deviceCounters);

#     // Free device memory
#     checkCudaError(cudaFree(deviceCounters));
#     checkCudaError(cudaFree(deviceSortedMortonCodes));
#     checkCudaError(cudaFree(deviceSortedIndices));

#     timer.Stop();

#     // Print report
#     BVHTree* hostTree = factory.DeviceToHostTree(deviceTree);
#     if (use64BitMortonCodes)
#     {
#         std::cout << std::endl << "LBVH64" << std::endl;
#     }
#     else
#     {
#         std::cout << std::endl << "LBVH32" << std::endl;
#     }    
#     std::cout << "\tBuild time: " << timer.ElapsedTime() << " ms" << std::endl;
#     std::cout << "\tSAH: " << hostTree->SAH() << std::endl;
#     std::cout << "\tKernel execution times:" << std::endl;
#     std::cout << "\t  Calculate Morton codes: " << timeMortonCodes << " ms" << std::endl;
#     std::cout << "\t       Sort Morton codes: " << timeSort << " ms" << std::endl;
#     std::cout << "\t              Build tree: " << timeBuildTree << " ms" << std::endl;
#     std::cout << "\tCalculate bounding boxes: " << timeBoundingBoxes << " ms" << std::endl;
#     std::cout << "\tGlobal memory used: " << globalMemoryUsed << " B" << std::endl << std::endl;
#     delete hostTree;

#     checkCudaError(cudaGetLastError());

#     return deviceTree;
# }

# }

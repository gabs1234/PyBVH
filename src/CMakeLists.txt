cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(UnitTestBVH CUDA)

# # This example uses the CMake Package Manager (CPM) to simplify fetching CCCL from GitHub
# # For more information, see https://github.com/cpm-cmake/CPM.cmake
# include(cmake/CPM.cmake)


# # We define these as variables so they can be overriden in CI to pull from a PR instead of CCCL `main`
# # In your project, these variables are unncessary and you can just use the values directly
# set(CCCL_REPOSITORY "nvidia/cccl" CACHE STRING "GitHub repository to fetch CCCL from")
# set(CCCL_TAG "main" CACHE STRING "Git tag/branch to fetch from CCCL repository")

# # This will automatically clone CCCL from GitHub and make the exported cmake targets available
# CPMAddPackage(
#     NAME CCCL
#     GITHUB_REPOSITORY ${CCCL_REPOSITORY}
#     GIT_TAG ${CCCL_TAG}
# )

# Default to building for the GPU on the current system
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# include opencv 
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# Add the main CUDA file and other CUDA source files
set(CUDA_SOURCES
    test/test_mesh_bvh.cu
    Quaternion.cu
    RotationQuaternion.cu
    Basis.cu
    Ray.cu
    tree_prokopenko.cu
    RayTracer.cu
    SceneManager.cu
)

# Add the executable
add_executable(UnitTestBVH ${CUDA_SOURCES})

# "Links" the CCCL Cmake target to the `UnitTestBVH` executable. This configures everything needed to use
# CCCL headers, including setting up include paths, compiler flags, etc.
# target_link_libraries(UnitTestBVH PRIVATE CCCL::CCCL)

# Specify target properties (optional)
set_target_properties(UnitTestBVH PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    # CUDA_ARCHITECTURES "60;70;75;80"  # Adjust based on your GPU's architecture
)

target_link_libraries(UnitTestBVH ${OpenCV_LIBS})
# Link libraries if necessary (usually not needed for Thrust/CUB)
# target_link_libraries(UnitTestBVH PRIVATE ${THRUST_LIBRARIES})
# target_link_libraries(UnitTestBVH PRIVATE ${CUB_LIBRARIES})
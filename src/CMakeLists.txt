cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(UnitTestBVH CUDA CXX C)

# Default to building for the GPU on the current system
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# include opencv 
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(VTK COMPONENTS 
  CommonColor
  CommonCore
  IOGeometry
  InteractionStyle
  RenderingContextOpenGL2
  RenderingCore
  RenderingFreeType
  RenderingGL2PSOpenGL2
  RenderingOpenGL2
)

if (NOT VTK_FOUND)
  message(FATAL_ERROR "ReadSTL: Unable to find the VTK build folder.")
endif()

set(RayTracerSources
    Quaternion.cu
    RotationQuaternion.cu
    Basis.cu
    Ray.cu
    tree_prokopenko.cu
    RayTracer.cu
    SceneManager.cu
)

# Add the executable
add_executable(UnitTestRay ${RayTracerSources} test/test_ray_triangle.cu)
add_executable(UnitTestBVH ${RayTracerSources} test/test_mesh_bvh.cu)

# vtk_module_autoinit is needed
vtk_module_autoinit(
  TARGETS UnitTestBVH
  MODULES ${VTK_LIBRARIES}
)

# Specify target properties (optional)
set_target_properties(UnitTestBVH PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    # CUDA_RESOLVE_DEVICE_SYMBOLS ON
    # CUDA_ARCHITECTURES "60;70;75;80"  # Adjust based on your GPU's architecture
)

# Specify target properties (optional)
set_target_properties(UnitTestRay PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    # CUDA_ARCHITECTURES "60;70;75;80"  # Adjust based on your GPU's architecture
)

# Add Debug flag
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")


target_link_libraries(UnitTestBVH ${OpenCV_LIBS})
target_link_libraries(UnitTestBVH ${VTK_LIBRARIES})
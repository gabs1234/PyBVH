#pragma once

// #include <vector_types.h>
// #include "BVHTree.h"
#include "Scene.h"

/// <summary> Calculate 30-bit Morton codes for the scene triangles. </summary>
///
/// <remarks> Leonardo, 12/30/2014. </remarks>
///
/// <param name="numberOfTriangles"> Number of triangles. </param>
/// <param name="scene">             The scene. </param>
/// <param name="mortonCodes">       [out] Morton codes. </param>
/// <param name="sortIndices">       [out] Sort indices. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceGenerateMortonCodes(unsigned int numberOfTriangles, const Scene* scene,
        unsigned int* mortonCodes, unsigned int* sortIndices);

/// <summary> Calculate 63-bit Morton codes for the scene triangles. </summary>
///
/// <remarks> Leonardo, 02/11/2015. </remarks>
///
/// <param name="numberOfTriangles"> Number of triangles. </param>
/// <param name="scene">             The scene. </param>
/// <param name="mortonCodes">       [out] Morton codes. </param>
/// <param name="sortIndices">       [out] Sort indices. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceGenerateMortonCodes(unsigned int numberOfTriangles, const Scene* scene,
        unsigned long long int* mortonCodes, unsigned int* sortIndices);

/// <summary> Calculate 30-bit Morton codes for the scene triangles when triangle splitting is
///           performed. </summary>
///
/// <remarks> Leonardo, 12/30/2014. </remarks>
///
/// <param name="numberOfTriangles"> Number of triangles. </param>
/// <param name="scene">             The scene. </param>
/// <param name="boundingBoxesMin">  [out] The bounding boxes minimum array. </param>
/// <param name="boundingBoxesMax">  [out] The bounding boxes maximum array. </param>
/// <param name="mortonCodes">       [out] Morton codes. </param>
/// <param name="routingIndices">    [out] Routing indices. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceGenerateMortonCodes(unsigned int numberOfTriangles, const Scene* scene,
        float4* boundingBoxesMin, float4* boundingBoxesMax, unsigned int* mortonCodes, 
        unsigned int* routingIndices);

/// <summary> Calculate 63-bit Morton codes for the scene triangles when triangle splitting is
///           performed. </summary>
///
/// <remarks> Leonardo, 02/11/2015. </remarks>
///
/// <param name="numberOfTriangles"> Number of triangles. </param>
/// <param name="scene">             The scene. </param>
/// <param name="boundingBoxesMin">  [out] The bounding boxes minimum array. </param>
/// <param name="boundingBoxesMax">  [out] The bounding boxes maximum array. </param>
/// <param name="mortonCodes">       [out] Morton codes. </param>
/// <param name="routingIndices">    [out] Routing indices. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceGenerateMortonCodes(unsigned int numberOfTriangles, const Scene* scene,
        float4* boundingBoxesMin, float4* boundingBoxesMax,
        unsigned long long int* mortonCodes, unsigned int* routingIndices);
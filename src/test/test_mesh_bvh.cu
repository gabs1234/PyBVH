#include <iostream>
#include <limits>
#include <math.h>
#include <opencv2/opencv.hpp>

#include "../Ray.cuh"
#include "../tree_prokopenko.cuh"
#include "../SceneManager.cuh"

using namespace std;

namespace Color {
    enum Code {
        FG_RED      = 31,
        FG_GREEN    = 32,
        FG_BLUE     = 34,
        FG_DEFAULT  = 39,
        BG_RED      = 41,
        BG_GREEN    = 42,
        BG_BLUE     = 44,
        BG_DEFAULT  = 49
    };
    class Modifier {
        Code code;
    public:
        Modifier(Code pCode) : code(pCode) {}
        friend std::ostream&
        operator<<(std::ostream& os, const Modifier& mod) {
            return os << "\033[" << mod.code << "m";
        }
    };
}

Color::Modifier red(Color::FG_RED);
Color::Modifier green(Color::FG_GREEN);

/**
 * Non-overlaping triangles
 * 1 |     |
 * .5| \   | \ [...]
 * 0 _  _  _  _
 *   0  .5 1  1.5 
 */
Scene create_non_overlaping_triangles (unsigned int N) {
    float4 *vertices = new float4[N * 3];
    float4 *bbMinLeaf = new float4[N];
    float4 *bbMaxLeaf = new float4[N];

    for (unsigned int i = 0; i < N; i++) {
        vertices[i * 3 + 0] = make_float4(i, 0.0, 0.0, 0.0);
        vertices[i * 3 + 2] = make_float4(i + .5, 0.0, 0.0, 0.0);
        vertices[i * 3 + 1] = make_float4(i, 1.0, 0.0, 0.0);

        bbMinLeaf[i] = make_float4(i, 0.0, 0.0, 0.0);
        bbMaxLeaf[i] = make_float4(i + .5, 1.0, 0.0, 0.0);
    }
    
    Scene scene;
    scene.vertices = vertices;
    scene.bbMinLeaf = bbMinLeaf;
    scene.bbMaxLeaf = bbMaxLeaf;
    scene.bbMinScene = make_float4(0, 0, 0, 0);
    scene.bbMaxScene = make_float4(N, 1, 0, 0);

    return scene;
}

void free_scene (Scene scene) {
    delete[] scene.vertices;
    delete[] scene.bbMinLeaf;
    delete[] scene.bbMaxLeaf;
}

int main() {
    // Get the bbMin and bbMax of the scene, and the bbMins and bbMaxs of the leafs
    unsigned int N = 10;
    Scene scene = create_non_overlaping_triangles(N);
    float4 *vertices = scene.vertices;
    float4 *bbMinLeaf = scene.bbMinLeaf;
    float4 *bbMaxLeaf = scene.bbMaxLeaf;
    float4 bbMinScene = scene.bbMinScene;
    float4 bbMaxScene = scene.bbMaxScene;


    // Manage the device
    SceneManager manager(
        N, bbMinLeaf, bbMaxLeaf, 
        bbMinScene, bbMaxScene, 
        vertices);

    // manager.printNodes();

    manager.setupAccelerationStructure();

    // pi
    float pi = M_PI;


    uint2 n = make_uint2(600, 400);
    float2 D = make_float2(20, 10);
    float4 spherical = make_float4(pi, 0, 1, 0);
    float4 euler = make_float4(0, 0, 0, 0);
    float4 meshOrigin = make_float4(-10, 5, 0, 0);

    float *image = manager.projectPlaneRays(n, D, spherical, euler, meshOrigin);
    // CollisionList primitives = manager.getCollisionList(2);

    // // Show the collision list
    // for (int i = 0; i < primitives.count; i++) {
    //     cout << "Collision at " << primitives.collisions[i] << endl;
    // }


    // Test the results
    cv::Mat img(n.y, n.x, CV_32F, image);
    cv::imshow("image", img);

    cv::waitKey(0);

    // Free the scene
    free_scene(scene);


    // bool output = false;

    // if (output) {
    //     cout << green << "Test " << "TODO" << " passed" << endl;
    // } else {
    //     cout << red << "Test " << "TODO" << " failed" << endl;
    // }

    return 0;
}
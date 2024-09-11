#include <iostream>
#include <limits>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <random>

#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSTLReader.h>
#include <vtkOBJReader.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkPolyVertex.h>
#include <vtkCell.h>
#include <vtkIdList.h>

#include "../SceneManager.cuh"
#include "../Ray.cuh"
#include "../tree_prokopenko.cuh"

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
Color::Modifier def(Color::FG_DEFAULT);

Scene extract_scene_from_polydata (vtkPolyData *polydata) {
    vtkPoints *points = polydata->GetPoints();
    vtkCellArray *cells = polydata->GetPolys();
    double bounds[6];
    polydata->GetBounds(bounds);

    vtkIdType numberOfCells = polydata->GetNumberOfCells();

    float4 *vertices = new float4[numberOfCells * 3];
    float4 *bbMinLeaf = new float4[numberOfCells];
    float4 *bbMaxLeaf = new float4[numberOfCells];

    for (vtkIdType cellId = 0; cellId < numberOfCells; ++cellId) {
        // Get the current cell
        vtkSmartPointer<vtkCell> cell = polydata->GetCell(cellId);

        if ( cell->GetNumberOfPoints() != 3 ) {
            std::cout << "Error: The cell does not have 3 points." << std::endl;
            break;
        }

        // Iterate through the points in the current cell
        vtkPoints *cellPoints = cell->GetPoints();
        for (vtkIdType i = 0; i < cell->GetNumberOfPoints(); ++i) {
            double point[3];
            cellPoints->GetPoint(i, point);
            vertices[cellId * 3 + i] = make_float4(point[0], point[1], point[2], 0);
        }


        // Calculate the bounding box of the triangle
        double bounds[6];
        cell->GetBounds(bounds);
        bbMinLeaf[cellId] = make_float4(bounds[0], bounds[2], bounds[4], 0);
        bbMaxLeaf[cellId] = make_float4(bounds[1], bounds[3], bounds[5], 0);
    }

    Scene scene;
    scene.nb_keys = numberOfCells;
    scene.vertices = vertices;
    scene.bbMinLeaf = bbMinLeaf;
    scene.bbMaxLeaf = bbMaxLeaf;
    scene.bbMinScene = make_float4(bounds[0], bounds[2], bounds[4], 0);
    scene.bbMaxScene = make_float4(bounds[1], bounds[3], bounds[5], 0);

    return scene;
}

Scene make_scene_from_stl (std::string filename, bool visualize) {
    vtkNew<vtkSTLReader> reader;
    reader->SetFileName(filename.c_str());
    reader->Update();

    if (visualize) {
        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputConnection(reader->GetOutputPort());

        vtkNew<vtkNamedColors> colors;

        vtkNew<vtkActor> actor;
        actor->SetMapper(mapper);
        actor->GetProperty()->SetDiffuse(0.8);
        actor->GetProperty()->SetDiffuseColor(colors->GetColor3d("LightSteelBlue").GetData());
        actor->GetProperty()->SetSpecular(0.3);
        actor->GetProperty()->SetSpecularPower(60.0);

        vtkNew<vtkRenderer> renderer;
        vtkNew<vtkRenderWindow> renderWindow;
        renderWindow->AddRenderer(renderer);
        renderWindow->SetWindowName("ReadSTL");

        vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
        renderWindowInteractor->SetRenderWindow(renderWindow);

        renderer->AddActor(actor);
        renderer->SetBackground(colors->GetColor3d("DarkOliveGreen").GetData());

        renderWindow->Render();
        renderWindowInteractor->Start();
    }

    std::cout << "Read " << filename << std::endl;

    return extract_scene_from_polydata(reader->GetOutput());
}

Scene make_scene_from_wavefront (std::string filename, bool visualize) {
    vtkNew<vtkOBJReader> reader;
    reader->SetFileName(filename.c_str());
    reader->Update();

    if (visualize) {
        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputConnection(reader->GetOutputPort());

        vtkNew<vtkNamedColors> colors;

        vtkNew<vtkActor> actor;
        actor->SetMapper(mapper);
        actor->GetProperty()->SetDiffuse(0.8);
        actor->GetProperty()->SetDiffuseColor(colors->GetColor3d("LightSteelBlue").GetData());
        actor->GetProperty()->SetSpecular(0.3);
        actor->GetProperty()->SetSpecularPower(60.0);

        vtkNew<vtkRenderer> renderer;
        vtkNew<vtkRenderWindow> renderWindow;
        renderWindow->AddRenderer(renderer);
        renderWindow->SetWindowName("ReadSTL");

        vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
        renderWindowInteractor->SetRenderWindow(renderWindow);

        renderer->AddActor(actor);
        renderer->SetBackground(colors->GetColor3d("DarkOliveGreen").GetData());

        renderWindow->Render();
        renderWindowInteractor->Start();
    }

    std::cout << "Read " << filename << std::endl;

    return extract_scene_from_polydata(reader->GetOutput());
}

/**
 * Non-overlaping triangles
 * 1 |     |
 * .5| \   | \ [...]
 * 0 _  _  _  _
 *   0  .5 1  1.5 
 */
Scene create_non_overlaping_triangles (unsigned int N, float3 D) {
    float4 *vertices = new float4[N * 3];
    float4 *bbMinLeaf = new float4[N];
    float4 *bbMaxLeaf = new float4[N];

    float dx = D.x/N;
    float dy = D.y/2;
    float dz = D.z/N;

    for (unsigned int i = 0; i < N; i++) {
        vertices[i * 3 + 0] = make_float4(i * dx, 0.0, i * dz, 0.0);
        vertices[i * 3 + 2] = make_float4((i + .5)* dx, 0.0, i * dz, 0.0);
        vertices[i * 3 + 1] = make_float4(i * dx, dy, i * dz, 0.0);
    }

    for (unsigned int i = 0; i < N; i++) {
        bbMinLeaf[i] = make_float4(i * dx, 0.0, i * dz, 0.0);
        bbMaxLeaf[i] = make_float4((i + .5) * dx, dy, i * dz, 0.0);
    }
    
    Scene scene;
    scene.nb_keys = N;
    scene.vertices = vertices;
    scene.bbMinLeaf = bbMinLeaf;
    scene.bbMaxLeaf = bbMaxLeaf;
    scene.bbMinScene = make_float4(0, 0, 0, 0);
    scene.bbMaxScene = make_float4(D.x, dy, D.z, 0);

    return scene;
}

Scene create_simple_scene (unsigned int N, float3 D) {
    long unsigned int n_trig = N * (N + 1) / 2;
    long unsigned int n_vertices = n_trig * 3;
    std::cout << "n vertices = " << n_vertices << std::endl;
    float4 *vertices = new float4[n_vertices];
    float4 *bbMinLeaf = new float4[n_trig];
    float4 *bbMaxLeaf = new float4[n_trig];

    float dx = D.x/(N-1);
    float dy = D.y;
    float dz = D.z;

    float y = dy;

    int v_index = 0, index = 0;
    for (unsigned int ix=0; ix < N; ix++) {
        float x = ix * dx;
        for (unsigned int iz=0; iz <= ix; iz++) {
            float z = iz * dz;
            if (iz % 2 == 0) {
                vertices[v_index + 1] = make_float4(x, 0, z , 0);
                vertices[v_index + 2] = make_float4(x + dx/2, 0, z , 0);
                vertices[v_index] = make_float4(x, y, z, 0);
            } else {
                vertices[v_index] = make_float4(x, 0, z  , 0);
                vertices[v_index + 1] = make_float4(x + dx/2, 0, z , 0);
                vertices[v_index + 2] = make_float4(x, y, z, 0);
            }
            

            // printf ("v_index = %d, x = %f, y = %f, z = %f\n", v_index, x, y, z);

            bbMinLeaf[index] = make_float4(x, 0, z , 0);
            bbMaxLeaf[index] = make_float4(x + dx/2, y, z, 0);

            v_index += 3;
            index++;
        }
    }
    Scene scene;
    scene.nb_keys = n_trig;
    scene.vertices = vertices;
    scene.bbMinLeaf = bbMinLeaf;
    scene.bbMaxLeaf = bbMaxLeaf;
    scene.bbMinScene = make_float4(0, 0, 0, 0);
    scene.bbMaxScene = make_float4(D.x + dx/2, dy, D.z, 0);

    return scene;
}

void calculateTriangleBoundingBox(
    float4 const &vertex1, float4 const &vertex2, float4 const &vertex3,
    float4& boundingBoxMin, float4& boundingBoxMax) {
    boundingBoxMin.x = min(vertex1.x, vertex2.x);
    boundingBoxMin.x = min(boundingBoxMin.x, vertex3.x);
    boundingBoxMax.x = max(vertex1.x, vertex2.x);
    boundingBoxMax.x = max(boundingBoxMax.x, vertex3.x);

    boundingBoxMin.y = min(vertex1.y, vertex2.y);
    boundingBoxMin.y = min(boundingBoxMin.y, vertex3.y);
    boundingBoxMax.y = max(vertex1.y, vertex2.y);
    boundingBoxMax.y = max(boundingBoxMax.y, vertex3.y);

    boundingBoxMin.z = min(vertex1.z, vertex2.z);
    boundingBoxMin.z = min(boundingBoxMin.z, vertex3.z);
    boundingBoxMax.z = max(vertex1.z, vertex2.z);
    boundingBoxMax.z = max(boundingBoxMax.z, vertex3.z);
}


Scene create_random_scene (unsigned int N, float3 D) {
    float4 *vertices = new float4[N * 3];
    float4 *bbMinLeaf = new float4[N];
    float4 *bbMaxLeaf = new float4[N];

    float resolution = 1000;
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist (0, resolution);

    for (unsigned int i = 0; i < N; i++) {
        for (unsigned int j = 0; j < 3; j++) {
            float x = dist(rng) * D.x / resolution;
            float y = dist(rng) * D.y / resolution;
            float z = dist(rng) * D.z / resolution;
            
            x -= D.x/2;
            y -= D.y/2;
            z -= D.z/2;
            vertices[i * 3 + j] = make_float4(x, y, z, 0);

            // printf ("(%f, %f, %f)\n", x, y, z);
        }

        calculateTriangleBoundingBox(
            vertices[i * 3 + 0], vertices[i * 3 + 1], vertices[i * 3 + 2],
            bbMinLeaf[i], bbMaxLeaf[i]);
    }

    Scene scene;
    scene.nb_keys = N;
    scene.vertices = vertices;
    scene.bbMinLeaf = bbMinLeaf;
    scene.bbMaxLeaf = bbMaxLeaf;
    scene.bbMinScene = make_float4(-D.x/2, -D.y/2, -D.z/2, 0);
    scene.bbMaxScene = make_float4(D.x/2, D.y/2, D.z/2, 0);

    return scene;
}

void free_scene (Scene scene) {
    delete[] scene.vertices;
    delete[] scene.bbMinLeaf;
    delete[] scene.bbMaxLeaf;
}

bool project_mesh (std::string filename) {
    // Get the bbMin and bbMax of the scene, and the bbMins and bbMaxs of the leafs
    std::cout << "Create the scene" << std::endl;
    Scene scene = make_scene_from_stl (filename, false);

    // Print all vertices
    std::cout << "nb keys" << scene.nb_keys << std::endl;
    // Manage the device
    SceneManager manager(scene.vertices, scene.nb_keys, scene.bbMinScene, scene.bbMaxScene);
    manager.setupAccelerationStructure();

    // manager.getTreeStructure();

    // manager.sanityCheck();


    float pi = M_PI;

    float2 viewport = make_float2(scene.bbMaxScene.x * 2.2, scene.bbMaxScene.y * 2.2);
    int mulx = viewport.x * 1500;
    int muly = viewport.y * 1500;
    uint2 n = make_uint2(mulx, muly);
    float4 spherical = make_float4(0, 0, 100, 0);
    float4 euler = make_float4(0, 0, 0, 0); // yaw, pitch, roll
    float4 meshOrigin = make_float4(scene.bbMinScene.x / 2, scene.bbMinScene.y / 2, scene.bbMinScene.z / 2, 0);

    float *image = manager.projectPlaneRays(n, viewport, spherical, euler, meshOrigin);

    // Test the results
    cv::Mat img(n.y, n.x, CV_32F, image);
    
    // Normalize the image
    cv::normalize(img, img, 0, 256, cv::NORM_MINMAX);
    
    // Save the image
    std::string output = filename + ".png";
    cv::imwrite(output, img);

    // Free the scene
    std::cout << "Free the scene" << std::endl;
    free_scene(scene);

    return true;
}


void test_intersection (unsigned int N) {
    float3 extents = make_float3(1, 1, 1);
    std::cout << "Create the scene" << std::endl;
    Scene scene = create_simple_scene (N, extents);

    // Print all vertices
    std::cout << "nb keys" << scene.nb_keys << std::endl;
    // for (int i = 0; i < scene.nb_keys; i++) {
    //     std::cout << "Triangle " << i << std::endl;
    //     for (int j = 0; j < 3; j++) {
    //         float4 v = scene.vertices[i * 3 + j];
    //         std::cout << "V" << j << " = (" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
    //     }
    // }

    // Manage the device
    SceneManager manager(scene);

    manager.setupAccelerationStructure();

    // manager.getTreeStructure();

    // Test the intersection
    for (int i = 0; i < N; i++) {
        unsigned int row = i;
        unsigned int actual_id = row * (row + 1) / 2 * 3;
        CollisionList collisions = manager.getCollisionList(actual_id);

        if (collisions.count == 0) {
            std::cout << "No collision" << std::endl;
        }
        else {
            if (collisions.count != i + 1) {
                std::cout << red << "Error: expected " << i + 1 << " collisions, got " << collisions.count << green << std::endl;
            }
            else {
                cout << "Collision " << i << std::endl;
                for (int j = 0; j < collisions.count; j++) {
                    std::cout << "Collision " << j << " = " << collisions.collisions[j] << std::endl;
                }
            }
            
        }
    }
    
    free_scene(scene);
}

void project_random_scene (unsigned int N, float3 D) {
    Scene scene = create_random_scene (N, D);

    // Print all vertices
    std::cout << "nb keys" << scene.nb_keys << std::endl;

    // Manage the device
    SceneManager manager(scene);
    manager.setupAccelerationStructure();

    manager.getTreeStructure();

    if (!manager.sanityCheck()) {
        std::cout << red << "Error: sanity check failed" << std::endl;
        return;
    }
    else {
        std::cout << def << "Sanity check passed" << std::endl;
    }

    float2 viewport = make_float2(D.x,D.y);
    int mulx = viewport.x;
    int muly = viewport.y;
    uint2 n = make_uint2(mulx, muly);
    // float pi = M_PI;
    float4 spherical = make_float4(0, 0, 10, 0);
    float4 euler = make_float4(0, 0, 0, 0); // yaw, pitch, roll
    float4 meshOrigin = make_float4(0, 0, 0, 0);

    float *image = manager.projectPlaneRays(n, viewport, spherical, euler, meshOrigin);

    // Test the results
    cv::Mat img(n.y, n.x, CV_32F, image);
    
    // Normalize the image
    cv::normalize(img, img, 0, 256, cv::NORM_MINMAX);
    
    // Save the image
    std::string output = "random_triangle_scene.png";
    cv::imwrite(output, img);

    // Free the scene
    std::cout << "Free the scene" << std::endl;
    free_scene(scene);
}

int main(int argc, char **argv) {
    // Input N and D
    if (argc != 5) {
        std::cout << "Usage: test_mesh_bvh N Dx Dy Dz" << std::endl;
        return 1;
    }

    unsigned int N = atoi(argv[1]);

    float Dx = atof(argv[2]);
    float Dy = atof(argv[3]);
    float Dz = atof(argv[4]);
    float3 D = make_float3(Dx, Dy, Dz);

    // std::string filenames[] = {
    //     // "../blender/255.stl",
    //     // "../blender/256.stl",
    //     "../blender/monkey.stl",
    //     "../blender/cube.stl",
    //     "../blender/sphere.stl",
    //     "../blender/torus.stl",
    //     // "../blender/surface.stl",
    //     "../blender/phantom.stl",
    //     // "../blender/triangle.stl",
    //     // "../blender/triangle2.stl"
    // };
    // for (int i = 0; i < 10; i++) {
    //     std::string filename = filenames[i];
    //     std::cout << "Projecting " << filename << std::endl;
    //     project_mesh(filename);
    // }

    // test_intersection (10);

    project_random_scene(N, D);

    // // Big monkey
    // project_mesh("../blender/big_monkey.stl");

    return 0;
}
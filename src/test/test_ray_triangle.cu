#include <iostream>
#include <limits>
#include <cmath>

#include "../Ray.cuh"

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

typedef enum {
    EQUAL_TO,
    NOT_EQUAL_TO,
    SMALLER_THAN,
    STRICTLY_SMALLER_THAN,
    GREATER_THAN,
    STRICTLY_GREATER_THAN,
    INDETERMINATE
} Comparison;

typedef struct {
    float t;
    bool intersects;
    bool expected;
    float expected_t;
    Comparison comparison_t;
} TestResult;

typedef struct {
    const char *name;
    const char *description;
    TestResult (*test)();
} TestUnit;

TestResult test_0 () {
    float4 tail = make_float4(0.5f, 0.5f, 0.0f, 0.0f);
    float4 direction = make_float4(1.0f, 1.0f, 1.0f, 0.0f);
    float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 v1 = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    float4 v2 = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    
    Ray ray = Ray(tail, direction);

    float t = 0;

    bool res = ray.intersects(v0, v1, v2, t);

    TestResult result = {
        .t = t,
        .intersects = res,
        .expected = true,
        .expected_t = 0.0f,
        .comparison_t = EQUAL_TO
    };

    return result;
}

TestResult test_1 () {
    float4 tail = make_float4(0, 0, 0, 0);
    float4 direction = make_float4(1.0f, 1.0f, 1.0f, 0.0f);
    float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 v1 = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    float4 v2 = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    
    Ray ray = Ray(tail, direction);

    float t = 0;

    bool res = ray.intersects(v0, v1, v2, t);
    
    TestResult result = {
        .t = t,
        .intersects = res,
        .expected = true,
        .expected_t = 0.0f,
        .comparison_t = EQUAL_TO
    };

    return result;
}

TestResult test_2 () {
    float4 tail = make_float4(0.5f, 0.5f, 0.0f, 0.0f);
    float4 direction = make_float4(1, 1, 0, 0);
    float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 v1 = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    float4 v2 = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    
    Ray ray = Ray(tail, direction);

    float t = 0;

    bool res = ray.intersects(v0, v1, v2, t);

    TestResult result = {
        .t = t,
        .intersects = res,
        .expected = true,
        .expected_t = 0.0f,
        .comparison_t = EQUAL_TO
    };

    return result;
}

TestResult test_3 () {
    float4 tail = make_float4(0, 0, 0.0f, 0.0f);
    float4 direction = make_float4(1, 1, 0, 0);
    float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 v1 = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    float4 v2 = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    
    Ray ray = Ray(tail, direction);

    float t = 0;

    bool res = ray.intersects(v0, v1, v2, t);

    TestResult result = {
        .t = t,
        .intersects = res,
        .expected = true,
        .expected_t = 0.0f,
        .comparison_t = EQUAL_TO
    };

    return result;
}

TestResult test_4 () {
    float4 tail = make_float4(0.5f, 0.5f, 1.0f, 0.0f);
    float4 direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f);
    float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 v1 = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    float4 v2 = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    
    Ray ray = Ray(tail, direction);

    float t = 0;

    bool res = ray.intersects(v0, v1, v2, t);

    TestResult result = {
        .t = t,
        .intersects = res,
        .expected = true,
        .expected_t = 0.0f,
        .comparison_t = STRICTLY_GREATER_THAN
    };

    return result;
}

TestResult test_5 () {
    float4 tail = make_float4(0, 0, 1.0f, 0.0f);
    float4 direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f);
    float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 v1 = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    float4 v2 = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    
    Ray ray = Ray(tail, direction);

    float t = 0;

    bool res = ray.intersects(v0, v1, v2, t);

    TestResult result = {
        .t = t,
        .intersects = res,
        .expected = true,
        .expected_t = 0.0f,
        .comparison_t = STRICTLY_GREATER_THAN
    };

    return result;
}

TestResult test_6 () {
    float4 tail = make_float4(0.5f, 0.5f, 1.0f, 0.0f);
    float4 direction = make_float4(-1, -1, 0, 0.0f);
    float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 v1 = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    float4 v2 = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    
    Ray ray = Ray(tail, direction);

    float t = 0;

    bool res = ray.intersects(v0, v1, v2, t);

    TestResult result = {
        .t = t,
        .intersects = res,
        .expected = false,
        .expected_t = 0.0f,
        .comparison_t = INDETERMINATE
    };

    return result;
}

TestResult test_7 () {
    float4 tail = make_float4(0.5f, 0.5f, 1.0f, 0.0f);
    float4 direction = make_float4(0, 0, 1, 0.0f);
    float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 v1 = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    float4 v2 = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    
    Ray ray = Ray(tail, direction);

    float t = 0;

    bool res = ray.intersects(v0, v1, v2, t);

    TestResult result = {
        .t = t,
        .intersects = res,
        .expected = false,
        .expected_t = 0.0f,
        .comparison_t = INDETERMINATE
    };

    return result;
}

TestResult test_8 () {
    float4 tail = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    float4 direction = make_float4(0, 0, 1, 0.0f);
    float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 v1 = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    float4 v2 = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    
    Ray ray = Ray(tail, direction);

    float t = 0;

    bool res = ray.intersects(v0, v1, v2, t);

    TestResult result = {
        .t = t,
        .intersects = res,
        .expected = true,
        .expected_t = 0.0f,
        .comparison_t = EQUAL_TO
    };

    return result;
}

TestResult test_9 () {
    float epsilon = std::numeric_limits<float>::epsilon();
    float4 tail = make_float4(epsilon, epsilon, epsilon, 0.0f);
    float4 direction = make_float4(0, 0, -1, 0.0f);
    float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 v1 = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    float4 v2 = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    
    Ray ray = Ray(tail, direction);

    float t = 0;

    bool res = ray.intersects(v0, v1, v2, t);

    TestResult result = {
        .t = t,
        .intersects = res,
        .expected = true,
        .expected_t = 0.0f,
        .comparison_t = STRICTLY_GREATER_THAN
    };

    return result;
}

TestResult test_10 () {
    float epsilon = std::numeric_limits<float>::epsilon()/ 2.0;
    float4 tail = make_float4(-epsilon, -epsilon, epsilon, 0.0f);
    float4 direction = make_float4(0, 0, -1, 0.0f);
    float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 v1 = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    float4 v2 = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    
    Ray ray = Ray(tail, direction);

    float t = 0;

    bool res = ray.intersects(v0, v1, v2, t);

    TestResult result = {
        .t = t,
        .intersects = res,
        .expected = false,
        .expected_t = 0.0f,
        .comparison_t = INDETERMINATE
    };

    return result;
}

float4 add (float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0.0f);
}

float4 substract (float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0.0f);
}

float scalar_product (float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float squared_sum (float4 a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

float calculate_t_value_sphere (float4 Or, float4 direction, float4 Os, float radius, float &t1, float &t2) {
    float4 O = substract(Or, Os);
    float od = scalar_product(O, direction);
    float delta = 4 * od * od - 4 * squared_sum(direction) * (squared_sum(O) - radius * radius);

    if (delta < 0) {
        return -1;
    }

    if (delta == 0) {
        return -od / (2 * squared_sum(direction));
    }

    t1 = (-od + sqrt(delta)) / (2 * squared_sum(direction));
    t2 = (-od - sqrt(delta)) / (2 * squared_sum(direction));

    printf("theory => t1: %f, t2: %f\n", t1, t2);

    return t1;
}

float4 spherical_to_cartesian (float theta, float phi, float r) {
    return make_float4(
        r * sin(theta) * cos(phi),
        r * sin(theta) * sin(phi),
        r * cos(theta),
        0.0f
    );
}

void triangle_on_sphere (
    float4 &v0, float4 &v1, float4 &v2,
    float4 Os, float radius,
    float epsilon, float theta, float phi) {
    v0 = add(Os, spherical_to_cartesian(theta, phi, radius));
    v1 = add(Os, spherical_to_cartesian(theta, phi + epsilon, radius));
    v2 = add(Os, spherical_to_cartesian(theta + epsilon, phi, radius));
}

TestResult test_11 () {
    // Calculate the triangle on the sphere
    float4 v0, v1, v2;
    float4 Os = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float radius = 1.0f;
    float epsilon = 0.01f;
    float theta = M_PI / 4;
    float phi = M_PI / 4;

    triangle_on_sphere(v0, v1, v2, Os, radius, epsilon, theta, phi);

    // Define the ray
    float4 tail = v0;
    float4 direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f);
    Ray ray = Ray(tail, direction);

    float t = 0;

    bool res = ray.intersects(v0, v1, v2, t);

    float t1 = 0;
    float t2 = 0;

    TestResult result = {
        .t = t,
        .intersects = res,
        .expected = true,
        .expected_t = calculate_t_value_sphere(tail, direction, Os, radius, t1, t2),
        .comparison_t = EQUAL_TO
    };

    return result;
}
bool compare(float a, float b, Comparison comparison) {
    switch (comparison) {
        case EQUAL_TO:
            return a == b;
        case NOT_EQUAL_TO:
            return a != b;
        case SMALLER_THAN:
            return a <= b;
        case STRICTLY_SMALLER_THAN:
            return a < b;
        case GREATER_THAN:
            return a >= b;
        case STRICTLY_GREATER_THAN:
            return a > b;
        case INDETERMINATE:
            return true;
        default:
            return false;
    }
}

int main() {
    int nb_tests = 12;
    TestUnit test_units[] = {
        {
            .name = "test_0",
            .description = "tail is contained in the triangle (edge excluded), direction is non-parallel to the triangle, triangle is a right triangle",
            .test = test_0,
        },
        {
            .name = "test_1",
            .description = "tail is on the edge of the triangle, direction is non-parallel to the triangle, triangle is a right triangle",
            .test = test_1,
        },
        {
            .name = "test_2",
            .description = "tail is contained in the triangle (edge excluded), direction is parallel to the triangle, triangle is a right triangle",
            .test = test_2,
        },
        {
            .name = "test_3",
            .description = "tail is on the edge of the triangle, direction is parallel to the triangle, triangle is a right triangle",
            .test = test_3,
        },
         // now the tail is outside the triangle
        {
            .name = "test_4",
            .description = "tail is not in the triangle (on the positive side of z axis), direction is towards the inside of the triangle (towards negative z axis), triangle is a right triangle",
            .test = test_4,
        },
        {
            .name = "test_5",
            .description = "tail is not in the triangle (on the positive side of z axis), direction is towards the edge of the triangle (towards negative z axis), triangle is a right triangle",
            .test = test_5,
        },
        {
            .name = "test_6",
            .description = "tail is not in the triangle (on the positive side of z axis), direction is parallel to the triangle, triangle is a right triangle",
            .test = test_6,
        },
        {
            .name = "test_7",
            .description = "tail is not in the triangle (on the positive side of z axis), direction is away from triangle (towards + z axis), triangle is a right triangle",
            .test = test_7,
        },
        {
            .name = "test_8",
            .description = "tail is on the edge of the triangle, direction is away from the triangle, triangle is a right triangle",
            .test = test_8,
        },
        {
            .name = "test_9",
            .description = "Test numerical precision of the algorithm: tail is on the edge of the triangle, direction is to the triangle, triangle is a right triangle",
            .test = test_9,
        },
        {
            .name = "test_10",
            .description = "Test numerical precision of the algorithm: tail is outside edge of the triangle, direction is normal to the triangle, triangle is a right triangle",
            .test = test_10,
        },
        {
            .name = "test_11",
            .description = "Test the intersection of a ray with a triangle on a sphere",
            .test = test_11,
        }
    };

    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);

    for (int i = 0; i < nb_tests; i++) {
        TestUnit test_unit = test_units[i];
        TestResult result = test_unit.test();

        bool output = (compare(result.t, result.expected_t, result.comparison_t)) && result.intersects == result.expected;

        if (output) {
            cout << green << "Test " << test_unit.name << " passed" << endl;
        } else {
            cout << red << "Test " << test_unit.name << " failed" << endl;
            cout << red << "Description: " << test_unit.description << endl;
            cout << red << "Expected: " << result.expected << endl;
            cout << red << "Expected t: " << result.expected_t << endl;
            cout << red << "Expected comparison: " << result.comparison_t << endl;
            cout << red << "Got: " << result.intersects << endl;
            cout << red << "Got t: " << result.t << endl;
        }

    }


    return 0;
}
#include <iostream>
#include <limits>
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
} TestResult;

typedef struct {
    const char *name;
    const char *description;
    TestResult (*test)();
    bool expected;
    float expected_t;
    Comparison comparison_t;
} TestUnit;

TestResult test_0 () {
    float4 tail = make_float4(0.5f, 0.5f, 0.0f, 0.0f);
    float4 direction = make_float4(1.0f, 1.0f, 1.0f, 0.0f);
    float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 v1 = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    float4 v2 = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    
    Ray ray = Ray(tail, direction);

    float t = 0.0f;

    bool res = ray.intersects(v0, v1, v2, t);

    TestResult result = {
        .t = t,
        .intersects = res
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

    float t = 0.0f;

    bool res = ray.intersects(v0, v1, v2, t);
    
    TestResult result = {
        .t = t,
        .intersects = res
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

    float t = 0.0f;

    bool res = ray.intersects(v0, v1, v2, t);

    TestResult result = {
        .t = t,
        .intersects = res
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

    float t = 0.0f;

    bool res = ray.intersects(v0, v1, v2, t);

    TestResult result = {
        .t = t,
        .intersects = res
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

    float t = 0.0f;

    bool res = ray.intersects(v0, v1, v2, t);

    TestResult result = {
        .t = t,
        .intersects = res
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

    float t = 0.0f;

    bool res = ray.intersects(v0, v1, v2, t);

    TestResult result = {
        .t = t,
        .intersects = res
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

    float t = 0.0f;

    bool res = ray.intersects(v0, v1, v2, t);

    TestResult result = {
        .t = t,
        .intersects = res
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

    float t = 0.0f;

    bool res = ray.intersects(v0, v1, v2, t);

    TestResult result = {
        .t = t,
        .intersects = res
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

    float t = 0.0f;

    bool res = ray.intersects(v0, v1, v2, t);

    TestResult result = {
        .t = t,
        .intersects = res
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

    float t = 0.0f;

    bool res = ray.intersects(v0, v1, v2, t);

    TestResult result = {
        .t = t,
        .intersects = res
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

    float t = 0.0f;

    bool res = ray.intersects(v0, v1, v2, t);

    TestResult result = {
        .t = t,
        .intersects = res
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
    int nb_tests = 11;
    TestUnit test_units[] = {
        {
            .name = "test_0",
            .description = "tail is contained in the triangle (edge excluded), direction is non-parallel to the triangle, triangle is a right triangle",
            .test = test_0,
            .expected = true,
            .expected_t = 0.0f,
            .comparison_t = EQUAL_TO
        },
        {
            .name = "test_1",
            .description = "tail is on the edge of the triangle, direction is non-parallel to the triangle, triangle is a right triangle",
            .test = test_1,
            .expected = true,
            .expected_t = 0.0f,
            .comparison_t = EQUAL_TO
        },
        {
            .name = "test_2",
            .description = "tail is contained in the triangle (edge excluded), direction is parallel to the triangle, triangle is a right triangle",
            .test = test_2,
            .expected = false,
            .expected_t = 0.0f,
            .comparison_t = EQUAL_TO
        },
        {
            .name = "test_3",
            .description = "tail is on the edge of the triangle, direction is parallel to the triangle, triangle is a right triangle",
            .test = test_3,
            .expected = false,
            .expected_t = 0.0f,
            .comparison_t = EQUAL_TO
        },
         // now the tail is outside the triangle
        {
            .name = "test_4",
            .description = "tail is not in the triangle (on the positive side of z axis), direction is towards the inside of the triangle (towards negative z axis), triangle is a right triangle",
            .test = test_4,
            .expected = true,
            .expected_t = 0.0f,
            .comparison_t = STRICTLY_GREATER_THAN
        },
        {
            .name = "test_5",
            .description = "tail is not in the triangle (on the positive side of z axis), direction is towards the edge of the triangle (towards negative z axis), triangle is a right triangle",
            .test = test_5,
            .expected = true,
            .expected_t = 0.0f,
            .comparison_t = STRICTLY_GREATER_THAN
        },
        {
            .name = "test_6",
            .description = "tail is not in the triangle (on the positive side of z axis), direction is parallel to the triangle, triangle is a right triangle",
            .test = test_6,
            .expected = false,
            .expected_t = 0.0f,
            .comparison_t = INDETERMINATE
        },
        {
            .name = "test_7",
            .description = "tail is not in the triangle (on the positive side of z axis), direction is away from triangle (towards + z axis), triangle is a right triangle",
            .test = test_7,
            .expected = false,
            .expected_t = 0.0f,
            .comparison_t = INDETERMINATE
        },
        {
            .name = "test_8",
            .description = "tail is on the edge of the triangle, direction is away from the triangle, triangle is a right triangle",
            .test = test_8,
            .expected = true,
            .expected_t = 0.0f,
            .comparison_t = EQUAL_TO
        },
        {
            .name = "test_9",
            .description = "Test numerical precision of the algorithm: tail is on the edge of the triangle, direction is to the triangle, triangle is a right triangle",
            .test = test_9,
            .expected = true,
            .expected_t = 0.0f,
            .comparison_t = STRICTLY_GREATER_THAN
        },
        {
            .name = "test_10",
            .description = "Test numerical precision of the algorithm: tail is outside edge of the triangle, direction is normal to the triangle, triangle is a right triangle",
            .test = test_10,
            .expected = false,
            .expected_t = 0.0f,
            .comparison_t = INDETERMINATE
        }
    };

    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);

    for (int i = 0; i < nb_tests; i++) {
        TestUnit test_unit = test_units[i];
        TestResult result = test_unit.test();

        bool output = (compare(result.t, test_unit.expected_t, test_unit.comparison_t)) && result.intersects == test_unit.expected;

        if (output) {
            cout << green << "Test " << test_unit.name << " passed" << endl;
        } else {
            cout << red << "Test " << test_unit.name << " failed" << endl;
            cout << red << "Description: " << test_unit.description << endl;
            cout << red << "Expected: " << test_unit.expected << endl;
            cout << red << "Expected t: " << test_unit.expected_t << endl;
            cout << red << "Expected comparison: " << test_unit.comparison_t << endl;
            cout << red << "Got: " << result.intersects << endl;
            cout << red << "Got t: " << result.t << endl;
        }

    }


    return 0;
}
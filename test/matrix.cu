#include <catch2/catch.hpp>
#include "util.cuh"
#include "check_cuda_errors.h"
#include <cmath>

TEST_CASE("mat4 immutable operations", "[util]") {
    pathtracer::mat4 mat1{0,1,2,3,
                          4,5,6,7,
                          8,9,10,11,
                          12,13,14,15};

    pathtracer::mat4 mat2{0.25f,1.25f,2.25f,3.25f,
                          4.25f,5.25f,6.25f,7.25f,
                          8.25f,9.25f,10.25f,11.25f,
                          12.25f,13.25f,14.25f,15.25f};

    SECTION("mat4 equality") {
        bool res = (mat1 == mat2);

        REQUIRE(res == false);

        pathtracer::mat4 mat3{0.25f,1.25f,2.25f,3.25f,
                          4.25f,5.25f,6.25f,7.25f,
                          8.25f,9.25f,10.25f,11.25f,
                          12.25f,13.25f,14.25f,15.25f};

        res = (mat2 == mat3);

        REQUIRE(res == true);
    }

    SECTION("mat4 mat4 multiplication") {
        pathtracer::mat4 expected{57.5,63.5,69.5,75.5,
                                  157.5,179.5,201.5,223.5,
                                  257.5,295.5,333.5,371.5,
                                  357.5,411.5,465.5,519.5};

        bool res = ((mat1 * mat2) == expected);

        REQUIRE(res == true);
    }
}

TEST_CASE("mat4 mutable operations", "[util]") {
    pathtracer::mat4 mat1{0,1,2,3,
                          4,5,6,7,
                          8,9,10,11,
                          12,13,14,15};

    pathtracer::mat4 mat2{0.25f,1.25f,2.25f,3.25f,
                          4.25f,5.25f,6.25f,7.25f,
                          8.25f,9.25f,10.25f,11.25f,
                          12.25f,13.25f,14.25f,15.25f};

    SECTION("mat4 mat4 multiplication") {
        pathtracer::mat4 expected{57.5,63.5,69.5,75.5,
                                  157.5,179.5,201.5,223.5,
                                  257.5,295.5,333.5,371.5,
                                  357.5,411.5,465.5,519.5};

        bool res = ((mat1 * mat2) == expected);

        REQUIRE(res == true);
    }
}
#include <catch2/catch.hpp>
#include "util.cuh"
#include "check_cuda_errors.h"
#include <cmath>
#include <iostream>

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

    SECTION("Identity factory") {
        pathtracer::mat4 expected{1.f, 0.f, 0.f, 0.f,
                                  0.f, 1.f, 0.f, 0.f,
                                  0.f, 0.f, 1.f, 0.f,
                                  0.f, 0.f, 0.f, 1.f};
    }

    SECTION("Transpose") {
        pathtracer::mat4 expected{0,4,8,12,
                                  1,5,9,13,
                                  2,6,10,14,
                                  3,7,11,15};

        bool res = (mat1.transpose() == expected);

        REQUIRE(res == true);
    }

    SECTION("Inverse") {
        pathtracer::mat4 mat3{8,-5,9,2,
                              7,5,6,1,
                              -6,0,9,6,
                              -3,0,-9,-4};

        pathtracer::mat4 expected{-0.15385 , -0.15385 , -0.28205 , -0.53846 ,
                                  -0.07692 ,  0.12308 ,  0.02564 ,  0.03077 ,
                                   0.35897 ,  0.35897 ,  0.43590 ,  0.92308 ,
                                  -0.69231 , -0.69231 , -0.76923 , -1.92308};

        bool success_flag = false;

        bool res = (mat3.inverse(success_flag) == expected);

        REQUIRE(res == true);

        REQUIRE(success_flag == true);

        res = (mat1.inverse(success_flag) == expected);

        REQUIRE(success_flag == false);
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
        pathtracer::mat4 expected{1.f, 0.f, 0.f, 0.f,
                                  0.f, 1.f, 0.f, 0.f,
                                  0.f, 0.f, 1.f, 0.f,
                                  0.f, 0.f, 0.f, 1.f};

        bool res = (pathtracer::mat4::get_identity() == expected);

        REQUIRE(res == true);
    }

    SECTION("Assignment") {
        pathtracer::mat4 expected{0.25f,1.25f,2.25f,3.25f,
                                  4.25f,5.25f,6.25f,7.25f,
                                  8.25f,9.25f,10.25f,11.25f,
                                  12.25f,13.25f,14.25f,15.25f};

        bool res = ((mat1 = mat2) == expected);

        REQUIRE(res == true);
    }
}
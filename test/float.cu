#include <catch2/catch.hpp>
#include "util.cuh"
#include "check_cuda_errors.h"

__global__ void float_equality_test(float a, float b, bool* res) {
    *res = pathtracer::f_equal(a, b);
}

TEST_CASE("Float equality", "[util]") {
    float a = 0.359;

    SECTION("Equal floats") {
        float b = 0.359;
        bool* res;

        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&res), sizeof(bool)) );

        float_equality_test<<<1,1>>>(a, b, res);

        checkCudaErrors( cudaDeviceSynchronize() );

        REQUIRE(*res == true);

        cudaFree(res);
    }

    SECTION("Unequal floats") {
        float b = 0.360;
        bool* res;

        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&res), sizeof(bool)) );

        float_equality_test<<<1,1>>>(a, b, res);

        checkCudaErrors( cudaDeviceSynchronize() );

        REQUIRE(*res == false);

        cudaFree(res);
    }
}
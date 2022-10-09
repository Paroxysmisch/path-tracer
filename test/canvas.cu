#include <catch2/catch.hpp>
#include "check_cuda_errors.h"
#include "util.cuh"
#include "constants.h"

TEST_CASE("Canvas", "[util]") {
    pathtracer::canvas<pathtracer::height, pathtracer::width> c{};

    SECTION("Byte conversion") {
        float n = 0.25;

        bool res = (pathtracer::to_byte(n) == 63);

        REQUIRE(res == true);
    }

    SECTION("Setting pixel values and PPM output") {
        pathtracer::vec3 color{0.75f, 0.25f, 0.f};
        pathtracer::vec3 expected{0.75f, 0.25f, 0.f};

        for (size_t i{0}; i < pathtracer::height; i += 2) {
            for (size_t j{0}; j < pathtracer::width; ++j) {
                c.write_pixel(i, j, color);
            }
        }

        bool res = (c.m_data[pathtracer::height / 2 - 1][pathtracer::width / 2] == expected);

        c.export_as_PPM("PPM_Test.ppm");
    }
}
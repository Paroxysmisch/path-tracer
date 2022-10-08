#include <catch2/catch.hpp>
#include "util.cuh"
#include "constants.h"

TEST_CASE("Canvas", "[util]") {
    

    SECTION("Byte conversion") {
        float n = 0.25;

        bool res = (pathtracer::to_byte(n) == 63);

        REQUIRE(res == true);
    }
}
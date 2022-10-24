#include <catch2/catch.hpp>
#include "phong.cuh"
#include "shapes.cuh"
#include "util.cuh"

TEST_CASE("Phong lighting", "[phong]") {
    pathtracer::object* object;
    cudaMallocManaged(reinterpret_cast<void**>(&object), sizeof(pathtracer::object));

    object->shape_t = pathtracer::SPHERE;
    object->shape_d = {pathtracer::sphere(pathtracer::mat4::get_identity())};
    object->mat_t = pathtracer::PHONG;
    object->mat_d = {pathtracer::phong({1.f, 1.f, 1.f}, 0.1, 0.9f, 0.9f, 200.f)};

    pathtracer::object* light;
    cudaMallocManaged(reinterpret_cast<void**>(&light), sizeof(pathtracer::object));

    light->shape_t = pathtracer::SPHERE;
    light->shape_d = {pathtracer::sphere(pathtracer::mat4::get_translation(0.f, 10.f, -10.f))};
    light->mat_t = pathtracer::LIGHT;
    light->mat_d = {pathtracer::light({1.f, 1.f, 1.f})};

    pathtracer::point point{0.f, 0.f, 0.f};

    pathtracer::vector eye{0.f, -0.7071067811f, -0.7071067811f};
    pathtracer::vector normal{0.f, 0.f, -1.f};

    // auto temp = pathtracer::phong_lighting(object, light, &point, &eye, &normal);

    // std::cout << temp.x << " " << temp.y << " " << temp.z << std::endl;

    pathtracer::vec3 expected1{1.63639f, 1.63639f, 1.63639f};

    REQUIRE((pathtracer::phong_lighting(object, light, &point, &eye, &normal) == expected1) == true);

    eye = {0.f, 0.f, -1.f};
    normal = {0.f, 0.f, -1.f};

    light->shape_d = {pathtracer::sphere(pathtracer::mat4::get_translation(0.f, 0.f, -10.f))};

    pathtracer::vec3 expected2{1.9f, 1.9f, 1.9f};

    REQUIRE((pathtracer::phong_lighting(object, light, &point, &eye, &normal) == expected2) == true);

    eye = {0.f, 0.f, -1.f};
    normal = {0.f, 0.f, -1.f};

    light->shape_d = {pathtracer::sphere(pathtracer::mat4::get_translation(0.f, 0.f, 10.f))};

    pathtracer::vec3 expected3{0.1f, 0.1f, 0.1f};

    REQUIRE((pathtracer::phong_lighting(object, light, &point, &eye, &normal) == expected3) == true);
}
#include <catch2/catch.hpp>
#include "shapes.cuh"
#include "world.cuh"

TEST_CASE("Free World") {
        dim3 blocks(16, 16);
        dim3 threads(16, 16);

        pathtracer::object obj0{pathtracer::SPHERE,
             pathtracer::sphere(pathtracer::mat4::get_translation(-0.5f, 0.f, -0.8f) * pathtracer::mat4::get_scaling(0.25f, 0.25f, 0.25f)),
             pathtracer::LIGHT,
             pathtracer::phong({0.95f, 0.25f, 0.5f}, 0.3, 0.7, 0.5, 10)};
        obj0.mat_d.light = pathtracer::light({0.95f, 0.4f, 0.25});

        pathtracer::object obj1{pathtracer::SPHERE,
             pathtracer::sphere(pathtracer::mat4::get_translation(-0.5f, 0.f, 0.f) * pathtracer::mat4::get_scaling(0.25f, 0.25f, 0.25f)),
             pathtracer::MICROFACET,
             pathtracer::phong{{0.f, 0.f, 0.f}, 0.f, 0.f, 0.f, 0.f}};
        obj1.mat_d.microfacet = pathtracer::microfacet{{0.35f, 0.25f, 0.85f}, {0.f, 0.f, 0.f}, 1.f, 0.2f, 0.f, 1.f, 0.04f, 0.f, 0.f, 0.25f, 0.f, 1.f};

        pathtracer::object obj2{pathtracer::SPHERE,
             pathtracer::sphere(pathtracer::mat4::get_translation(0.5f, 0.f, 0.f) * pathtracer::mat4::get_scaling(0.25f, 0.25f, 0.25f)),
             pathtracer::MICROFACET,
             pathtracer::phong{{0.f, 0.f, 0.f}, 0.f, 0.f, 0.f, 0.f}};
        obj2.mat_d.microfacet = pathtracer::microfacet{{0.35f, 0.85f, 0.45f}, {0.f, 0.f, 0.f}, 0.f, 0.8f, 0.95f, 1.1f, 0.02f, 0.6f, 0.f, 0.f, 0.5f, 1.f};

        pathtracer::object obj3{pathtracer::SPHERE,
             pathtracer::sphere(pathtracer::mat4::get_translation(0.5f, 0.f, 0.f) * pathtracer::mat4::get_scaling(0.15f, 0.15f, 0.15f)),
             pathtracer::MICROFACET,
             pathtracer::phong{{0.f, 0.f, 0.f}, 0.f, 0.f, 0.f, 0.f}};
        obj3.mat_d.microfacet = pathtracer::microfacet{{0.35f, 0.85f, 0.45f}, {0.f, 0.f, 0.f}, 0.f, 0.8f, 0.95f, 1.3f, 0.02f, 0.6f, 0.f, 0.f, 0.f, 1.f};

        int alloc_dealloc_count{1};

        for (int i{0}; i < alloc_dealloc_count; ++i) {
            pathtracer::world w({
                &obj0, &obj1, &obj2, &obj3
            }, {"teapot_full.obj", "teapot_full.obj"}, {pathtracer::mat4::get_scaling(0.01f, 0.01f, 0.01f)}, {{"cursed.exr", 618, 1100}, {"", 0, 0}}, "env.exr", blocks, threads);
            
            w.free_world();
        }
}

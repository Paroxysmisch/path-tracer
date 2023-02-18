#include <iostream>
#include "camera.cuh"
#include "check_cuda_errors.h"
#include "shapes.cuh"
#include "util.cuh"
#include "constants.h"
#include "world.cuh"
#include "render.cuh"

int main() {
    std::cout << "Hello World" << std::endl;

    pathtracer::camera camera(1000, 1000, pathtracer::pi / 2.f, {0.f, 0.f, -0.85f}, {0.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, pathtracer::mat4::get_identity());


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


        dim3 blocks(16, 16);
        dim3 threads(16, 16);

        pathtracer::world world({
            &obj0, &obj1, &obj2, &obj3
        }, {"teapot_full.obj"}, {pathtracer::mat4::get_scaling(0.01f, 0.01f, 0.01f)}, {{"cursed.exr", 618, 1100}}, "env.exr", blocks, threads);

        std::string filename1 = "main_render_no_AS";
        std::string filename2 = "main_render_AS";

        pathtracer::render(camera, world, filename1, 100, false, 0.05f);
        pathtracer::render(camera, world, filename2, 1000, true, 0.05f);

        world.free_world();

}

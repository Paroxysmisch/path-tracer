#include <iostream>
#include "camera.cuh"
#include "check_cuda_errors.h"
#include "shapes.cuh"
#include "util.cuh"
#include "constants.h"
#include "world.cuh"
#include "render.cuh"
#include <chrono>
#include <ratio>

int main() {
    std::cout << "Hello World" << std::endl;

//     pathtracer::camera camera(1024, 1024, pathtracer::pi / 2.f, {0.f, 0.f, -0.85f}, {0.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, pathtracer::mat4::get_identity());

//      pathtracer::object obj0{pathtracer::SPHERE,
//           pathtracer::sphere(pathtracer::mat4::get_translation(-0.5f, 0.f, -0.8f) * pathtracer::mat4::get_scaling(0.25f, 0.25f, 0.25f)),
//           pathtracer::LIGHT,
//           pathtracer::phong({0.95f, 0.25f, 0.5f}, 0.3, 0.7, 0.5, 10)};
//      obj0.mat_d.light = pathtracer::light({0.95f, 0.4f, 0.25});

//      pathtracer::object obj1{pathtracer::SPHERE,
//           pathtracer::sphere(pathtracer::mat4::get_translation(-0.5f, 0.f, 0.f) * pathtracer::mat4::get_scaling(0.25f, 0.25f, 0.25f)),
//           pathtracer::MICROFACET,
//           pathtracer::phong{{0.f, 0.f, 0.f}, 0.f, 0.f, 0.f, 0.f}};
//      obj1.mat_d.microfacet = pathtracer::microfacet{{0.35f, 0.25f, 0.85f}, {0.f, 0.f, 0.f}, 1.f, 0.18f, 0.f, 1.f, 0.04f, 0.f, 0.f, 0.25f, 0.f, 1.f};

//      pathtracer::object obj2{pathtracer::SPHERE,
//           pathtracer::sphere(pathtracer::mat4::get_translation(0.5f, 0.f, 0.f) * pathtracer::mat4::get_scaling(0.25f, 0.25f, 0.25f)),
//           pathtracer::MICROFACET,
//           pathtracer::phong{{0.f, 0.f, 0.f}, 0.f, 0.f, 0.f, 0.f}};
//      obj2.mat_d.microfacet = pathtracer::microfacet{{0.35f, 0.85f, 0.45f}, {0.f, 0.f, 0.f}, 0.f, 0.8f, 0.95f, 1.1f, 0.02f, 0.6f, 0.f, 0.f, 0.5f, 1.f};

//      pathtracer::object obj3{pathtracer::SPHERE,
//           pathtracer::sphere(pathtracer::mat4::get_translation(0.5f, 0.f, 0.f) * pathtracer::mat4::get_scaling(0.15f, 0.15f, 0.15f)),
//           pathtracer::MICROFACET,
//           pathtracer::phong{{0.f, 0.f, 0.f}, 0.f, 0.f, 0.f, 0.f}};
//      obj3.mat_d.microfacet = pathtracer::microfacet{{0.35f, 0.85f, 0.45f}, {0.f, 0.f, 0.f}, 0.f, 0.8f, 0.95f, 1.3f, 0.02f, 0.6f, 0.f, 0.f, 0.f, 1.f};


//      dim3 blocks(16, 16);
//      dim3 threads(16, 16);

//      pathtracer::world world({
//           &obj0, &obj1, &obj2, &obj3
//      }, {"teapot_full.obj"}, {pathtracer::mat4::get_scaling(0.01f, 0.01f, 0.01f)}, {{"cursed.exr", 618, 1100}}, "env.exr", blocks, threads);

//      std::string filename1 = "0_ground_high_res";
//      // std::string filename2 = "main_render_AS";

//      auto start = std::chrono::high_resolution_clock::now();

//      pathtracer::render(camera, world, filename1, 1024, false, 0.05f);
//      // pathtracer::render(camera, world, filename2, 1000, true, 0.05f);

//      auto stop = std::chrono::high_resolution_clock::now();
//      std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
//      std::cout << duration.count() << std::endl;

//      world.free_world();

     //////////////////////////////

     // pathtracer::object obj0{pathtracer::SPHERE,
     //      pathtracer::sphere(pathtracer::mat4::get_translation(0.5f, 0.f, -0.8f) * pathtracer::mat4::get_scaling(0.125f, 0.125f, 0.125f)),
     //      pathtracer::LIGHT,
     //      pathtracer::phong({0.95f, 0.25f, 0.5f}, 0.3, 0.7, 0.5, 10)};
     // obj0.mat_d.light = pathtracer::light({0.95f, 0.85f, 0.85f});

     // pathtracer::object obj1{pathtracer::SPHERE,
     //      pathtracer::sphere(pathtracer::mat4::get_translation(-0.76f, 0.f, 0.f) * pathtracer::mat4::get_scaling(0.15f, 0.15f, 0.15f)),
     //      pathtracer::MICROFACET,
     //      pathtracer::phong{{0.f, 0.f, 0.f}, 0.f, 0.f, 0.f, 0.f}};
     // obj1.mat_d.microfacet = pathtracer::microfacet{{0.90f, 0.25f, 0.05f}, {0.f, 0.f, 0.f}, 0.80f, 0.5f, 0.f, 1.f, 0.04f, 0.f, 0.f, 0.25f, 0.f, 1.f};

     // pathtracer::object obj2{pathtracer::SPHERE,
     //      pathtracer::sphere(pathtracer::mat4::get_translation(-0.35f, 0.f, 0.f) * pathtracer::mat4::get_scaling(0.15f, 0.15f, 0.15f)),
     //      pathtracer::MICROFACET,
     //      pathtracer::phong{{0.f, 0.f, 0.f}, 0.f, 0.f, 0.f, 0.f}};
     // obj2.mat_d.microfacet = pathtracer::microfacet{{0.90f, 0.25f, 0.05f}, {0.f, 0.f, 0.f}, 0.85f, 0.25f, 0.f, 1.f, 0.04f, 0.f, 0.f, 0.25f, 0.f, 1.f};

     // pathtracer::object obj3{pathtracer::SPHERE,
     //      pathtracer::sphere(pathtracer::mat4::get_translation(0.f, 0.f, 0.f) * pathtracer::mat4::get_scaling(0.15f, 0.15f, 0.15f)),
     //      pathtracer::MICROFACET,
     //      pathtracer::phong{{0.f, 0.f, 0.f}, 0.f, 0.f, 0.f, 0.f}};
     // obj3.mat_d.microfacet = pathtracer::microfacet{{0.90f, 0.25f, 0.05f}, {0.f, 0.f, 0.f}, 0.90f, 0.18f, 0.f, 1.f, 0.04f, 0.f, 0.f, 0.25f, 0.f, 1.f};

     // pathtracer::object obj4{pathtracer::SPHERE,
     //      pathtracer::sphere(pathtracer::mat4::get_translation(0.35f, 0.f, 0.f) * pathtracer::mat4::get_scaling(0.15f, 0.15f, 0.15f)),
     //      pathtracer::MICROFACET,
     //      pathtracer::phong{{0.f, 0.f, 0.f}, 0.f, 0.f, 0.f, 0.f}};
     // obj4.mat_d.microfacet = pathtracer::microfacet{{0.90f, 0.25f, 0.05f}, {0.f, 0.f, 0.f}, 0.95f, 0.14f, 0.f, 1.f, 0.04f, 0.f, 0.f, 0.25f, 0.f, 1.f};

     // pathtracer::object obj5{pathtracer::SPHERE,
     //      pathtracer::sphere(pathtracer::mat4::get_translation(0.76f, 0.f, 0.f) * pathtracer::mat4::get_scaling(0.15f, 0.15f, 0.15f)),
     //      pathtracer::MICROFACET,
     //      pathtracer::phong{{0.f, 0.f, 0.f}, 0.f, 0.f, 0.f, 0.f}};
     // obj5.mat_d.microfacet = pathtracer::microfacet{{0.90f, 0.25f, 0.05f}, {0.f, 0.f, 0.f}, 0.99f, 0.1f, 0.f, 1.f, 0.04f, 0.f, 0.f, 0.25f, 0.f, 1.f};


     // dim3 blocks(32, 32);
     // dim3 threads(16, 16);

     // pathtracer::world world({
     //      &obj0, &obj1, &obj2, &obj3, &obj4, &obj5
     // }, {}, {}, {}, "mountains.exr", blocks, threads);

     // std::string filename1 = "1_ground_16384";
     // // std::string filename2 = "1_1024_on_4_AS";

     // auto start = std::chrono::high_resolution_clock::now();

     // pathtracer::render(camera, world, filename1, 16384, false, 0.05f);
     // // pathtracer::render(camera, world, filename2, 1024, true, 0.05f);

     // auto stop = std::chrono::high_resolution_clock::now();
     // std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
     // std::cout << duration.count() << std::endl;

     // world.free_world();

     //////////////////////////////

     pathtracer::camera camera(1024, 1024, pathtracer::pi / 2.f, {0.f, 0.f, -0.3f}, {0.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, pathtracer::mat4::get_identity());


     pathtracer::object obj0{pathtracer::SPHERE,
          pathtracer::sphere(pathtracer::mat4::get_translation(0.5f, 0.f, -0.5f) * pathtracer::mat4::get_scaling(0.35f, 0.35f, 0.35f)),
          pathtracer::LIGHT,
          pathtracer::phong({0.95f, 0.25f, 0.5f}, 0.3, 0.7, 0.5, 10)};
     obj0.mat_d.light = pathtracer::light({0.95f, 0.85f, 0.85f});

     dim3 blocks(32, 32);
     dim3 threads(16, 16);

     // pathtracer::world world({
     //      &obj0,
     // }, {}, {}, {}, "mountains.exr", blocks, threads);

     pathtracer::world world({
          &obj0
     }, {"ICESat2.obj"}, {pathtracer::mat4::get_translation(0.f, 0.f, 0.f) * pathtracer::mat4::get_rotation_y(pathtracer::pi / 2.5f) * pathtracer::mat4::get_rotation_z(pathtracer::pi * 0.75f) * pathtracer::mat4::get_rotation_x(pathtracer::pi / 2.f) * pathtracer::mat4::get_scaling(0.1f, 0.1f, 0.1f)}, {{"", 618, 1100}}, "starmap_g8k.exr", blocks, threads);


     std::string filename1 = "double_test";
     // std::string filename2 = "2_1024_on_16";

     auto start = std::chrono::high_resolution_clock::now();

     pathtracer::render(camera, world, filename1, 128, false, 0.05f);
     // pathtracer::render(camera, world, filename2, 1024, true, 0.05f);

     auto stop = std::chrono::high_resolution_clock::now();
     std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
     std::cout << duration.count() << std::endl;

     world.free_world();

}

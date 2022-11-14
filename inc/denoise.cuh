#pragma once

#include "world.cuh"
#include "camera.cuh"
#include <stdlib.h>
#include <unistd.h>

namespace pathtracer {

    bool get_albedo_exr(int height, int width, const world& world, const camera& camera, const std::string &filename);

    bool get_normal_map_exr(int height, int width, const world& world, const camera& camera, const std::string &filename);

    pid_t call_optixDenoiser(char* argv[]);

    void denoise(int height, int width,  const std::string &in_filename, const world& world, const camera& camera, const std::string &out_filename);

}

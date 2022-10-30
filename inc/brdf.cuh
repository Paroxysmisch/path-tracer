#pragma once

#include "util.cuh"

namespace pathtracer {

    __host__ __device__ point cosine_sample_hemisphere(float u, float v, float& out_pdf);

    bool eval_brdf(float u, float v, vector normal, vector view, vector& out_ray_direction, vec3& out_sample_weight);

}
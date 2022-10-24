#pragma once

#include <cmath>
#include "shapes.cuh"

namespace pathtracer {

    __host__ __device__ vec3 phong_lighting(const object* thing,
                        const object* light,
                        const point* world_point,
                        const vec3* eye_vector,
                        const vec3* normal_vector);

}
#include <cmath>
#include "brdf.cuh"
#include "constants.h"
#include "util.cuh"

namespace pathtracer {

    // Samples points oriented along +Z axis
    // Needs to be transformed by a quaternion
    __host__ __device__ point cosine_sample_hemisphere(float u, float v, float& out_pdf) {
        float alpha = sqrtf(u);
        float beta = two_pi * v;

        point result {
            alpha * cosf(beta),
            alpha * sinf(beta),
            sqrtf(1.f - u)
        };

        out_pdf = one_over_pi * result.z;

        return result;
    }

}
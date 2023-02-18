#include "brdf.cuh"
#include "camera.cuh"
#include "util.cuh"
#include "world.cuh"
#include <curand_kernel.h>

namespace pathtracer {

    __global__ void render_kernel(pathtracer::canvas c,
                                  pathtracer::world world,
                                  pathtracer::camera camera,
                                  curandState* d_states,
                                  int num_samples = 100,
                                  bool enable_adaptive_sampling = false,
                                  const float adaptive_sampling_variance_threshold = 0.05f);

    __host__ void render(camera& camera,
                world& world,
                std::string filename,
                int num_samples = 100,
                bool enable_adaptive_sampling = false,
                const float adaptive_sampling_variance_threshold = 0.05f);

}

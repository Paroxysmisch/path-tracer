#include <cmath>
#include <cstdlib>
#include "brdf.cuh"
#include "constants.h"
#include "util.cuh"

namespace pathtracer {

    __host__ __device__ vec3 linear_interpolate(const vec3& begin, const vec3& end, float amount) {
        return begin + (end - begin) * amount;
    }

    __host__ __device__ vec3 base_color_to_specular_f0(const vec3& color, float metalness) {
        return linear_interpolate(vec3(min_dielectrics_f0, min_dielectrics_f0, min_dielectrics_f0), color, metalness);
    }

    __host__ __device__ vec3 base_color_to_diffuse_reflectance(const vec3& color, float metalness) {
        return color * (1.f - metalness);
    }

    __host__ __device__ vec3 eval_fresnel(const vec3& f0, float f90, float n_dot_s) {
        // Using Schlick's approximation to the Fresnel term
        return f0 + (-f0 + vec3(f90, f90, f90)) * powf(1.f - n_dot_s, 5.f);
    }

    __device__ float shadowed_f90(vec3 f0) {
        // For attenuation of f90 for very low f0
        const float c = (1.f / min_dielectrics_f0);
        return min(1.f, c * luminance(f0));
    }

    __host__ __device__ float luminance(const vec3& rgb) {
        return rgb * vec3(0.2126f, 0.7152f, 0.0722f);
    }

    __device__ brdf_data gen_brdf_data(const vec3& view, 
                                                const vec3& normal, 
                                                const vec3& light, 
                                                const microfacet& material) {
        brdf_data res;

        res.view = view;
        res.normal = normal;
        res.half = (view + light).normalize();
        res.light = light;

        res.n_dot_l = min(max(0.000001f, normal * light), 1.f);
        res.n_dot_v = min(max(0.000001f, normal * view), 1.f);

        res.l_dot_h = min(max(0.f, light * res.half), 1.f);
        res.n_dot_h = min(max(0.f, normal * res.half), 1.f);
        res.v_dot_h = min(max(0.f, view * res.half), 1.f);

        res.specularF0 = base_color_to_specular_f0(material.color, material.metalness);
        res.diffuseReflectance = base_color_to_diffuse_reflectance(material.color, material.metalness);

        res.roughness = material.roughness;
        res.alpha = material.roughness * material.roughness;
        res.alpha_squared = res.alpha * res.alpha;

        res.fresnel = eval_fresnel(res.specularF0, shadowed_f90(res.specularF0), res.l_dot_h);

        return res;
    }

    __host__ __device__ vec3 sample_specular(const vec3& view_local, 
                                             float alpha, 
                                             float alpha_squared, 
                                             const vec3& specularF0, 
                                             float u, 
                                             float v, 
                                             vec3& out_weight) {
        vec3 half_local;
        if (f_equal(alpha, 0.f)) {
            half_local = {0.f, 0.f, 0.f};
        } // Need to implement
        return {0.f, 0.f, 0.f};
    }

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

    __device__ bool eval_brdf(float u, 
                              float v,
                              float t,
                              float in_refractive_index, 
                              vector normal, 
                              vector view, 
                              vector& out_ray_direction, 
                              vec3& out_sample_weight,
                              float& out_refractive_index,
                              const microfacet& material) {
        if (0.f < t && t <= material.transmissiveness) {
            // We sample the hemisphere
            // around the perfectly refracted ray
            vector incident = -view.normalize();
            float n = in_refractive_index / material.refractive_index;
            const float cos_i = -(normal * incident);
            const float sin_t2 = n * n * (1.f - cos_i * cos_i);
            // if (sin_t2 > 1.f) return false; // We have Total Internal Reflection
            const float cos_t = sqrtf(1.f - sin_t2);
            vector refracted = (incident * n) + (normal * (n * cos_i - cos_t));
            float c = (-normal) * incident;
            refracted = (incident * n) + normal * (n * c - sqrtf(1 - powf(n, 2.f) * (1 - powf(c, 2.f))));

            // const quaternion q_normal_rotation_to_z = quaternion::get_rotation_to_z_axis(refracted.normalize());
            // float pdf;
            // pathtracer::point ray_direction_local = pathtracer::cosine_sample_hemisphere(u, v, pdf);
            // out_ray_direction = quaternion::rotate_vector_by_quaternion(ray_direction_local, quaternion::get_inverse_rotation(q_normal_rotation_to_z)).normalize();
            // out_sample_weight = vec3(1.f, 1.f, 1.f) * (fabsf(view * out_ray_direction) / (pdf * material.transmissiveness)) * one_over_pi;
            out_sample_weight = {1.f, 1.f, 1.f};
            out_ray_direction = refracted.normalize();
            if (normal * view <= 0.f) {
                out_refractive_index = material.refractive_index;
            } else {
                out_refractive_index = 1.f; // Assume the ray leaves into a vacuum
            }
            return true;
        }

        if (normal * view <= 0.f) return false;

        const quaternion q_normal_rotation_to_z = quaternion::get_rotation_to_z_axis(normal);
        const vector view_local = quaternion::rotate_vector_by_quaternion(view, q_normal_rotation_to_z);
        vector normal_local{0.f, 0.f, 1.f};

        float pdf;
        pathtracer::point ray_direction_local = pathtracer::cosine_sample_hemisphere(u, v, pdf);

        // Diffuse BRDF
        const brdf_data data = gen_brdf_data(view_local, normal_local, ray_direction_local, material);

        out_sample_weight = data.diffuseReflectance * (data.n_dot_l / (pdf * (1 - material.transmissiveness))) * one_over_pi;

        if (f_equal(luminance(out_sample_weight), 0.f)) return false;

        out_ray_direction = quaternion::rotate_vector_by_quaternion(ray_direction_local, quaternion::get_inverse_rotation(q_normal_rotation_to_z)).normalize();

        out_refractive_index = in_refractive_index;

        return true;
    }

}
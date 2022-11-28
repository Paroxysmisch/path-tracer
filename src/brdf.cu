#include <cmath>
#include <cstdlib>
#include <math.h>
#include "brdf.cuh"
#include "constants.h"
#include "util.cuh"

namespace pathtracer {

    __host__ __device__ vec3 linear_interpolate(const vec3& begin, const vec3& end, float amount) {
        return begin + (end - begin) * amount;
    }

    __host__ __device__ float linear_interpolate(float begin, float end, float amount) {
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

    __host__ __device__ float shadowed_f90(vec3 f0) {
        // For attenuation of f90 for very low f0
        const float c = (1.f / min_dielectrics_f0);
        return minf(1.f, c * luminance(f0));
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

    __host__ __device__ vec3 sample_GGX_VNDF(vec3 vec, float alpha_1, float alpha_2, float u, float v) {
        vec3 v_h = vec3(alpha_1 * vec.x, alpha_2 * vec.y, vec.z);

        float lensq = v_h.x * v_h.x + v_h.y * v_h.y;
        vec3 T1 = lensq > 0.f ? vec3(-v_h.y, v_h.x, 0.f) * (1 / sqrtf(lensq)) : vec3(1.f, 0.f, 0.f);
        vec3 T2 = v_h ^ T1;

        float r = sqrtf(u);
        float phi = two_pi * v;
        float t1 = r * cosf(phi);
        float t2 = r * sinf(phi);
        float s = 0.5f * (1.f + v_h.z);
        t2 = linear_interpolate(sqrtf(1.f - t1 * t1), t2, s);

        vec3 n_h = (T1 * t1) + (T2 * t2) + (v_h * sqrtf(maxf(0.f, 1.f - t1 * t1 - t2 * t2)));

        return vec3(n_h.x * alpha_1, n_h.y * alpha_2, maxf(0.f, n_h.z));
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
        } else {
            half_local = sample_GGX_VNDF(view_local, alpha, alpha, u, v);
        }

        vec3 l_local = (-view_local).reflect(half_local);

        float h_dot_l = maxf(0.00001f, minf(1.f, half_local * l_local));
        const vec3 n_local = vec3(0.f, 0.f, 1.f);
        float n_dot_l = maxf(0.00001f, minf(1.f, n_local * l_local));
        float n_dot_v = maxf(0.00001f, minf(1.f, n_local * view_local));
        float n_dot_h = maxf(0.00001f, minf(1.f, n_local * half_local));
        vec3 F = eval_fresnel(specularF0, shadowed_f90(specularF0), h_dot_l);

        // out_weight = F * spe

        // Need to implement
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
            const float cos_t = sqrtf(1.f - sin_t2);
            vector refracted = (incident * n) + (normal * (n * cos_i - cos_t));
            if (sin_t2 > 1.f) {
                // We have Total Internal Reflection
                if (normal * view > 0.f) {
                    refracted = incident.reflect(normal);
                } else {
                    refracted = incident.reflect(-normal);
                }
            } 
            // float c = (-normal) * incident;
            // refracted = (incident * n) + normal * (n * c - sqrtf(1 - powf(n, 2.f) * (1 - powf(c, 2.f))));

            // const quaternion q_normal_rotation_to_z = quaternion::get_rotation_to_z_axis(-normal);
            // float pdf;
            // pathtracer::point ray_direction_local = pathtracer::cosine_sample_hemisphere(u, v, pdf);
            // out_ray_direction = quaternion::rotate_vector_by_quaternion(ray_direction_local, quaternion::get_inverse_rotation(q_normal_rotation_to_z)).normalize();
            // out_sample_weight = vec3(1.f, 1.f, 1.f) * ((view * out_ray_direction) / (pdf)) * one_over_pi;
            // if (out_ray_direction * (-normal) >= 0.5f) return false;
            out_sample_weight = vec3(1.f, 1.f, 1.f) * 0.9f; // Replace with object's density
            out_ray_direction = refracted.normalize();
            if (normal * view <= 0.f) {
                out_refractive_index = material.refractive_index;
            } else {
                out_refractive_index = 1.f; // Assume the ray leaves into a vacuum
            }
            return true;
        } else {
            if (normal * view <= 0.f) return false;

            const quaternion q_normal_rotation_to_z = quaternion::get_rotation_to_z_axis(normal);
            const vector view_local = quaternion::rotate_vector_by_quaternion(view, q_normal_rotation_to_z);
            vector normal_local{0.f, 0.f, 1.f};

            float pdf;
            pathtracer::point ray_direction_local = pathtracer::cosine_sample_hemisphere(u, v, pdf);

            // Diffuse BRDF
            const brdf_data data = gen_brdf_data(view_local, normal_local, ray_direction_local, material);

            // Old Model
            // out_sample_weight = data.diffuseReflectance * (data.n_dot_l / (pdf)) * one_over_pi;

            vec3 f0 = base_color_to_specular_f0(material.color, material.metalness);
            vec3 F = eval_fresnel(f0, shadowed_f90(f0), data.v_dot_h);

            float r1 = 1.f / (4.f * data.alpha_squared * powf(data.n_dot_h, 4.f));
            float r2 = (data.n_dot_h * data.n_dot_h - 1.f) / (data.alpha_squared * data.n_dot_h * data.n_dot_h);
            float D = r1 * expf(r2);

            float two_n_dot_h = 2.f * data.n_dot_h;
            float g1 = (two_n_dot_h * data.n_dot_v) / data.v_dot_h;
            float g2 = (two_n_dot_h * data.n_dot_l) / data.v_dot_h;
            float G = minf(1.f, minf(g1, g2));

            float Rs = (D * G) * one_over_pi / (data.n_dot_l * data.n_dot_v);
            vec3 Rs_F = F * Rs;
            Rs_F *= 0.8f;
            Rs_F += vec3(0.2f, 0.2f, 0.2f);

            out_sample_weight = (data.diffuseReflectance * one_over_pi + (material.color & Rs_F)) * (data.n_dot_l / (pdf));

            if (f_equal(luminance(out_sample_weight), 0.f)) return false;

            out_ray_direction = quaternion::rotate_vector_by_quaternion(ray_direction_local, quaternion::get_inverse_rotation(q_normal_rotation_to_z)).normalize();

            out_refractive_index = in_refractive_index;

            return true;
        }        
    }

}
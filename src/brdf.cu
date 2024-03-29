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

    // __host__ __device__ vec3 sample_GGX_VNDF(vec3 vec, float alpha_1, float alpha_2, float u, float v) {
    //     vec3 v_h = vec3(alpha_1 * vec.x, alpha_2 * vec.y, vec.z);

    //     float lensq = v_h.x * v_h.x + v_h.y * v_h.y;
    //     vec3 T1 = lensq > 0.f ? vec3(-v_h.y, v_h.x, 0.f) * (1 / sqrtf(lensq)) : vec3(1.f, 0.f, 0.f);
    //     vec3 T2 = v_h ^ T1;

    //     float r = sqrtf(u);
    //     float phi = two_pi * v;
    //     float t1 = r * cosf(phi);
    //     float t2 = r * sinf(phi);
    //     float s = 0.5f * (1.f + v_h.z);
    //     t2 = linear_interpolate(sqrtf(1.f - t1 * t1), t2, s);

    //     vec3 n_h = (T1 * t1) + (T2 * t2) + (v_h * sqrtf(maxf(0.f, 1.f - t1 * t1 - t2 * t2)));

    //     return vec3(n_h.x * alpha_1, n_h.y * alpha_2, maxf(0.f, n_h.z));
    // }

    // __host__ __device__ vec3 sample_specular(const vec3& view_local,
    //                                          float alpha,
    //                                          float alpha_squared,
    //                                          const vec3& specularF0,
    //                                          float u,
    //                                          float v,
    //                                          vec3& out_weight) {
    //     vec3 half_local;
    //     if (f_equal(alpha, 0.f)) {
    //         half_local = {0.f, 0.f, 0.f};
    //     } else {
    //         half_local = sample_GGX_VNDF(view_local, alpha, alpha, u, v);
    //     }

    //     vec3 l_local = (-view_local).reflect(half_local);

    //     float h_dot_l = maxf(0.00001f, minf(1.f, half_local * l_local));
    //     const vec3 n_local = vec3(0.f, 0.f, 1.f);
    //     float n_dot_l = maxf(0.00001f, minf(1.f, n_local * l_local));
    //     float n_dot_v = maxf(0.00001f, minf(1.f, n_local * view_local));
    //     float n_dot_h = maxf(0.00001f, minf(1.f, n_local * half_local));
    //     vec3 F = eval_fresnel(specularF0, shadowed_f90(specularF0), h_dot_l);

    //     // out_weight = F * spe

    //     // Need to implement
    //     return {0.f, 0.f, 0.f};
    // }

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

    __host__ __device__ point ggx_sample_hemisphere(float u, float v, float roughness, const vec3 view_local, const vec3 normal_local, float& out_pdf) {
        float phi = 2.f * pi * u;
        float theta = acos(sqrt((1.0f - v)/
                                    ((roughness * roughness - 1.0f) * v + 1.0f)
                                  )
                          );

        point result {
            sinf(theta) * cosf(phi),
            sinf(theta) * sinf(phi),
            cosf(theta)
        };
        result = result.normalize();

        vec3 h = (view_local + result).normalize();
        float hdotn = minf(maxf(0.000001f, h * normal_local), 1.f);
        float vdoth = minf(maxf(0.000001f, view_local * h), 1.f);

        float t = hdotn*hdotn*roughness*roughness - (hdotn*hdotn - 1.0f);
        float D = (roughness*roughness)*(1.f / ((t*t) + epsilon))*one_over_pi;
        // out_pdf = D*hdotn*(1.f / ((4.0f*fabsf(vdoth)) + epsilon));
        out_pdf = D_GGX(hdotn, roughness) * hdotn / (4 * vdoth);

        return result;

    }

    // __host__ __device__ point ggx_vndf_sample_hemisphere(float u, float v, float roughness, const vec3 Ve, float& out_pdf) {
    //     vec3 Vh = vec3(roughness * Ve.x, roughness * Ve.y, Ve.z).normalize();

    //     float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    //     vec3 T1 = lensq > 0.f ? vec3(-Vh.y, Vh.x, 0.f) * (1.f / sqrtf(lensq)) : vec3(0.f, 0.f, 1.f);
    //     vec3 T2 = Vh ^ T1;

    //     float r = sqrtf(u);
    //     float phi = 2.f * pi * v;
    //     float t1 = r * cos(phi);
    //     float t2 = r * sin(phi);
    //     float s = 0.5f * (1.f + Vh.z);
    //     t2 = (1.f - s) * sqrtf(1.f - t1 * t1) + s * t2;

    //     vec3 Nh = T1 * t1 + T2 * t2 + Vh * sqrtf(maxf(0.f, 1.f - t1*t1- t2*t2));
    //     vec3 Ne = vec3(roughness * Nh.x, roughness * Nh.y, maxf(0.f, Nh.z)).normalize();

    //     return Ne;
    // }

    __host__ __device__ point power_cosine_sample_hemisphere(float u, float v, float power, float& out_pdf) {
        float phi = 2.f * pi * u;
        float theta = acosf(powf(v, 1.f / (power + 1.f)));

        point result {
            sinf(theta) * cosf(phi),
            sinf(theta) * sinf(phi),
            cosf(theta)
        };
        result = result.normalize();

        out_pdf = 0.5f * one_over_pi * (power + 1.f) * powf(cosf(theta), power) * sinf(theta);

        return result;
    }

    __host__ __device__ point uniform_sample_hemisphere(float u, float v, float& out_pdf) {
        float phi = 2.f * pi * u;
        float theta = acosf(v);

        point result {
            sinf(theta) * cosf(phi),
            sinf(theta) * sinf(phi),
            cosf(theta)
        };
        result = result.normalize();

        out_pdf = 0.5f * one_over_pi * sinf(theta);

        return result;
    }

    __host__ __device__ float D_GGX(float NoH, float roughness) {
        float a = NoH * roughness;
        float k = roughness / (1.f - NoH * NoH + a * a);
        return k * k * one_over_pi;
    }

    __host__ __device__ vec3 F_Schlick(float u, vec3 f0) {
        return f0 + (vec3(1.f, 1.f, 1.f) - f0) * powf(1.f - u, 5.f);
    }

    __host__ __device__ float V_SmithGGXCorrelated(float NoV, float NoL, float roughness) {
        float a2 = roughness * roughness;
        float GGXL = NoV * sqrt(NoL * NoL * (1.f - a2) + a2);
        float GGXV = NoL * sqrt(NoV * NoV * (1.f - a2) + a2);
        return 0.5f / (GGXV + GGXL);
    }

    __host__ __device__ float Fd_Lambert() {
        return one_over_pi;
    }

    __host__ __device__ float Fd_DisneyDiffuse(brdf_data data) {
        float FD90MinusOne = 2.f * data.roughness * data.l_dot_h * data.l_dot_h - 0.5f;

        float FDL = 1.f + (FD90MinusOne * pow(1.0f - data.n_dot_l, 5.f));
        float FDV = 1.f + (FD90MinusOne * pow(1.0f - data.n_dot_v, 5.f));

        return FDL * FDV * one_over_pi;
    }

    __host__ __device__ float V_Kelemen(float LoH) {
        return 0.25f / (LoH * LoH);
    }

    __host__ __device__ float D_GGX_Anisotropic(float NoH, const vec3 h, const vec3 t, const vec3 b, float at, float ab) {
        float ToH = t * h;
        float BoH = b * h;
        float a2 = at * ab;
        vec3 v = vec3(ab * ToH, at * BoH, a2 * NoH);
        float v2 = v * v;
        float w2 = a2 / v2;
        return a2 * w2 * w2 * one_over_pi;
    }

    __host__ __device__ float V_SmithGGXCorrelated_Anisotropic(float at, float ab, float ToV, float BoV, float ToL, float BoL, float NoV, float NoL) {
        float lambdaV = NoL * vec3(at * ToV, ab * BoV, NoV).mag();
        float lambdaL = NoV * vec3(at * ToL, ab * BoL, NoL).mag();
        float v = 0.5f / (lambdaV + lambdaL);
        return minf(v, mediump_flt_max);
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
            // out_ray_direction = (refracted + out_ray_direction).normalize();
            // out_sample_weight = vec3(1.f, 1.f, 1.f) * ((view * out_ray_direction) / (pdf)) * one_over_pi;
            // if (out_ray_direction * (-normal) >= 0.5f) return false;
            // out_sample_weight = vec3(1.f, 1.f, 1.f) * 0.9f;
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

            // Disabled
            // vec3 f0 = base_color_to_specular_f0(material.color, material.metalness);
            // vec3 F = eval_fresnel(f0, shadowed_f90(f0), data.v_dot_h);

            // float r1 = 1.f / (epsilon + 4.f * data.alpha * powf(data.n_dot_h, 4.f));
            // float r2 = (data.n_dot_h * data.n_dot_h - 1.f) / (epsilon + data.alpha * data.n_dot_h * data.n_dot_h);
            // float D = r1 * expf(r2);

            // float two_n_dot_h = 2.f * data.n_dot_h;
            // float g1 = (two_n_dot_h * data.n_dot_v) / data.v_dot_h;
            // float g2 = (two_n_dot_h * data.n_dot_l) / data.v_dot_h;
            // float G = minf(1.f, minf(g1, g2));

            // float Rs = (D * G) * one_over_pi / (data.n_dot_l * data.n_dot_v);
            // vec3 Rs_F = F * Rs;
            // Rs_F *= 0.8f;
            // Rs_F += vec3(0.2f, 0.2f, 0.2f);

            // out_sample_weight = (data.diffuseReflectance * one_over_pi * (1.f - material.metalness) + (material.color & Rs_F)) * (data.n_dot_l / (pdf));

            // perceptually linear roughness to roughness
            float roughness = material.roughness * material.roughness;
            vec3 f0 = vec3(0.16f * material.reflectance * material.reflectance * (1.f - material.metalness)) + material.color * material.metalness;

            float D = D_GGX(data.n_dot_h, roughness);
            vec3 F = F_Schlick(data.l_dot_h, f0);
            float V = V_SmithGGXCorrelated(data.n_dot_v, data.n_dot_l, roughness);

            // specular BRDF
            vec3 Fr = F * (D * V);

            // diffuse BRDF
            vec3 diffuseColor = material.color * (1.0 - material.metalness);
            vec3 Fd = diffuseColor * Fd_DisneyDiffuse(data);

            // clear coat
            float clear_coat_perceptual_roughness = minf(maxf(material.clear_coat_roughness, 0.089f), 1.f);
            float clear_coat_roughness = clear_coat_perceptual_roughness * clear_coat_perceptual_roughness;

            float  Dc = D_GGX(clear_coat_roughness, data.n_dot_h);
            float Vc = V_SmithGGXCorrelated(data.n_dot_v, data.n_dot_l, clear_coat_roughness);
            float  Fc = F_Schlick(0.04, data.l_dot_h).x * material.clear_coat_strength;
            float Frc = (Dc * Vc) * Fc;

            out_sample_weight = material.emission + ((Fr * (1.f - Fc)) + (Fd * (1.f - Fc)) + Frc) * (data.n_dot_l / (pdf));

            // if (f_equal(luminance(out_sample_weight), 0.f)) return false;

            out_ray_direction = quaternion::rotate_vector_by_quaternion(ray_direction_local, quaternion::get_inverse_rotation(q_normal_rotation_to_z)).normalize();

            out_refractive_index = in_refractive_index;

            return true;
        }        
    }

    __device__ bool eval_brdf_anisotropic(float u,
                              float v,
                              float t,
                              vector normal,
                              vector view,
                              vector& out_ray_direction,
                              vec3& out_sample_weight,
                              const microfacet& material,
                              const vec3 tangent,
                              const vec3 bitangent,
                              const float t_value,
                              const bool from_inside,
                              float* refractive_idx_buffer,
                              int& refractive_idx_buffer_ptr) {
        if (0.f < t && t <= material.transmissiveness) {
            // We sample the hemisphere
            // around the perfectly refracted ray
            // where the perfectly refracted ray is calculated from a perturbed normal

            // Random perturbation to normal
            const quaternion q_normal_rotation_to_z = quaternion::get_rotation_to_z_axis(normal);
            float pdf;
            pathtracer::point random_ray = pathtracer::power_cosine_sample_hemisphere(u, v, 1.f, pdf);
            // const vector view_local = quaternion::rotate_vector_by_quaternion(view, q_normal_rotation_to_z);
            // vector _view_local = -view_local;
            // vector normal_local{0.f, 0.f, 1.f};
            // pathtracer::point random_ray = pathtracer::ggx_sample_hemisphere(u, v, material.roughness * material.roughness, view_local, normal_local, pdf);
            // random_ray =_view_local.reflect(random_ray);

            vector perturbation = quaternion::rotate_vector_by_quaternion(random_ray, quaternion::get_inverse_rotation(q_normal_rotation_to_z)).normalize();

            float roughness = powf(material.transmissive_roughness, 4.f);
            normal = linear_interpolate(normal, perturbation, roughness);

            vector incident = -view.normalize();
            float in_refractive_index = refractive_idx_buffer[refractive_idx_buffer_ptr];
            float n = in_refractive_index / material.refractive_index;
            const float cos_i = -(normal * incident);
            const float sin_t2 = n * n * (1.f - cos_i * cos_i);
            const float cos_t = sqrtf(1.f - sin_t2);
            vector refracted = (incident * n) + (normal * (n * cos_i - cos_t));
            if (sin_t2 > 1.f) {
                // We have Total Internal Reflection
                refracted = incident.reflect(normal);
                // if (normal * view > 0.f) {
                //     refracted = incident.reflect(normal);
                // } else {
                //     refracted = incident.reflect(-normal); // should be able to remove this as this never happens due to comp flipping normals when hitting from the inside of an object
                // }
            }

            // float l_dot_h = min(max(0.f, out_ray_direction * (view + out_ray_direction).normalize()), 1.f);
            out_ray_direction = refracted.normalize();
            out_sample_weight = {1.f};

            if (from_inside) {
                if (refractive_idx_buffer_ptr > 0) {
                    refractive_idx_buffer_ptr -= 1;
                }
                out_sample_weight = {expf(-material.color.x * t_value * material.optical_density), expf(-material.color.y * t_value * material.optical_density), expf(-material.color.z * t_value * material.optical_density)};
            } else {
                // we are entering a new material
                refractive_idx_buffer_ptr += 1;
                refractive_idx_buffer[refractive_idx_buffer_ptr] = material.refractive_index;
            }

            // if (normal * view <= 0.f) {
            //     out_refractive_index = material.refractive_index;
            // } else {
            //     out_refractive_index = 1.f; // Assume the ray leaves into a vacuum
            // }
            return true;
        } else {
            // if (normal * view <= 0.f) return false;

            const quaternion q_normal_rotation_to_z = quaternion::get_rotation_to_z_axis(normal);
            const vector view_local = quaternion::rotate_vector_by_quaternion(view, q_normal_rotation_to_z);
            vector normal_local{0.f, 0.f, 1.f};

            float pdf;
            // pathtracer::point ray_direction_local = pathtracer::cosine_sample_hemisphere(u, v, pdf);
            // pathtracer::point ray_direction_local = pathtracer::uniform_sample_hemisphere(u, v, pdf);
            point _ray_direction_local = pathtracer::ggx_sample_hemisphere(u, v, material.roughness * material.roughness, view_local, normal_local, pdf);
            vector _view_local = -view_local;

            // point ray_direction_local = _view_local - (_ray_direction_local * 2 * (_view_local * _ray_direction_local));
            point ray_direction_local = _view_local.reflect(_ray_direction_local).normalize();

            // BRDF computations
            const brdf_data data = gen_brdf_data(view_local, normal_local, ray_direction_local, material);

            // perceptually linear roughness to roughness
            float roughness = material.roughness * material.roughness;
            float at = maxf(roughness * (1.f + material.anisotropy), 0.001f);
            float ab = maxf(roughness * (1.f - material.anisotropy), 0.001f);
            vec3 f0 = vec3(0.16f * material.reflectance * material.reflectance * (1.f - material.metalness)) + material.color * material.metalness;

            float D = D_GGX_Anisotropic(data.n_dot_h, data.half, tangent, bitangent, at, ab);
            vec3 F = F_Schlick(data.l_dot_h, f0);
            float V = V_SmithGGXCorrelated_Anisotropic(at, ab, tangent * data.view, bitangent * data.view, tangent * data.light, bitangent * data.light, data.n_dot_v, data.n_dot_l);

            // specular BRDF
            vec3 Fr = F * (D * V);

            // diffuse BRDF
            vec3 diffuseColor = material.color * (1.0 - material.metalness);
            vec3 Fd = diffuseColor * Fd_DisneyDiffuse(data);

            // clear coat
            float clear_coat_perceptual_roughness = minf(maxf(material.clear_coat_roughness, 0.089f), 1.f);
            float clear_coat_roughness = clear_coat_perceptual_roughness * clear_coat_perceptual_roughness;

            float  Dc = D_GGX(clear_coat_roughness, data.n_dot_h);
            float Vc = V_SmithGGXCorrelated(data.n_dot_v, data.n_dot_l, clear_coat_roughness);
            float  Fc = F_Schlick(0.04, data.l_dot_h).x * material.clear_coat_strength;
            float Frc = (Dc * Vc) * Fc;

            out_sample_weight = material.emission + ((Fr * (1.f - Fc)) + (Fd * (1.f - Fc)) + Frc) * (data.n_dot_l / (pdf));

            // if (f_equal(luminance(out_sample_weight), 0.f)) return false;

            out_ray_direction = quaternion::rotate_vector_by_quaternion(ray_direction_local, quaternion::get_inverse_rotation(q_normal_rotation_to_z)).normalize();

            return true;
        }
    }

}
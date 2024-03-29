#pragma once

#include "util.cuh"
#include "shapes.cuh"

namespace pathtracer {

    constexpr float min_dielectrics_f0{0.04f};

    struct brdf_data {
        // Calculated material properties
        vec3 specularF0;
        vec3 diffuseReflectance;

        // Roughness properties
        float roughness;
        float alpha;
        float alpha_squared;

        // Fresnel term calculation
        vec3 fresnel;

        // Calculated vectors
        vec3 view;
        vec3 normal;
        vec3 half; // Half vectors acts as the microfacet normal
        vec3 light; // direction vector of the reflected ray

        float n_dot_l;
        float n_dot_v;
        float l_dot_h;
        float n_dot_h;
        float v_dot_h;
    };

    __host__ __device__ vec3 linear_interpolate(const vec3& begin, const vec3& end, float amount);

    __host__ __device__ float linear_interpolate(float begin, float end, float amount);

    __host__ __device__ vec3 base_color_to_specular_f0(const vec3& color, float metalness);

    __host__ __device__ vec3 base_color_to_diffuse_reflectance(const vec3& color, float metalness);

    __host__ __device__ vec3 eval_fresnel(const vec3& f0, float f90, float n_dot_s);

    __host__ __device__ float shadowed_f90(vec3 f0);

    __host__ __device__ float luminance(const vec3& rgb);

    __device__ brdf_data gen_brdf_data(const vec3& view, 
                                       const vec3& normal, 
                                       const vec3& light, 
                                       const microfacet& material);

    __host__ __device__ vec3 sample_GGX_VNDF(vec3 vec, float alpha_1, float alpha_2, float u, float v);

    __host__ __device__ vec3 sample_specular(const vec3& view_local, 
                                             float alpha, 
                                             float alpha_squared, 
                                             const vec3& specularF0, 
                                             float u, 
                                             float v, 
                                             vec3& out_weight);

    __host__ __device__ point cosine_sample_hemisphere(float u, float v, float& out_pdf);

    __host__ __device__ point ggx_sample_hemisphere(float u, float v, float roughness, const vec3 view_local, const vec3 normal_local, float& out_pdf);

    // __host__ __device__ point ggx_vndf_sample_hemisphere(float u, float v, float roughness, const vec3 view_local, float& out_pdf);

    __host__ __device__ point power_cosine_sample_hemisphere(float u, float v, float power, float& out_pdf);

    __host__ __device__ point uniform_sample_hemisphere(float u, float v, float& out_pdf);

    __host__ __device__ float D_GGX(float NoH, float roughness);

    __host__ __device__ vec3 F_Schlick(float u, vec3 f0);

    __host__ __device__ float V_SmithGGXCorrelated(float NoV, float NoL, float roughness);

    __host__ __device__ float Fd_Lambert();

    __host__ __device__ float Fd_DisneyDiffuse(brdf_data data);

    __host__ __device__ float V_Kelemen(float LoH);

    __host__ __device__ float D_GGX_Anisotropic(float NoH, const vec3 h, const vec3 t, const vec3 b, float at, float ab);

    __host__ __device__ float V_SmithGGXCorrelated_Anisotropic(float at, float ab, float ToV, float BoV, float ToL, float BoL, float NoV, float NoL);

    __device__ bool eval_brdf(float u, 
                              float v,
                              float t,
                              float in_refractive_index, 
                              vector normal, 
                              vector view, 
                              vector& out_ray_direction, 
                              vec3& out_sample_weight,
                              float& out_refractive_index,
                              const microfacet& material);

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
                                          int& refractive_idx_buffer_ptr);

}
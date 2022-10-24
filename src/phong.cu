#include "phong.cuh"

namespace pathtracer {

    __host__ __device__ vec3 phong_lighting(const object* thing,
                        const object* light,
                        const point* world_point,
                        const vec3* eye_vector,
                        const vec3* normal_vector) {
        vec3 effective_color = (thing->mat_d.phong.color & light->mat_d.light.color);

        point light_position = light->shape_d.sphere.transformation_to_world.transform_point({0.f, 0.f, 0.f});

        vector light_vector = (light_position - *world_point).normalize();

        vec3 ambient = effective_color * thing->mat_d.phong.ambient;

        vec3 diffuse;
        vec3 specular;

        float light_dot_normal = light_vector * *normal_vector;

        if (light_dot_normal < 0) {
            diffuse = {0.f, 0.f, 0.f};
            specular = {0.f, 0.f, 0.f};
        } else {
            diffuse = effective_color * thing->mat_d.phong.diffuse * light_dot_normal;

            vector reflection_vector = (-light_vector).reflect(*normal_vector);

            float reflect_dot_eye = reflection_vector * *eye_vector;

            if (reflect_dot_eye < 0) {
                specular = {0.f, 0.f, 0.f};
            } else {
                float factor = powf(reflect_dot_eye, thing->mat_d.phong.shininess);
                specular = light->mat_d.light.color * (thing->mat_d.phong.specular * factor);
            }
        }

        return ambient + diffuse + specular;
    }

}
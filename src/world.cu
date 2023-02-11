#include "check_cuda_errors.h"
#include "constants.h"
#include "scene.cuh"
#include "shapes.cuh"
#include "util.cuh"
#include "world.cuh"
#include "bvh.cuh"
#include "OBJ_Loader.h"

namespace pathtracer {

    // Do not use this with triangles!
    world::world(const std::initializer_list<object> l, dim3 blocks, dim3 threads):
        num_objects(l.size()), arena(new bvh_arena(l.size())) {
        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&objects), num_objects * sizeof(pathtracer::object)) );
        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&collision_buffer), blocks.x * blocks.y * threads.x * threads.y * 2 * num_objects * sizeof(int)) );
        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&intersection_buffer), blocks.x * blocks.y * threads.x * threads.y * 2 * num_objects * sizeof(intersection)) );

        int current_object{0};
        for (auto it = l.begin(); it < l.end(); ++it) {
            objects[current_object].shape_t = it->shape_t;
            switch (it->shape_t) {
                case SPHERE:
                    objects[current_object].shape_d.sphere = it->shape_d.sphere;
                    break;
            }
            objects[current_object].mat_t = it->mat_t;
            objects[current_object].mat_d = it->mat_d;
            ++current_object;
        }

        bvh_root = pathtracer::gen_bvh(objects, num_objects, arena);
    }

    world::world(const std::initializer_list<object*> l, dim3 blocks, dim3 threads):
        num_objects(l.size()), arena(new bvh_arena(l.size())) {
        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&objects), num_objects * sizeof(pathtracer::object)) );
        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&collision_buffer), blocks.x * blocks.y * threads.x * threads.y * 2 * num_objects * sizeof(int)) );
        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&intersection_buffer), blocks.x * blocks.y * threads.x * threads.y * 2 * num_objects * sizeof(intersection)) );

        int current_object{0};
        for (auto it = l.begin(); it < l.end(); ++it) {
            objects[current_object].shape_t = (*it)->shape_t;
            switch ((*it)->shape_t) {
                case SPHERE:
                    objects[current_object].shape_d.sphere = (*it)->shape_d.sphere;
                    break;
                case TRIANGLE:
                    objects[current_object].shape_d.triangle = (*it)->shape_d.triangle;
                    break;
                }
            objects[current_object].mat_t = (*it)->mat_t;
            objects[current_object].mat_d = (*it)->mat_d;
            ++current_object;
        }

        bvh_root = pathtracer::gen_bvh(objects, num_objects, arena);
    }

    world::world(const std::vector<object*> l, const std::vector<std::string> obj_filenames, const std::vector<mat4> obj_to_world_transformations, const std::vector<texture_data> texture_datas, const std::string environment_map_filename, dim3 blocks, dim3 threads) {
        int total_objects{static_cast<int>(l.size())};
        textures_length = obj_filenames.size();
        objl::Loader loader;

        // Calculate the total number of objects
        for (const std::string& filename : obj_filenames) {
            bool loadout = loader.LoadFile(filename);

            if (loadout) {
                for (int i{0}; i < loader.LoadedMeshes.size(); ++i) {
                    objl::Mesh curMesh = loader.LoadedMeshes[i];

                    total_objects += (curMesh.Indices.size() / 3);
                }
            } else {
                std::cout << "Failed to load file: " << filename << std::endl;
            }
        }

        // Initialize member variables
        num_objects = total_objects;
        arena = new bvh_arena(num_objects);
        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&objects), num_objects * sizeof(pathtracer::object)) );
        if (collision_buffer_limit_enable && num_objects > collision_buffer_limit) {
            checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&collision_buffer), blocks.x * blocks.y * threads.x * threads.y * 2 * collision_buffer_limit * sizeof(int)) );
        } else {
            checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&collision_buffer), blocks.x * blocks.y * threads.x * threads.y * 2 * num_objects * sizeof(int)) );
        }
        if (intersection_buffer_limit_enable && num_objects > intersection_buffer_limit) {
            checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&intersection_buffer), blocks.x * blocks.y * threads.x * threads.y * 2 * intersection_buffer_limit * sizeof(intersection)) );
        } else {
            checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&intersection_buffer), blocks.x * blocks.y * threads.x * threads.y * 2 * num_objects * sizeof(intersection)) );

        }
        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&textures), obj_filenames.size() * sizeof(float*)) );
        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&is_texture_idx_valid), obj_filenames.size() * sizeof(bool)) );
        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&(this->texture_datas)), obj_filenames.size() * sizeof(texture_data)) );

        for (int i{0}; i < obj_filenames.size(); ++i) {
            is_texture_idx_valid[i] = false;
        }

        // Build the objects
        int current_object{0};
        for (auto it = l.begin(); it < l.end(); ++it) {
            objects[current_object].shape_t = (*it)->shape_t;
            switch ((*it)->shape_t) {
                case SPHERE:
                    objects[current_object].shape_d.sphere = (*it)->shape_d.sphere;
                    break;
                case TRIANGLE:
                    objects[current_object].shape_d.triangle = (*it)->shape_d.triangle;
                    break;
                }
            objects[current_object].mat_t = (*it)->mat_t;
            objects[current_object].mat_d = (*it)->mat_d;
            ++current_object;
        }

        // Build the triangles
        for (int f{0}; f < obj_filenames.size(); ++f) {
            const std::string& filename = obj_filenames[f];
            bool loadout = loader.LoadFile(filename);

            // Load the texture corresponding the obj_filename
            const std::string& texture_filename = texture_datas[f].filename;
            int texture_idx = -1;
            if (texture_filename.size() != 0) {
                float* out; // height * width * RGBA
                int width;
                int height;
                const char* err = nullptr;

                int ret = LoadEXR(&out, &width, &height, texture_filename.c_str(), &err);

                if (ret != TINYEXR_SUCCESS) {
                    if (err) {
                    fprintf(stderr, "ERR : %s\n", err);
                    FreeEXRErrorMessage(err); // release memory of error message.
                    }
                } else {
                    // Copy image data into CUDA-managed memory
                    float* texture;
                    checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&texture), width * height * 4 * sizeof(float)) );
                    for (int t{0}; t < width * height * 4; ++t) {
                        texture[t] = out[t];
                    }
                    textures[f] = texture;
                    this->texture_datas[f] = texture_datas[f];
                    free(out); // release memory of image data
                }
                texture_idx = f;
                is_texture_idx_valid[f] = true;

                // for (int p{0}; p < width * height * 4; p += 4) {
                //     std::cout << textures[f][p] << " " << textures[f][p + 1] << " " << textures[f][p + 2] << " " << textures[f][p + 3] << " " << std::endl;
                // }

                // std::cout << pathtracer::vector {textures[f][offset + 0], textures[f][offset + 1], textures[f][offset + 2]} << std::endl;
            }

            if (loadout) {
                for (int i{0}; i < loader.LoadedMeshes.size(); ++i) {
                    objl::Mesh curMesh = loader.LoadedMeshes[i];

                    for (int j{0}; j < curMesh.Indices.size(); j += 3) {
                        objl::Vector3 vertex1 = curMesh.Vertices[curMesh.Indices[j]].Position;
                        objl::Vector3 vertex2 = curMesh.Vertices[curMesh.Indices[j + 1]].Position;
                        objl::Vector3 vertex3 = curMesh.Vertices[curMesh.Indices[j + 2]].Position;

                        vec3 v1 = obj_to_world_transformations[f].transform_point({vertex1.X, vertex1.Y, vertex1.Z});
                        vec3 v2 = obj_to_world_transformations[f].transform_point({vertex2.X, vertex2.Y, vertex2.Z});
                        vec3 v3 = obj_to_world_transformations[f].transform_point({vertex3.X, vertex3.Y, vertex3.Z});

                        // vec3 v1 = {vertex1.X, vertex1.Y, vertex1.Z};
                        // vec3 v2 = {vertex2.X, vertex2.Y, vertex2.Z};
                        // vec3 v3 = {vertex3.X, vertex3.Y, vertex3.Z};

                        objl::Vector3 normal1 = curMesh.Vertices[curMesh.Indices[j]].Normal;
                        objl::Vector3 normal2 = curMesh.Vertices[curMesh.Indices[j + 1]].Normal;
                        objl::Vector3 normal3 = curMesh.Vertices[curMesh.Indices[j + 2]].Normal;

                        vec3 n1 = {normal1.X, normal1.Y, normal1.Z};
                        vec3 n2 = {normal2.X, normal2.Y, normal2.Z};
                        vec3 n3 = {normal3.X, normal3.Y, normal3.Z};

                        // vector average_normal = {
                        //     (normal1.X + normal2.X + normal3.X) / 3.f,
                        //     (normal1.Y + normal2.Y + normal3.Y) / 3.f,
                        //     (normal1.Z + normal2.Z + normal3.Z) / 3.f
                        // };
                        bool success_flag;
                        n1 = obj_to_world_transformations[f].inverse(success_flag).transpose().transform_vector(n1).normalize();
                        n2 = obj_to_world_transformations[f].inverse(success_flag).transpose().transform_vector(n2).normalize();
                        n3 = obj_to_world_transformations[f].inverse(success_flag).transpose().transform_vector(n3).normalize();
                        // average_normal = obj_to_world_transformations[f].inverse(success_flag).transpose().transform_vector(average_normal).normalize();

                        objl::Vector2 tex1 = curMesh.Vertices[curMesh.Indices[j]].TextureCoordinate;
                        objl::Vector2 tex2 = curMesh.Vertices[curMesh.Indices[j + 1]].TextureCoordinate;
                        objl::Vector2 tex3 = curMesh.Vertices[curMesh.Indices[j + 2]].TextureCoordinate;

                        vec3 t1 = {tex1.X, tex1.Y, 0.f};
                        vec3 t2 = {tex2.X, tex2.Y, 0.f};
                        vec3 t3 = {tex3.X, tex3.Y, 0.f};

                        objects[current_object].shape_t = TRIANGLE;
                        objects[current_object].shape_d.triangle = triangle(v1, v2, v3, n1, n2, n3, t1, t2, t3, texture_idx);
                        // objects[current_object].shape_d.triangle.normal = average_normal;
                        objects[current_object].mat_t = MICROFACET;
                        vec3 color{
                            curMesh.MeshMaterial.Ka.X,
                            curMesh.MeshMaterial.Ka.Y,
                            curMesh.MeshMaterial.Ka.Z
                        };
                        objects[current_object].mat_d.microfacet = microfacet{color, {0.f, 0.f, 0.f}, 1.f, 0.3f, 0.f, 4.f, 0.04f, 0.f, 0.f, -0.3f};
                        ++current_object;
                    }
                }
            } else {
                std::cout << "Failed to load file: " << filename << std::endl;
            }
        }

        // Create the environment map
        if (environment_map_filename != "") {
            float* out; // height * width * RGBA
            int width;
            int height;
            const char* err = nullptr;

            int ret = LoadEXR(&out, &width, &height, environment_map_filename.c_str(), &err);

            if (ret != TINYEXR_SUCCESS) {
                if (err) {
                fprintf(stderr, "ERR : %s\n", err);
                FreeEXRErrorMessage(err); // release memory of error message.
                }
            } else {
                // Copy image data into CUDA-managed memory
                checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&environment_map), width * height * 4 * sizeof(float)) );
                for (int t{0}; t < width * height * 4; ++t) {
                    environment_map[t] = out[t];
                }
                environment_map_height = height;
                environment_map_width = width;
                free(out); // release memory of image data
            }
        } else {
            environment_map = nullptr;
        }

        bvh_root = pathtracer::gen_bvh(objects, num_objects, arena);
    }

    __host__ __device__ computations world::prepare_computations(const intersection& intersection, const ray& r) {
        point surface_point = r.shoot_distance(intersection.t_value);
        vector surface_normal;
        object& intersected_object = objects[intersection.object_index];
        switch (intersected_object.shape_t) {
            case SPHERE:
                surface_normal = intersected_object.shape_d.sphere.world_normal_at(surface_point);
                break;
            case TRIANGLE:
                surface_normal = intersected_object.shape_d.triangle.world_normal_at(surface_point, intersection.u, intersection.v);
                break;
            }
        vector eye = (-r.d).normalize();
        bool inside = false;
        if (surface_normal * eye < 0) {
            inside = true;
            surface_normal = -surface_normal;
        }
        return {
            {intersection.t_value, intersection.object_index, intersection.u, intersection.v},
            surface_point,
            eye,
            surface_normal,
            inside
        };
    }

    __host__ __device__ computations world::intersect_world(const ray& r, bool& success_flag, int* collision_buffer, pathtracer::intersection* intersection_buffer) {
        int possible_intersections = r.find_intersections(bvh_root, collision_buffer);

        pathtracer::intersection* intersection_buffer_ptr = intersection_buffer;
        int intersection_buffer_size{0};

        for (int k{0}; k < possible_intersections; ++k) {
                int object_index = collision_buffer[k];

                int num_intersections;

                switch (objects[object_index].shape_t) {
                    case SPHERE:
                        num_intersections = objects[object_index].shape_d.sphere.find_intersections(r, intersection_buffer_ptr, object_index);
                        break;
                    case TRIANGLE:
                        num_intersections = objects[object_index].shape_d.triangle.find_intersections(r, intersection_buffer_ptr, object_index);
                        break;
                    }

                intersection_buffer_ptr += num_intersections;
                intersection_buffer_size += num_intersections;
        }

        pathtracer::intersection* closest_positive_intersection = pathtracer::get_closest_positive_intersection(intersection_buffer, intersection_buffer_size);

        if (closest_positive_intersection != nullptr) {
            success_flag = true;
            return prepare_computations(*closest_positive_intersection, r);
        } else {
            success_flag = false;
            return {};
        }
    }

    __host__ void world::free_world() {
        arena->free_arena();
        checkCudaErrors( cudaFree(objects) );
        checkCudaErrors( cudaFree(collision_buffer) );
        checkCudaErrors( cudaFree(intersection_buffer) );
        checkCudaErrors( cudaFree(texture_datas) );
        checkCudaErrors( cudaFree(environment_map) );

        for (int i{0}; i < textures_length; ++i) {
            checkCudaErrors( cudaFree(textures[i]) );
        }

        checkCudaErrors( cudaFree(textures) );
        checkCudaErrors( cudaFree(is_texture_idx_valid) );
    }

}
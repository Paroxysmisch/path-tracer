#pragma once

namespace pathtracer {

    __host__ __device__ bool f_equal(float a, float b);

    struct vec3 {
        float x;
        float y;
        float z;

        __host__ __device__ vec3 operator+(vec3& other);

        __host__ __device__ vec3& operator+=(vec3& other);

        __host__ __device__ vec3 operator-(vec3& other);

        __host__ __device__ vec3& operator-=(vec3& other);

        __host__ __device__ vec3 operator-();

        __host__ __device__ vec3 operator*(float scalar);

        __host__ __device__ vec3& operator*=(float scalar);

        __host__ __device__ vec3 operator/(float scalar);

        __host__ __device__ vec3& operator/=(float scalar);

        __host__ __device__ float mag();

        __host__ __device__ vec3& normalize();

        __host__ __device__ float operator*(vec3& other);

        __host__ __device__ vec3 operator^(vec3& other);

        __host__ __device__ vec3& operator^=(vec3& other);
    };

    using point = vec3;
    using vector = vec3;

}

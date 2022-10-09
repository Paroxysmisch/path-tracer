#pragma once

#include <array>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "check_cuda_errors.h"

namespace pathtracer {

    __host__ __device__ bool f_equal(float a, float b);

    struct vec3 {
        float x;
        float y;
        float z;

        __host__ __device__ bool operator==(const vec3& other) const;

        __host__ __device__ vec3 operator+(const vec3& other) const;

        __host__ __device__ vec3& operator+=(const vec3& other);

        __host__ __device__ vec3 operator-(const vec3& other) const;

        __host__ __device__ vec3& operator-=(const vec3& other);

        __host__ __device__ vec3 operator-() const;

        __host__ __device__ vec3 operator*(const float scalar) const;

        __host__ __device__ vec3& operator*=(const float scalar);

        __host__ __device__ vec3 operator&(const vec3& other) const;

        __host__ __device__ vec3& operator&=(const vec3& other);

        __host__ __device__ vec3 operator/(const float scalar) const;

        __host__ __device__ vec3& operator/=(const float scalar);

        __host__ __device__ float mag() const;

        __host__ __device__ vec3& normalize();

        __host__ __device__ float operator*(const vec3& other) const;

        __host__ __device__ vec3 operator^(const vec3& other) const;

        __host__ __device__ vec3& operator^=(const vec3& other);
    };

    using point = vec3;
    using vector = vec3;
    using color = vec3;

    __host__ __device__ unsigned char to_byte(float n);

    template <size_t height, size_t width>
    struct canvas {
        vec3 m_data[height][width];

        __host__ canvas();

        __host__ ~canvas();

        __host__ __device__ canvas& write_pixel(size_t y, size_t x, const vec3& data);

        __host__ void export_as_PPM(const std::string& filename) const;
    };

    template <size_t height, size_t width>
    __host__ canvas<height, width>::canvas() {
        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&m_data), 
                                           sizeof(std::array<std::array<vec3, width>, height>)) );

        for (size_t i{0}; i < height; ++i) {
            for (size_t j{0}; j < width; ++j) {
                (m_data)[i][j] = {0, 0, 0};
            }
        }
    }

    template <size_t height, size_t width>
    __host__ canvas<height, width>::~canvas() {
        cudaFree(m_data);
    }

    template <size_t height, size_t width>
    __host__ __device__ canvas<height, width>& canvas<height, width>::write_pixel(size_t y, size_t x, const vec3& data) {
            m_data[y][x] = data;
            return *this;
    }

    template <size_t height, size_t width>
    __host__ void canvas<height, width>::export_as_PPM(const std::string& filename) const {
        std::ofstream file{filename};

        file << "P3" << std::endl
             << width << " " << height << std::endl
             << 255 << std::endl;

        constexpr size_t max_per_line{64};

        size_t counter{0};

        for (size_t i{0}; i < height; ++i) {
            for (size_t j{0}; j < width; ++j) {
                if (counter >= max_per_line) {
                    file << std::endl;
                    counter = 0;
                }

                const vec3& color = m_data[i][j];

                file << static_cast<int>(to_byte(color.x)) << " "
                     << static_cast<int>(to_byte(color.y)) << " "
                     << static_cast<int>(to_byte(color.z)) << " ";

                ++counter;
            }
        }

        file.close();
    }

}

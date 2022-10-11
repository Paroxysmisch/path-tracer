#include "util.cuh"
#include "constants.h"
#include <cmath>
#include <cstddef>

namespace pathtracer {

    __host__ __device__ bool f_equal(float a, float b) {
        return fabs(a - b) < epsilon;
    }

    __host__ __device__ bool vec3::operator==(const vec3& other) const {
        return f_equal(x, other.x) && f_equal(y, other.y) && f_equal(z, other.z);
    }

    __host__ __device__ vec3 vec3::operator+(const vec3& other) const {
        return vec3{x + other.x, y + other.y, z + other.z};
    }

    __host__ __device__ vec3& vec3::operator+=(const vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    __host__ __device__ vec3 vec3::operator-(const vec3& other) const {
        return vec3{x - other.x, y - other.y, z - other.z};
    }

    __host__ __device__ vec3& vec3::operator-=(const vec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    __host__ __device__ vec3 vec3::operator-() const {
        return vec3{-x, -y, -z};
    }

    __host__ __device__ vec3 vec3::operator*(const float scalar) const {
        return vec3{x * scalar, y * scalar, z * scalar};
    }

    __host__ __device__ vec3& vec3::operator*=(const float scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    __host__ __device__ vec3 vec3::operator&(const vec3& other) const {
        return vec3{x * other.x, y * other.y, z * other.z};
    }

    __host__ __device__ vec3& vec3::operator&=(const vec3& other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        return *this;
    }

    __host__ __device__ vec3 vec3::operator/(const float scalar) const {
        return vec3{x / scalar, y / scalar, z / scalar};
    }

    __host__ __device__ vec3& vec3::operator/=(const float scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    __host__ __device__ float vec3::mag() const {
        return sqrtf(x*x + y*y + z*z);
    }

    __host__ __device__ vec3& vec3::normalize() {
        float mag = this->mag();
        (*this) /= mag;
        return *this;
    }

    __host__ __device__ float vec3::operator*(const vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    __host__ __device__ vec3 vec3::operator^(const vec3& other) const {
        return vec3{
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        };
    }

    __host__ __device__ vec3& vec3::operator^=(const vec3& other) {
        float temp_x = (y * other.z - z * other.y);
        float temp_y = (z * other.x - x * other.z);
        float temp_z = (x * other.y - y * other.x);

        x = temp_x;
        y = temp_y;
        z = temp_z;
        return *this;
    }

    __host__ __device__ unsigned char to_byte(float n) {
        if (n < 0.f) return 0;
        else if (n >= 1.f) return 255;
        else return (unsigned char) 255.99f * n;
    }

    __host__ __device__ mat4::mat4() {}

    __host__ __device__ mat4::mat4(float _00, float _01, float _02, float _03,
                                float _10, float _11, float _12, float _13,
                                float _20, float _21, float _22, float _23,
                                float _30, float _31, float _32, float _33) {
        m_data[0] = _00; m_data[1] = _01; m_data[2] = _02; m_data[3] = _03;
        m_data[4] = _10; m_data[5] = _11; m_data[6] = _12; m_data[7] = _13;
        m_data[8] = _20; m_data[9] = _21; m_data[10] = _22; m_data[11] = _23;
        m_data[12] = _30; m_data[13] = _31; m_data[14] = _32; m_data[15] = _33;
    }

    __host__ __device__ bool mat4::operator==(const mat4& other) const {
        for (size_t i{0}; i < 16; ++i) {
            if (!f_equal(m_data[i], other.m_data[i])) return false;
        }
        return true;
    }

    __host__ __device__ mat4 mat4::operator*(const mat4& other) const {
        mat4 result{};

        for (int i{0}; i < 4; ++i) {
            for (int j{0}; j < 4; ++j) {
                float v{0.f};
                for (int k{0}; k < 4; ++k) {
                    v += (m_data[i * 4 + k] * other.m_data[k * 4 + j]);
                }
                result.m_data[i * 4 + j] = v;
            }
        }

        return result;
    }

    __host__ __device__ mat4& mat4::operator*=(const mat4& other) {
        mat4 temp = (*this) * other;

        for (int i{0}; i < 16; ++i) {
            m_data[i] = temp.m_data[i];
        }

        return *this;
    }

}

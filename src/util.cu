#include "util.cuh"
#include "constants.h"
#include <cmath>

namespace pathtracer {

    __host__ __device__ bool f_equal(float a, float b) {
        return fabs(a - b) < epsilon;
    }

    __host__ __device__ bool vec3::operator==(vec3& other) {
        return f_equal(x, other.x) && f_equal(y, other.y) && f_equal(z, other.z);
    }

    __host__ __device__ vec3 vec3::operator+(vec3& other) {
        return vec3{x + other.x, y + other.y, z + other.z};
    }

    __host__ __device__ vec3& vec3::operator+=(vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    __host__ __device__ vec3 vec3::operator-(vec3& other) {
        return vec3{x - other.x, y - other.y, z - other.z};
    }

    __host__ __device__ vec3& vec3::operator-=(vec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    __host__ __device__ vec3 vec3::operator*(float scalar) {
        return vector{x * scalar, y * scalar, z * scalar};
    }

    __host__ __device__ vec3& vec3::operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    __host__ __device__ vec3 vec3::operator/(float scalar) {
        return vector{x / scalar, y / scalar, z / scalar};
    }

    __host__ __device__ vec3& vec3::operator/=(float scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    __host__ __device__ float vec3::mag() {
        return sqrt(x*x + y*y + z*z);
    }

    __host__ __device__ vec3& vec3::normalize() {
        float mag = this->mag();
        (*this) /= mag;
        return *this;
    }

    __host__ __device__ float vec3::operator*(vec3& other) {
        return x * other.x + y * other.y + z * other.z;
    }

    __host__ __device__ vec3 vec3::operator^(vec3& other) {
        return vec3{
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        };
    }

    __host__ __device__ vec3& vec3::operator^=(vec3& other) {
        x = y * other.z - z * other.y;
        y = z * other.x - x * other.z;
        z = x * other.y - y * other.x;
        return *this;
    }

}

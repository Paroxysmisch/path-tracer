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

    __host__ __device__ float vec3::mag_2() const {
        return x*x + y*y + z*z;
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

    __host__ __device__ vec3 vec3::gen_orthogonal(const vec3& v) {
        float x = fabs(v.x);
        float y = fabs(v.y);
        float z = fabs(v.z);

        vec3 temp = x < y ? (x < z ? vec3{1.f, 0.f, 0.f} : vec3{0.f, 0.f, 1.f}) :
                            (y < z ? vec3{0.f, 1.f, 0.f} : vec3{0.f, 0.f, 1.f});

        return v ^ temp;
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

    __host__ __device__  mat4 mat4::get_identity() {
        return mat4{1.f, 0.f, 0.f, 0.f,
                    0.f, 1.f, 0.f, 0.f,
                    0.f, 0.f, 1.f, 0.f,
                    0.f, 0.f, 0.f, 1.f};
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

    __host__ __device__ mat4 mat4::transpose() {
        mat4 result{};

        for (int i{0}; i < 4; ++i) {
            for (int j{0}; j < 4; ++j) {
                result.m_data[j * 4 + i] = m_data[i * 4 + j];
            }
        }

        return result;
    }

    __host__ __device__ mat4& mat4::operator=(mat4& other) {
        for (int i{0}; i < 16; ++i) {
            m_data[i] = other.m_data[i];
        }

        return *this;
    }

    __host__ __device__ mat4 mat4::inverse(bool& success_flag) {
        float* m = m_data;

        float inv_temp[16];

        inv_temp[0] = m[5]  * m[10] * m[15] - 
                      m[5]  * m[11] * m[14] - 
                      m[9]  * m[6]  * m[15] + 
                      m[9]  * m[7]  * m[14] +
                      m[13] * m[6]  * m[11] - 
                      m[13] * m[7]  * m[10];

        inv_temp[4] = -m[4]  * m[10] * m[15] + 
                       m[4]  * m[11] * m[14] + 
                       m[8]  * m[6]  * m[15] - 
                       m[8]  * m[7]  * m[14] - 
                       m[12] * m[6]  * m[11] + 
                       m[12] * m[7]  * m[10];

        inv_temp[8] = m[4]  * m[9] * m[15] - 
                      m[4]  * m[11] * m[13] - 
                      m[8]  * m[5] * m[15] + 
                      m[8]  * m[7] * m[13] + 
                      m[12] * m[5] * m[11] - 
                      m[12] * m[7] * m[9];

        inv_temp[12] = -m[4]  * m[9] * m[14] + 
                        m[4]  * m[10] * m[13] +
                        m[8]  * m[5] * m[14] - 
                        m[8]  * m[6] * m[13] - 
                        m[12] * m[5] * m[10] + 
                        m[12] * m[6] * m[9];

        inv_temp[1] = -m[1]  * m[10] * m[15] + 
                       m[1]  * m[11] * m[14] + 
                       m[9]  * m[2] * m[15] - 
                       m[9]  * m[3] * m[14] - 
                       m[13] * m[2] * m[11] + 
                       m[13] * m[3] * m[10];

        inv_temp[5] = m[0]  * m[10] * m[15] - 
                      m[0]  * m[11] * m[14] - 
                      m[8]  * m[2] * m[15] + 
                      m[8]  * m[3] * m[14] + 
                      m[12] * m[2] * m[11] - 
                      m[12] * m[3] * m[10];

        inv_temp[9] = -m[0]  * m[9] * m[15] + 
                       m[0]  * m[11] * m[13] + 
                       m[8]  * m[1] * m[15] - 
                       m[8]  * m[3] * m[13] - 
                       m[12] * m[1] * m[11] + 
                       m[12] * m[3] * m[9];

        inv_temp[13] = m[0]  * m[9] * m[14] - 
                       m[0]  * m[10] * m[13] - 
                       m[8]  * m[1] * m[14] + 
                       m[8]  * m[2] * m[13] + 
                       m[12] * m[1] * m[10] - 
                       m[12] * m[2] * m[9];

        inv_temp[2] = m[1]  * m[6] * m[15] - 
                      m[1]  * m[7] * m[14] - 
                      m[5]  * m[2] * m[15] + 
                      m[5]  * m[3] * m[14] + 
                      m[13] * m[2] * m[7] - 
                      m[13] * m[3] * m[6];

        inv_temp[6] = -m[0]  * m[6] * m[15] + 
                       m[0]  * m[7] * m[14] + 
                       m[4]  * m[2] * m[15] - 
                       m[4]  * m[3] * m[14] - 
                       m[12] * m[2] * m[7] + 
                       m[12] * m[3] * m[6];

        inv_temp[10] = m[0]  * m[5] * m[15] - 
                       m[0]  * m[7] * m[13] - 
                       m[4]  * m[1] * m[15] + 
                       m[4]  * m[3] * m[13] + 
                       m[12] * m[1] * m[7] - 
                       m[12] * m[3] * m[5];

        inv_temp[14] = -m[0]  * m[5] * m[14] + 
                        m[0]  * m[6] * m[13] + 
                        m[4]  * m[1] * m[14] - 
                        m[4]  * m[2] * m[13] - 
                        m[12] * m[1] * m[6] + 
                        m[12] * m[2] * m[5];

        inv_temp[3] = -m[1] * m[6] * m[11] + 
                       m[1] * m[7] * m[10] + 
                       m[5] * m[2] * m[11] - 
                       m[5] * m[3] * m[10] - 
                       m[9] * m[2] * m[7] + 
                       m[9] * m[3] * m[6];

        inv_temp[7] = m[0] * m[6] * m[11] - 
                      m[0] * m[7] * m[10] - 
                      m[4] * m[2] * m[11] + 
                      m[4] * m[3] * m[10] + 
                      m[8] * m[2] * m[7] - 
                      m[8] * m[3] * m[6];

        inv_temp[11] = -m[0] * m[5] * m[11] + 
                        m[0] * m[7] * m[9] + 
                        m[4] * m[1] * m[11] - 
                        m[4] * m[3] * m[9] - 
                        m[8] * m[1] * m[7] + 
                        m[8] * m[3] * m[5];

        inv_temp[15] = m[0] * m[5] * m[10] - 
                       m[0] * m[6] * m[9] - 
                       m[4] * m[1] * m[10] + 
                       m[4] * m[2] * m[9] + 
                       m[8] * m[1] * m[6] - 
                       m[8] * m[2] * m[5];

        float det = m[0] * inv_temp[0] + m[1] * inv_temp[4] + m[2] * inv_temp[8] + m[3] * inv_temp[12];

        if (det == 0) {
            success_flag = false;
            return {};
        };

        det = 1.f / det;

        for (int i{0}; i < 16; ++i) inv_temp[i] = inv_temp[i] * det;

        success_flag = true;

        return mat4{inv_temp[0],inv_temp[1],inv_temp[2],inv_temp[3],
                    inv_temp[4], inv_temp[5], inv_temp[6], inv_temp[7],
                    inv_temp[8], inv_temp[9], inv_temp[10], inv_temp[11],
                    inv_temp[12], inv_temp[13], inv_temp[14], inv_temp[15]};
    }

    __host__ __device__ quaternion::quaternion(float w, vec3 ijk): w(w), ijk(ijk) {}

    __host__ __device__ quaternion::quaternion(float w, float i, float j, float k): w(w), ijk({i,j,k}) {}

    __host__ __device__ quaternion quaternion::conjugate() {
        return quaternion{w, -ijk};
    }

    __host__ __device__ quaternion& quaternion::normalize() {
        float normalization_factor = sqrtf(w * w + ijk.mag_2());

        w /= normalization_factor;
        ijk /= normalization_factor;

        return *this;
    }

    __host__ __device__ quaternion quaternion::get_rotation_between(vec3 u, vec3 v) {
        u = u.normalize();
        v = v.normalize();

        if (u == -v)
            return quaternion(0.f, (u ^ v).normalize());

        vec3 half = (u + v).normalize();
        return quaternion(u * half, u ^ half);
    }

    __host__ __device__ vec3 quaternion::rotate_vector_by_quaternion(const vec3& v, const quaternion& q) {
        return q.ijk * (q.ijk * v) * 2.f +
               v * (q.w * q.w - q.ijk * q.ijk) +
               (q.ijk ^ v) * 2.f * q.w;
    }

    __host__ __device__ quaternion quaternion::get_rotation_to_z_axis(const vec3& normalized_v) {
        if (f_equal(normalized_v.z, -1.f)) return quaternion(0.f, 1.f, 0.f, 0.f);

        return quaternion(1.f + normalized_v.z, normalized_v.y, -normalized_v.x, 0.f).normalize();
    }

    __host__ __device__ quaternion quaternion::get_rotation_from_z_axis(const vec3& normalized_v) {
        if (f_equal(normalized_v.z, -1.f)) return quaternion(0.f, 1.f, 0.f, 0.f);

        return quaternion(1.f + normalized_v.z, -normalized_v.y, normalized_v.x, 0.f).normalize();
    }

    __host__ __device__ quaternion quaternion::get_inverse_rotation(const quaternion& q) {
        return quaternion(q.w, -q.ijk);
    }

}

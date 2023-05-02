#include "util.cuh"
#include "constants.h"
#include <cmath>
#include <cstddef>

namespace pathtracer {

    __host__ __device__ bool f_equal(float a, float b) {
        return fabs(a - b) < epsilon;
    }

    __host__ __device__ vec3_d::vec3_d() {}

    __host__ __device__ vec3_d::vec3_d(float x, float y, float z): x(x), y(y), z(z) {}

    __host__ __device__ vec3_d::vec3_d(double x, double y, double z): x(x), y(y), z(z) {}

    __host__ __device__ vec3_d& vec3_d::operator+=(const vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    __host__ __device__ vec3_d vec3_d::operator&(const vec3_d& other) const {
        return vec3_d{x * other.x, y * other.y, z * other.z};
    }

    __host__ __device__ vec3 vec3_d::operator-(const float scalar) {
        return vec3{static_cast<float>(x - scalar), static_cast<float>(y - scalar), static_cast<float>(z - scalar)};
    };

    __host__ __device__ vec3 vec3_d::operator-(const vec3& other) {
        return vec3{static_cast<float>(x - other.x), static_cast<float>(y - other.y), static_cast<float>(z - other.z)};
    };

    __host__ __device__ vec3 vec3_d::operator/(const float scalar) {
        return vec3{static_cast<float>(x / scalar), static_cast<float>(y / scalar), static_cast<float>(z / scalar)};
    };

    __host__ __device__ vec3_d& vec3_d::operator/=(const float scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    __host__ __device__ vec3::vec3() {}

    __host__ __device__ vec3::vec3(float x, float y, float z): x(x), y(y), z(z) {}

    __host__ __device__ vec3::vec3(float x): x(x), y(x), z(x) {}

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

    __host__ __device__ vector vec3::reflect(vector normal) {
        return *this - (normal * 2 * (*this * normal));
    }

    __host__ std::ostream &operator<<(std::ostream &os, const vec3& vec_3) {
        return os << "(" << vec_3.x << ", " << vec_3.y << ", " << vec_3.z << ")";
    }

    __host__ __device__ unsigned char to_byte(float n) {
        if (n < 0.f) return 0;
        else if (n >= 1.f) return 255;
        else return (unsigned char) 255.99f * n;
    }

    __host__ canvas::canvas(const int height, const int width): m_owner(true), m_height(height), m_width(width) {
        checkCudaErrors( cudaMallocManaged(reinterpret_cast<void**>(&m_data), 
                                           sizeof(vec3) * m_height * m_width) );

        for (size_t i{0}; i < m_height * m_width; ++i) {
            (m_data)[i] = {0, 0, 0};
        }
    }

    __host__ canvas::canvas(canvas& other): m_owner(false), m_data(other.m_data), m_height(other.m_height), m_width(other.m_width) {}

    __host__ canvas::~canvas() {
        if (m_owner) {
            cudaFree(m_data);
        }
    }

    __host__ __device__ canvas& canvas::write_pixel(int y, int x, const vec3& data) {
            m_data[y * m_width + x] = data;
            return *this;
    }

    __host__ void canvas::export_as_PPM(const std::string& filename) const {
        std::ofstream file{filename};

        file << "P3" << std::endl
             << m_width << " " << m_height << std::endl
             << 255 << std::endl;

        constexpr size_t max_per_line{64};

        size_t counter{0};

        for (size_t i{0}; i < m_height; ++i) {
            for (size_t j{0}; j < m_width; ++j) {
                if (counter >= max_per_line) {
                    file << std::endl;
                    counter = 0;
                }

                const vec3& color = m_data[i * m_width + j];

                file << static_cast<int>(to_byte(color.x)) << " "
                     << static_cast<int>(to_byte(color.y)) << " "
                     << static_cast<int>(to_byte(color.z)) << " ";

                ++counter;
            }
        }

        // PPMs end with a newline character
        file << std::endl;

        file.close();
    }

    __host__ bool canvas::export_as_EXR(const std::string& filename) const {
        float* data = new float[m_height * m_width * 3];

        for (int i{0}; i < m_height * m_width; ++i) {
            int index = i * 3;
            data[index] = m_data[i].x;
            data[index + 1] = m_data[i].y;
            data[index + 2] = m_data[i].z;
        }

        const char* err = new char[64];

        bool res = SaveEXR(data, m_width, m_height, 3, 0, filename.c_str(), &err);

        if (res != TINYEXR_SUCCESS)
            std::cout << err << std::endl;

        delete[] data;
        delete[] err;

        return res;
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

    __host__ __device__ mat4& mat4::operator=(const mat4& other) {
        for (int i{0}; i < 16; ++i) {
            m_data[i] = other.m_data[i];
        }

        return *this;
    }

    __host__ __device__ mat4 mat4::inverse(bool& success_flag) const {
        const float* m = m_data;

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

    __host__ __device__ mat4 mat4::get_translation(float x, float y, float z) {
        return mat4(1.f, 0.f, 0.f, x,
                    0.f, 1.f, 0.f, y,
                    0.f, 0.f, 1.f, z,
                    0.f, 0.f, 0.f, 1.f);
    }

    __host__ __device__ mat4 mat4::get_scaling(float x, float y, float z) {
        return mat4(x, 0.f, 0.f, 0.f,
                    0.f, y, 0.f, 0.f,
                    0.f, 0.f, z, 0.f,
                    0.f, 0.f, 0.f, 1.f);
    }

    __host__ __device__ mat4 mat4::get_rotation_x(float rad) {
        float cos_term = cosf(rad);
        float sin_term = sinf(rad);
        return mat4(1.f, 0.f,      0.f,       0.f,
                    0.f, cos_term, -sin_term, 0.f,
                    0.f, sin_term, cos_term,  0.f,
                    0.f, 0.f,      0.f,       1.f);
    }

    __host__ __device__ mat4 mat4::get_rotation_y(float rad) {
        float cos_term = cosf(rad);
        float sin_term = sinf(rad);
        return mat4(cos_term, 0.f, sin_term,  0.f,
                    0.f,      1.f, 0.f,       0.f,
                    -sin_term,0.f, cos_term,  0.f,
                    0.f,      0.f, 0.f,       1.f);
    }

    __host__ __device__ mat4 mat4::get_rotation_z(float rad) {
        float cos_term = cosf(rad);
        float sin_term = sinf(rad);
        return mat4(cos_term, -sin_term, 0.f, 0.f,
                    sin_term, cos_term,  0.f, 0.f,
                    0.f,      0.f,       1.f, 0.f,
                    0.f,      0.f,       0.f, 1.f);
    }

    __host__ __device__ mat4 mat4::get_shearing(float x_y, float x_z,
                                          float y_x, float y_z,
                                          float z_x, float z_y) {
        return mat4(1.f, x_y, x_z, 0.f,
                    y_x, 1.f, y_z, 0.f,
                    z_x, z_y, 1.f, 0.f,
                    0.f, 0.f, 0.f, 1.f);
    }

    __host__ __device__ point mat4::transform_point(const point& p) const {
        point result{};

        result.x = m_data[0] * p.x +
                   m_data[1] * p.y +
                   m_data[2] * p.z +
                   m_data[3];
        result.y = m_data[4 + 0] * p.x +
                   m_data[4 + 1] * p.y +
                   m_data[4 + 2] * p.z +
                   m_data[4 + 3];
        result.z = m_data[8 + 0] * p.x +
                   m_data[8 + 1] * p.y +
                   m_data[8 + 2] * p.z +
                   m_data[8 + 3];
            
        return result;
    }

    __host__ __device__ vector mat4::transform_vector(const vector& v) const {
        vector result{};

        result.x = m_data[0] * v.x +
                   m_data[1] * v.y +
                   m_data[2] * v.z;
        result.y = m_data[4 + 0] * v.x +
                   m_data[4 + 1] * v.y +
                   m_data[4 + 2] * v.z;
        result.z = m_data[8 + 0] * v.x +
                   m_data[8 + 1] * v.y +
                   m_data[8 + 2] * v.z;

        return result;
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

    __host__ __device__ float maxf(float a, float b) {
        return a >= b ? a : b;
    }

    __host__ __device__ float minf(float a, float b) {
        return a <= b ? a : b;
    }

}

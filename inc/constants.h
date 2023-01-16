#pragma once

namespace pathtracer {

    // epsilon for equivalence of floating point numbers
    inline constexpr float epsilon = 0.00001;

    inline constexpr size_t height = 400;

    inline constexpr size_t width = 600;

    inline constexpr float pi = 3.14159265f;

    inline constexpr float two_pi = 6.283185307f;

    inline constexpr float one_over_pi = 0.31830988f;

    inline constexpr bool collision_buffer_limit_enable = true;

    inline constexpr int collision_buffer_limit = 512;

    inline constexpr bool intersection_buffer_limit_enable = true;

    inline constexpr int intersection_buffer_limit = 32;

    inline constexpr float mediump_flt_max = 65504.0;
    
}

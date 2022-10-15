#include <cmath>
#include "ray.cuh"
#include <iostream>
#include <math.h>

namespace pathtracer {

    __host__ __device__ ray::ray(const point& origin, const vector& direction): 
        o(origin), d(direction), d_inv({1.f / direction.x, 1.f / direction.y, 1.f / direction.z}) {}

    __host__  bool ray::check_bvh_node_intersection(bvh_node* b) {
        // https://tavianator.com/2011/ray_box.html#:~:text=The%20fastest%20method%20for%20performing,remains%2C%20it%20intersected%20the%20box
        // https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
        // https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525

        // float tx1 = (b->lower.x - o.x) * d_inv.x;
        // float tx2 = (b->upper.x - o.x) * d_inv.x;

        // float tmin = fminf(tx1, tx2);
        // float tmax = fmaxf(tx1, tx2);

        // float ty1 = (b->lower.y - o.y) * d_inv.y;
        // float ty2 = (b->upper.y - o.y) * d_inv.y;

        // tmin = fmaxf(tmin, fminf(ty1, ty2));
        // tmax = fminf(tmax, fmaxf(ty1, ty2));

        // float tz1 = (b->lower.z - o.z) * d_inv.z;
        // float tz2 = (b->upper.z - o.z) * d_inv.z;

        // tmin = fmaxf(tmin, fminf(tz1, tz2));
        // tmax = fminf(tmax, fmaxf(tz1, tz2));
        
        // std::cout << d_inv.y << std::endl;
        // std::cout << tx1 << " " << tx2 << std::endl;
        // std::cout << ty1 << " " << ty2 << std::endl;
        // std::cout << tz1 << " " << tz2 << std::endl;
        // std::cout << tmin << " " << tmax << std::endl;

        // return tmax >= fmaxf(0.f, tmin);

        // Special case when ray origin is within the bounding box,
        // intersection is guaranteed (even with a 0 vector for efficiency)
        if ((b->lower.x <= o.x) && (o.x <= b->upper.x) &&
            (b->lower.y <= o.y) && (o.y <= b->upper.y) &&
            (b->lower.z <= o.z) && (o.z <= b->upper.z)) return true;

        vec3 t0 = (b->lower - o) & d_inv;
        vec3 t1 = (b->upper - o) & d_inv;
        vec3 tmin{fminf(t0.x, t1.x),
                  fminf(t0.y, t1.y),
                  fminf(t0.z, t1.z)};

        vec3 tmax{fmaxf(t0.x, t1.x),
                  fmaxf(t0.y, t1.y),
                  fmaxf(t0.z, t1.z)};
        
        return (tmin.max_component() <= tmax.min_component()) && tmin.max_component() >= 0.f;
    }

}

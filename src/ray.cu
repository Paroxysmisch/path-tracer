#include <cmath>
#include "ray.cuh"

namespace pathtracer {

    __host__ __device__ ray::ray(const point& origin, const vector& direction): 
        o(origin), d(direction), d_inv({1.f / direction.x, 1.f / direction.y, 1.f / direction.z}) {}

    __host__ __device__ bool ray::check_bvh_node_intersection(bvh_node* b) {
        // https://tavianator.com/2011/ray_box.html#:~:text=The%20fastest%20method%20for%20performing,remains%2C%20it%20intersected%20the%20box
        // https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection

        double tx1 = (b->lower.x - o.x) * d_inv.x;
        double tx2 = (b->upper.x - o.x) * d_inv.x;

        double tmin = fminf(tx1, tx2);
        double tmax = fmaxf(tx1, tx2);

        double ty1 = (b->lower.y - o.y) * d_inv.y;
        double ty2 = (b->upper.y - o.y) * d_inv.y;

        tmin = fmaxf(tmin, fminf(ty1, ty2));
        tmax = fminf(tmax, fmaxf(ty1, ty2));

        double tz1 = (b->lower.z - o.z) * d_inv.z;
        double tz2 = (b->upper.z - o.z) * d_inv.z;

        tmin = fmaxf(tmin, fminf(tz1, tz2));
        tmax = fminf(tmax, fmaxf(tz1, tz2));

        return tmax >= tmin;
    }

}

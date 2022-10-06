#include <iostream>
#include "check_cuda_errors.h"
#include "util.cuh"

int main() {
    std::cout << "Hello World" << std::endl;

    pathtracer::vec3 vec{1,2,3};
    vec *= 2;
    vec.normalize();
    std::cout << vec.y << std::endl;
}
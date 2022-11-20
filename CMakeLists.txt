cmake_minimum_required(VERSION 3.20)

project(Path_Tracer LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_language(CUDA)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(Catch2 REQUIRED)

option(TINYEXR_USE_MINIZ "Use miniz" ON)

if(TINYEXR_USE_MINIZ)
    enable_language(C)
    add_library(miniz STATIC src/miniz.c)
    target_include_directories(miniz PUBLIC inc/)
    list(APPEND TINYEXR_EXT_LIBRARIES miniz)
endif()

set(path_tracer_sources
    main.cu
    check_cuda_errors.cu
    tinyexr.cc
    util.cu
    bvh.cu
    ray.cu
    shapes.cu
    scene.cu
    phong.cu
    world.cu
    camera.cu
    brdf.cu
    denoise.cu
)

list(TRANSFORM path_tracer_sources PREPEND "src/")

add_executable(path_tracer ${path_tracer_sources})

set_property(TARGET path_tracer PROPERTY CXX_STANDARD 17)

target_include_directories(path_tracer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/inc")

target_link_libraries(path_tracer PUBLIC ${TINYEXR_EXT_LIBRARIES})

set(test_sources
    main.cu
    float.cu
    vector.cu
    canvas.cu
    matrix.cu
    quaternion.cu
    bvh.cu
    ray.cu
    shapes.cu
    scene.cu
    phong.cu
    world.cu
    camera.cu
    brdf.cu
    mesh.cu
)

list(TRANSFORM test_sources PREPEND "test/")

list(REMOVE_ITEM path_tracer_sources "src/main.cu")

add_executable(test ${test_sources} ${path_tracer_sources})

set_property(TARGET test PROPERTY CXX_STANDARD 17)

target_include_directories(test PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/inc")

target_link_libraries(test PUBLIC Catch2::Catch2 ${TINYEXR_EXT_LIBRARIES})

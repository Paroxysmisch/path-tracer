cmake_minimum_required(VERSION 3.20)

project(Path_Tracer LANGUAGES CXX CUDA)

enable_language(CUDA)

find_package(Catch2 REQUIRED)

set(path_tracer_sources
    main.cu
    check_cuda_errors.cu
)

list(TRANSFORM path_tracer_sources PREPEND "src/")

add_executable(path_tracer ${path_tracer_sources})

set_property(TARGET path_tracer PROPERTY CXX_STANDARD 17)

target_include_directories(path_tracer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/inc")

set(test_sources
    main.cu
)

list(TRANSFORM test_sources PREPEND "test/")

add_executable(test ${test_sources})

set_property(TARGET test PROPERTY CXX_STANDARD 17)

target_link_libraries(test PUBLIC Catch2::Catch2)

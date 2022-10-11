#pragma once

#include <exception>

class path_tracer_exception : std::exception {};

class divide_by_zero_exception : path_tracer_exception {};
cmake_minimum_required(VERSION 3.22.1)

project(cpp_interpolator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)


set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(PYBIND11_FINDPYTHON ON)
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 2.10 REQUIRED CONFIG)
pybind11_add_module(cpp_interpolator
    b_spline.cpp
    interpolator_bindings.cpp
)

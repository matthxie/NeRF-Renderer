cmake_minimum_required(VERSION 3.15)
project(raytracer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python3 3.12.7 COMPONENTS Interpreter Development REQUIRED)

# Add pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6  # Adjust version as needed
)
FetchContent_MakeAvailable(pybind11)

# Add your source files
pybind11_add_module(raytracer 
    pybind11_bindings.cpp
)

# Include directories
target_include_directories(raytracer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
#pragma once
#include "Vector3.hpp"

struct Light {
    Vector3 position;
    double intensity;
    
    Light(const Vector3& pos, double i = 1.0) : position(pos), intensity(i) {}
};
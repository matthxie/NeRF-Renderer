#pragma once
#include <vector>
#include "Vector3.hpp"

struct Color {
    double r, g, b;
    Color(double r = 0, double g = 0, double b = 0) : r(r), g(g), b(b) {}
    
    Color operator*(double t) const {
        return Color(r * t, g * t, b * t);
    }
    
    Color operator+(const Color& c) const {
        return Color(r + c.r, g + c.g, b + c.b);
    }
};

struct Face {
    std::vector<int> vertexIndices;
    Vector3 normal;
    Color color;
};

struct Mesh {
    std::vector<Vector3> vertices;
    std::vector<Face> faces;
};
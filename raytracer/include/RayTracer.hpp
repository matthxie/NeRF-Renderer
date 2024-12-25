#pragma once
#include <vector>
#include "Vector3.hpp"
#include "Light.hpp"
#include "Mesh.hpp"

class RayTracer {
public:
    RayTracer(const Vector3& cameraPos) : cameraPosition(cameraPos) {}
    
    void addLight(const Light& light) {
        lights.push_back(light);
    }
    
    Color tracePoint(const Vector3& point, const Vector3& normal, const Color& surfaceColor) {
        Color finalColor(0, 0, 0);
        
        for (const Light& light : lights) {
            Vector3 lightDir = (light.position - point).normalize();
            double diffuse = std::max(0.0, normal.dot(lightDir));
            
            const double ambientIntensity = 0.1;
            Color ambientColor = surfaceColor * ambientIntensity;
            
            Color diffuseColor = surfaceColor * (diffuse * light.intensity);
            
            finalColor = finalColor + ambientColor + diffuseColor;
        }
        
        return finalColor;
    }
    
    Color getColorAtPoint(const Mesh& mesh, const Vector3& point) {
        for (const Face& face : mesh.faces) {
            return tracePoint(point, face.normal, face.color);
        }
        return Color(0, 0, 0);
    }

private:
    Vector3 cameraPosition;
    std::vector<Light> lights;
};
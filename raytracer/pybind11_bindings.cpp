#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "RayTracer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(raytracer, m) {
    py::class_<Vector3>(m, "Vector3")
        .def(py::init<double, double, double>())
        .def_readwrite("x", &Vector3::x)
        .def_readwrite("y", &Vector3::y)
        .def_readwrite("z", &Vector3::z);
        
    py::class_<Color>(m, "Color")
        .def(py::init<double, double, double>())
        .def_readwrite("r", &Color::r)
        .def_readwrite("g", &Color::g)
        .def_readwrite("b", &Color::b);
        
    py::class_<Light>(m, "Light")
        .def(py::init<Vector3, double>())
        .def_readwrite("position", &Light::position)
        .def_readwrite("intensity", &Light::intensity);
        
    py::class_<Face>(m, "Face")
        .def(py::init<>())
        .def_readwrite("vertexIndices", &Face::vertexIndices)
        .def_readwrite("normal", &Face::normal)
        .def_readwrite("color", &Face::color);
        
    py::class_<Mesh>(m, "Mesh")
        .def(py::init<>())
        .def_readwrite("vertices", &Mesh::vertices)
        .def_readwrite("faces", &Mesh::faces);
        
    py::class_<RayTracer>(m, "RayTracer")
        .def(py::init<Vector3>())
        .def("addLight", &RayTracer::addLight)
        .def("getColorAtPoint", &RayTracer::getColorAtPoint)
        .def("tracePoint", &RayTracer::tracePoint);
}
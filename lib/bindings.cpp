#include <config.hpp>
#include <core.hpp>
#include <iostream>
#include <process.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<std::array<double, 3>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::array<int, 3>>);

PYBIND11_MODULE(visacd, m)
{
    py::bind_vector<std::vector<std::array<double, 3>>>(
        m, "VecArray3d"); // 3D vector array
    py::bind_vector<std::vector<std::array<int, 3>>>(
        m, "VecArray3i"); // triangle array

    py::class_<neural_acd::Mesh>(m, "Mesh")
        .def_readwrite("vertices", &neural_acd::Mesh::vertices)
        .def_readwrite("triangles", &neural_acd::Mesh::triangles)
        .def(py::init<>());

    py::bind_vector<neural_acd::MeshList>(m, "MeshList");

    py::class_<neural_acd::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("return_parts", &neural_acd::Config::return_parts)
        .def_readwrite("score_mode", &neural_acd::Config::score_mode)
        .def_readwrite("support_surface_min_area",
                       &neural_acd::Config::support_surface_min_area)
        .def_readwrite("use_support_surfaces",
                       &neural_acd::Config::use_support_surfaces)
        .def_readwrite("support_surface_k",
                       &neural_acd::Config::support_surface_k)
        .def_readwrite("use_merging", &neural_acd::Config::use_merging);

    m.def("make_vecarray3i", [](py::array_t<int> input)
          {
    auto buf = input.request();
    std::vector<std::array<int, 3>> result;

    int X = buf.shape[0];
    int *ptr = (int *)buf.ptr;

    for (size_t idx = 0; idx < X; idx++) {
      std::array<int, 3> arr;
      arr[0] = ptr[idx * 3];
      arr[1] = ptr[idx * 3 + 1];
      arr[2] = ptr[idx * 3 + 2];
      result.push_back(arr);
    }

    return result; });

    m.def("set_seed", &neural_acd::set_seed, py::arg("seed"));

    m.attr("config") =
        py::cast(&neural_acd::config, py::return_value_policy::reference);

    py::class_<neural_acd::ProcessResult>(m, "ProcessResult")
        .def_readwrite("parts", &neural_acd::ProcessResult::parts)
        .def_readwrite("concavity", &neural_acd::ProcessResult::concavity)
        .def_readwrite("num_parts", &neural_acd::ProcessResult::num_parts);

    m.def("process", &neural_acd::process, py::arg("mesh"), py::arg("concavity"),
          py::arg("num_parts"));
}

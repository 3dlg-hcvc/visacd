#include <clip.hpp>
#include <config.hpp>
#include <core.hpp>
#include <cost.hpp>
#include <iostream>
#include <preprocess.hpp>
#include <process.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <support_surface.hpp>
namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<std::array<double, 3>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::array<double, 4>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::array<int, 3>>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<std::pair<unsigned int, unsigned int>>);

PYBIND11_MODULE(lib_neural_acd, m)
{
    py::bind_vector<std::vector<std::array<double, 3>>>(
        m, "VecArray3d"); // 3D vector array
    py::bind_vector<std::vector<std::array<int, 3>>>(
        m, "VecArray3i"); // triangle array
    py::bind_vector<std::vector<std::array<double, 4>>>(
        m, "VecArray4d");                                 // plane array
    py::bind_vector<std::vector<double>>(m, "VecDouble"); // cut verts
    py::bind_vector<std::vector<int>>(m, "VecInt");       // sample triangle ids
    py::bind_vector<std::vector<std::pair<unsigned int, unsigned int>>>(
        m, "VecPairUInt"); // intersecting edges

    py::class_<neural_acd::Surface>(m, "Surface")
        .def(py::init<>())
        .def_readwrite("triangle_ids", &neural_acd::Surface::triangle_ids)
        .def_readwrite("area", &neural_acd::Surface::area);

    py::class_<neural_acd::Mesh>(m, "Mesh")
        .def_readwrite("vertices", &neural_acd::Mesh::vertices)
        .def_readwrite("triangles", &neural_acd::Mesh::triangles)
        .def_readwrite("cut_verts", &neural_acd::Mesh::cut_verts)
        .def_readwrite("intersecting_edges",
                       &neural_acd::Mesh::intersecting_edges)
        .def(py::init<>())
        .def("extract_point_set",
             static_cast<void (neural_acd::Mesh::*)(
                 std::vector<std::array<double, 3>> &, std::vector<int> &,
                 size_t)>(&neural_acd::Mesh::extract_point_set),
             py::arg("samples"), py::arg("sample_tri_ids"),
             py::arg("resolution") = 10000);

    py::bind_vector<neural_acd::MeshList>(m, "MeshList");

    py::class_<neural_acd::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("pcd_res", &neural_acd::Config::pcd_res)
        .def_readwrite("remesh_res", &neural_acd::Config::remesh_res)
        .def_readwrite("remesh_threshold", &neural_acd::Config::remesh_threshold)
        .def_readwrite("cost_rv_k", &neural_acd::Config::cost_rv_k)
        .def_readwrite("return_parts", &neural_acd::Config::return_parts)
        .def_readwrite("separate_disjoint",
                       &neural_acd::Config::separate_disjoint)
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

    m.attr("config") =
        py::cast(&neural_acd::config, py::return_value_policy::reference);

    m.def("set_seed", &neural_acd::set_seed, py::arg("seed"));
    m.def("multiclip", &neural_acd::multiclip, py::arg("mesh"),
          py::arg("planes"));
    m.def("process", &neural_acd::process, py::arg("mesh"), py::arg("concavity"),
          py::arg("num_parts"), py::arg("stats_file"));
    m.def("preprocess", &neural_acd::manifold_preprocess, py::arg("mesh"),
          py::arg("scale"), py::arg("level_set"));
    m.def("get_eval_concavity", &neural_acd::compute_final_concavity, py::arg("parts"),
          py::arg("hulls"));
    m.def("get_support_surfaces", &neural_acd::get_support_surfaces,
          py::arg("mesh"));
    m.def("preprocess_mesh", &neural_acd::preprocess_mesh, py::arg("mesh"));
}

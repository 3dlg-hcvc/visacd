#pragma once

#include <core.hpp>
#include <limits>
#include <unordered_map>
#include <map>

namespace neural_acd {

MeshList clip(const Mesh mesh, Plane plane, int *&pos_proj, int *&neg_proj);
MeshList multiclip(const Mesh mesh, const std::vector<Plane> &planes);

inline bool same_point_detect(Vec3D p0, Vec3D p1, float eps = 1e-5) {
  double dx, dy, dz;
  dx = fabs(p0[0] - p1[0]);
  dy = fabs(p0[1] - p1[1]);
  dz = fabs(p0[2] - p1[2]);
  if (dx < eps && dy < eps && dz < eps)
    return true;
  return false;
}

inline void add_point(std::map<int, int> &vertex_map,
                      std::vector<Vec3D> &border, Vec3D pt, int id, int &idx) {
  if (vertex_map.find(id) == vertex_map.end()) {
    int flag = -1;
    for (int i = 0; i < (int)border.size(); i++) {
      if ((fabs(border[i][0] - pt[0])) < 1e-4 &&
          (fabs(border[i][1] - pt[1])) < 1e-4 &&
          (fabs(border[i][2] - pt[2])) < 1e-4) {
        flag = i;
        break;
      }
    }
    if (flag == -1) {
      vertex_map[id] = idx;
      border.push_back(pt);
      idx++;
    } else
      vertex_map[id] = flag;
  }
}

inline void add_edge_point(std::map<std::pair<int, int>, int> &edge_map,
                           std::vector<Vec3D> &border, Vec3D pt, int id1,
                           int id2, int &idx) {
  std::pair<int, int> edge1 = std::make_pair(id1, id2);
  std::pair<int, int> edge2 = std::make_pair(id2, id1);
  if (edge_map.find(edge1) == edge_map.end() &&
      edge_map.find(edge2) == edge_map.end()) {
    int flag = -1;
    for (int i = 0; i < (int)border.size(); i++) {
      if ((fabs(border[i][0] - pt[0])) < 1e-4 &&
          (fabs(border[i][1] - pt[1])) < 1e-4 &&
          (fabs(border[i][2] - pt[2])) < 1e-4) {
        flag = i;
        break;
      }
    }
    if (flag == -1) {
      edge_map[edge1] = idx;
      edge_map[edge2] = idx;
      border.push_back(pt);
      idx++;
    } else {
      edge_map[edge1] = flag;
      edge_map[edge2] = flag;
    }
  }
}

inline bool face_overlap(std::map<int, bool> overlap_map,
                         std::array<int, 3> triangle) {
  int idx0 = triangle[0], idx1 = triangle[1], idx2 = triangle[2];
  if (overlap_map.find(idx0) == overlap_map.end() &&
      overlap_map.find(idx1) == overlap_map.end() &&
      overlap_map.find(idx2) == overlap_map.end())
    return false;
  return true;
}

} // namespace neural_acd
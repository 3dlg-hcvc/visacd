//Based on https://github.com/SarahWeiii/CoACD/blob/main/src/clip.cpp. Credit: Xinyue Wei
#include "clip.hpp"
#include "core.hpp"
#include <CDT.h>
#include <CDTUtils.h>
#include <cost.hpp>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <preprocess.hpp>
#include <set>
#include <string>

using namespace std;

namespace neural_acd {

bool CreatePlaneRotationMatrix(vector<Vec3D> &border,
                               vector<pair<int, int>> border_edges, Vec3D &T,
                               double R[3][3], Plane &plane) {
  int idx0 = 0;
  int idx1;
  int idx2;
  bool flag = 0;

  for (int i = 1; i < (int)border.size(); i++) {
    double dist = sqrt(pow(border[idx0][0] - border[i][0], 2) +
                       pow(border[idx0][1] - border[i][1], 2) +
                       pow(border[idx0][2] - border[i][2], 2));
    if (dist > 0.01) {
      flag = 1;
      idx1 = i;
      break;
    }
  }
  if (!flag)
    return false;
  flag = 0;

  for (int i = 2; i < (int)border.size(); i++) {
    if (i == idx1)
      continue;
    Vec3D p0 = border[idx0];
    Vec3D p1 = border[idx1];
    Vec3D p2 = border[i];
    Vec3D AB, BC;
    AB[0] = p1[0] - p0[0];
    AB[1] = p1[1] - p0[1];
    AB[2] = p1[2] - p0[2];
    BC[0] = p2[0] - p1[0];
    BC[1] = p2[1] - p1[1];
    BC[2] = p2[2] - p1[2];

    double dot_product = AB[0] * BC[0] + AB[1] * BC[1] + AB[2] * BC[2];
    double res =
        dot_product / (sqrt(pow(AB[0], 2) + pow(AB[1], 2) + pow(AB[2], 2)) *
                       sqrt(pow(BC[0], 2) + pow(BC[1], 2) + pow(BC[2], 2)));
    if (fabs(fabs(res) - 1) > 1e-6 &&
        fabs(res) < INF) // AB not \\ BC, dot product != 1
    {
      flag = 1;
      idx2 = i;
      break;
    }
  }
  if (!flag)
    return false;

  double t0, t1, t2;
  Vec3D p0 = border[idx0], p1 = border[idx1], p2 = border[idx2];
  Vec3D normal = calc_face_normal(p0, p1, p2);

  double dot =
      normal[0]*plane.a +
      normal[1]*plane.b +
      normal[2]*plane.c;

  if (dot > 0.0) {
      std::swap(p0, p2);
  }
  plane.pFlag = true;
  plane.p0 = p2;
  plane.p1 = p1;
  plane.p2 = p0;

  // translate to origin
  T = p0;

  // rotation matrix
  double eps = 0.0;
  R[0][0] =
      (p0[0] - p1[0]) / (sqrt(pow(p0[0] - p1[0], 2) + pow(p0[1] - p1[1], 2) +
                              pow(p0[2] - p1[2], 2)) +
                         eps);
  R[0][1] =
      (p0[1] - p1[1]) / (sqrt(pow(p0[0] - p1[0], 2) + pow(p0[1] - p1[1], 2) +
                              pow(p0[2] - p1[2], 2)) +
                         eps);
  R[0][2] =
      (p0[2] - p1[2]) / (sqrt(pow(p0[0] - p1[0], 2) + pow(p0[1] - p1[1], 2) +
                              pow(p0[2] - p1[2], 2)) +
                         eps);

  t0 = (p2[2] - p0[2]) * R[0][1] - (p2[1] - p0[1]) * R[0][2];
  t1 = (p2[0] - p0[0]) * R[0][2] - (p2[2] - p0[2]) * R[0][0];
  t2 = (p2[1] - p0[1]) * R[0][0] - (p2[0] - p0[0]) * R[0][1];
  R[2][0] = t0 / (sqrt(pow(t0, 2) + pow(t1, 2) + pow(t2, 2)) + eps);
  R[2][1] = t1 / (sqrt(pow(t0, 2) + pow(t1, 2) + pow(t2, 2)) + eps);
  R[2][2] = t2 / (sqrt(pow(t0, 2) + pow(t1, 2) + pow(t2, 2)) + eps);

  t0 = R[2][2] * R[0][1] - R[2][1] * R[0][2];
  t1 = R[2][0] * R[0][2] - R[2][2] * R[0][0];
  t2 = R[2][1] * R[0][0] - R[2][0] * R[0][1];
  R[1][0] = t0 / (sqrt(pow(t0, 2) + pow(t1, 2) + pow(t2, 2)) + eps);
  R[1][1] = t1 / (sqrt(pow(t0, 2) + pow(t1, 2) + pow(t2, 2)) + eps);
  R[1][2] = t2 / (sqrt(pow(t0, 2) + pow(t1, 2) + pow(t2, 2)) + eps);

  return true;
}

short Triangulation(vector<Vec3D> &border, vector<pair<int, int>> border_edges,
                    vector<array<int, 3>> &border_triangles, Plane &plane) {
  double R[3][3];
  Vec3D T;

  bool flag = CreatePlaneRotationMatrix(border, border_edges, T, R, plane);
  if (!flag)
    return 1;

  vector<array<double, 2>> vertices, nodes;

  double x_min = INF, x_max = -INF, y_min = INF, y_max = -INF;
  for (int i = 0; i < (int)border.size(); i++) {
    double x, y, z, px, py;
    x = border[i][0] - T[0];
    y = border[i][1] - T[1];
    z = border[i][2] - T[2];

    px = R[0][0] * x + R[0][1] * y + R[0][2] * z;
    py = R[1][0] * x + R[1][1] * y + R[1][2] * z;

    vertices.push_back({px, py});

    x_min = min(x_min, px);
    x_max = max(x_max, px);
    y_min = min(y_min, py);
    y_max = max(y_max, py);
  }

  int borderN = (int)vertices.size();

  CDT::Triangulation<double> cdt(CDT::VertexInsertionOrder::AsProvided);
  try {
    cdt.insertVertices(
        vertices.begin(), vertices.end(),
        [](const array<double, 2> &p) { return p[0]; },
        [](const array<double, 2> &p) { return p[1]; });
    cdt.insertEdges(
        border_edges.begin(), border_edges.end(),
        [](const pair<int, int> &p) { return (int)p.first - 1; },
        [](const pair<int, int> &p) { return (int)p.second - 1; });
    cdt.eraseOuterTrianglesAndHoles();
  } catch (const runtime_error &e) {
    cout << e.what() << endl;
    return 2;
  }

  border_triangles.clear();
  for (int i = 0; i < (int)cdt.triangles.size(); i++) {
    border_triangles.push_back({(int)cdt.triangles[i].vertices[0],
                                (int)cdt.triangles[i].vertices[1],
                                (int)cdt.triangles[i].vertices[2]});
  }

  // border.clear();

  for (int i = borderN; i < cdt.vertices.size(); i++) {
    double x, y, z;
    CDT::V2d<double> vertex = cdt.vertices[i];
    x = R[0][0] * vertex.x + R[1][0] * vertex.y + T[0];
    y = R[0][1] * vertex.x + R[1][1] * vertex.y + T[1];
    z = R[0][2] * vertex.x + R[1][2] * vertex.y + T[2];
    border.push_back({x, y, z});
  }

  return 0;
}

MeshList clip(const Mesh mesh, Plane plane, int *&pos_proj, int *&neg_proj) {
  Mesh t = mesh;
  Mesh pos, neg;
  vector<Vec3D> border;
  vector<Vec3D> overlap;
  vector<array<int, 3>> border_triangles, final_triangles;
  vector<pair<int, int>> border_edges;
  map<int, int> border_map;
  vector<Vec3D> final_border;
  vector<bool> pos_map, neg_map;

  const int N = (int)mesh.vertices.size();
  int idx = 0;
  pos_map.resize(N, false);
  neg_map.resize(N, false);

  map<pair<int, int>, int> edge_map;
  map<int, int> vertex_map;

  for (int i = 0; i < (int)mesh.triangles.size(); i++) {
    int id0, id1, id2;
    id0 = mesh.triangles[i][0];
    id1 = mesh.triangles[i][1];
    id2 = mesh.triangles[i][2];
    Vec3D p0, p1, p2;
    p0 = mesh.vertices[id0];
    p1 = mesh.vertices[id1];
    p2 = mesh.vertices[id2];
    short s0 = plane.side(p0), s1 = plane.side(p1), s2 = plane.side(p2);

    short sum = s0 + s1 + s2;

    if (s0 == 0 && s1 == 0 && s2 == 0) {
      s0 = s1 = s2 = plane.cut_side(p0, p1, p2, plane);
      sum = s0 + s1 + s2;
      overlap.push_back(p0);
      overlap.push_back(p1);
      overlap.push_back(p2);
    }

    if (sum == 3 || sum == 2 ||
        (sum == 1 &&
         ((s0 == 1 && s1 == 0 && s2 == 0) || (s0 == 0 && s1 == 1 && s2 == 0) ||
          (s0 == 0 && s1 == 0 && s2 == 1)))) // pos side
    {

      // the plane cross the triangle edge
      if (sum == 1) {
        if (s0 == 1 && s1 == 0 && s2 == 0) {
          add_point(vertex_map, border, p1, id1, idx);
          add_point(vertex_map, border, p2, id2, idx);
          if (vertex_map[id1] != vertex_map[id2]) {
            border_edges.push_back(
                std::pair<int, int>(vertex_map[id1] + 1, vertex_map[id2] + 1));
            pos.triangles.push_back(
                {id0, -1 * vertex_map[id1] - 1, -1 * vertex_map[id2] - 1});
            pos_map[id0] = true;
          }

        } else if (s0 == 0 && s1 == 1 && s2 == 0) {
          add_point(vertex_map, border, p2, id2, idx);
          add_point(vertex_map, border, p0, id0, idx);
          if (vertex_map[id2] != vertex_map[id0]) {
            border_edges.push_back(
                std::pair<int, int>(vertex_map[id2] + 1, vertex_map[id0] + 1));
            pos.triangles.push_back(
                {id1, -1 * vertex_map[id2] - 1, -1 * vertex_map[id0] - 1});
            pos_map[id1] = true;
          }
        } else if (s0 == 0 && s1 == 0 && s2 == 1) {
          add_point(vertex_map, border, p0, id0, idx);
          add_point(vertex_map, border, p1, id1, idx);
          if (vertex_map[id0] != vertex_map[id1]) {
            border_edges.push_back(
                std::pair<int, int>(vertex_map[id0] + 1, vertex_map[id1] + 1));
            pos.triangles.push_back(
                {id2, -1 * vertex_map[id0] - 1, -1 * vertex_map[id1] - 1});
            pos_map[id2] = true;
          }
        }
      } else if (sum == 2) {
        if (s0 == 1 && s1 == 1 && s2 == 0) {
          add_point(vertex_map, border, p2, id2, idx);
          pos.triangles.push_back({id0, id1, -1 * vertex_map[id2] - 1});
          pos_map[id0] = true;
          pos_map[id1] = true;
        } else if (s0 == 0 && s1 == 1 && s2 == 1) {
          add_point(vertex_map, border, p0, id0, idx);
          pos.triangles.push_back({id1, id2, -1 * vertex_map[id0] - 1});
          pos_map[id1] = true;
          pos_map[id2] = true;
        } else if (s0 == 1 && s1 == 0 && s2 == 1) {
          add_point(vertex_map, border, p1, id1, idx);
          pos.triangles.push_back({id0, -1 * vertex_map[id1] - 1, id2});
          pos_map[id0] = true;
          pos_map[id2] = true;
        }
      } else {
        pos.triangles.push_back(mesh.triangles[i]);
        pos_map[id0] = true;
        pos_map[id1] = true;
        pos_map[id2] = true;
      }

    } else if (sum == -3 || sum == -2 ||
               (sum == -1 && ((s0 == -1 && s1 == 0 && s2 == 0) ||
                              (s0 == 0 && s1 == -1 && s2 == 0) ||
                              (s0 == 0 && s1 == 0 && s2 == -1)))) // neg side
    {

      // the plane cross the triangle edge
      if (sum == -1) {
        if (s0 == -1 && s1 == 0 && s2 == 0) {
          add_point(vertex_map, border, p2, id2, idx);
          add_point(vertex_map, border, p1, id1, idx);
          if (vertex_map[id2] != vertex_map[id1]) {
            border_edges.push_back(
                std::pair<int, int>(vertex_map[id2] + 1, vertex_map[id1] + 1));
            neg.triangles.push_back(
                {id0,-1 * vertex_map[id1] - 1, -1 * vertex_map[id2] - 1});
            neg_map[id0] = true;
          }
        } else if (s0 == 0 && s1 == -1 && s2 == 0) {
          add_point(vertex_map, border, p0, id0, idx);
          add_point(vertex_map, border, p2, id2, idx);
          if (vertex_map[id0] != vertex_map[id2]) {
            border_edges.push_back(
                std::pair<int, int>(vertex_map[id0] + 1, vertex_map[id2] + 1));
            neg.triangles.push_back(
                {id1, -1 * vertex_map[id2] - 1, -1 * vertex_map[id0] - 1});
            neg_map[id1] = true;
          }
        } else if (s0 == 0 && s1 == 0 && s2 == -1) {
          add_point(vertex_map, border, p1, id1, idx);
          add_point(vertex_map, border, p0, id0, idx);
          if (vertex_map[id1] != vertex_map[id0]) {
            border_edges.push_back(
                std::pair<int, int>(vertex_map[id1] + 1, vertex_map[id0] + 1));
            neg.triangles.push_back(
                {id2, -1 * vertex_map[id0] - 1, -1 * vertex_map[id1] - 1});
            neg_map[id2] = true;
          }
        }
      } else if (sum == -2) {
        if (s0 == -1 && s1 == -1 && s2 == 0) {
          add_point(vertex_map, border, p2, id2, idx);

          neg.triangles.push_back({id0, id1, -1 * vertex_map[id2] - 1});
          neg_map[id0] = true;
          neg_map[id1] = true;

        } else if (s0 == 0 && s1 == -1 && s2 == -1) {
          add_point(vertex_map, border, p0, id0, idx);

          neg.triangles.push_back({id1, id2, -1 * vertex_map[id0] - 1});
          neg_map[id1] = true;
          neg_map[id2] = true;

        } else if (s0 == -1 && s1 == 0 && s2 == -1) {
          add_point(vertex_map, border, p1, id1, idx);

          neg.triangles.push_back({id0, -1 * vertex_map[id1] - 1, id2});
          neg_map[id0] = true;
          neg_map[id2] = true;
        }
      } else {
        neg.triangles.push_back(mesh.triangles[i]);
        neg_map[id0] = true;
        neg_map[id1] = true;
        neg_map[id2] = true;
      }

    } else // different side
    {
      bool f0, f1, f2;
      Vec3D pi0, pi1, pi2;
      f0 = plane.intersect_segment(p0, p1, pi0);
      f1 = plane.intersect_segment(p1, p2, pi1);
      f2 = plane.intersect_segment(p2, p0, pi2);

      if (f0 && f1 && !f2) {
        // record the vertices
        // f0
        add_edge_point(edge_map, border, pi0, id0, id1, idx);
        // f1
        add_edge_point(edge_map, border, pi1, id1, id2, idx);

        // record the edges
        int f0_idx = edge_map[std::pair<int, int>(id0, id1)];
        int f1_idx = edge_map[std::pair<int, int>(id1, id2)];
        if (s1 == 1) {
          if (f1_idx != f0_idx) {
            border_edges.push_back(
                std::pair<int, int>(f1_idx + 1, f0_idx + 1)); // border
            pos_map[id1] = true;
            neg_map[id0] = true;
            neg_map[id2] = true;
            pos.triangles.push_back(
                {id1, -1 * f1_idx - 1,
                 -1 * f0_idx - 1}); // make sure it is not zero
            neg.triangles.push_back({id0, -1 * f0_idx - 1, -1 * f1_idx - 1});
            neg.triangles.push_back({-1 * f1_idx - 1, id2, id0});
          } else {
            neg_map[id0] = true;
            neg_map[id2] = true;
            neg.triangles.push_back({-1 * f1_idx - 1, id2, id0});
          }
        } else {
          if (f0_idx != f1_idx) {
            border_edges.push_back(
                std::pair<int, int>(f0_idx + 1, f1_idx + 1)); // border
            neg_map[id1] = true;
            pos_map[id0] = true;
            pos_map[id2] = true;
            neg.triangles.push_back({id1, -1 * f1_idx - 1, -1 * f0_idx - 1});
            pos.triangles.push_back({id0, -1 * f0_idx - 1, -1 * f1_idx - 1});
            pos.triangles.push_back({-1 * f1_idx - 1, id2, id0});
          } else {
            pos_map[id0] = true;
            pos_map[id2] = true;
            pos.triangles.push_back({-1 * f1_idx - 1, id2, id0});
          }
        }
      } else if (f1 && f2 && !f0) {
        // f1
        add_edge_point(edge_map, border, pi1, id1, id2, idx);
        // f2
        add_edge_point(edge_map, border, pi2, id2, id0, idx);

        // record the edges
        int f1_idx = edge_map[std::pair<int, int>(id1, id2)];
        int f2_idx = edge_map[std::pair<int, int>(id2, id0)];
        if (s2 == 1) {
          if (f2_idx != f1_idx) {
            border_edges.push_back(std::pair<int, int>(f2_idx + 1, f1_idx + 1));
            pos_map[id2] = true;
            neg_map[id0] = true;
            neg_map[id1] = true;
            pos.triangles.push_back({id2, -1 * f2_idx - 1, -1 * f1_idx - 1});
            neg.triangles.push_back({id0, -1 * f1_idx - 1, -1 * f2_idx - 1});
            neg.triangles.push_back({-1 * f1_idx - 1, id0, id1});
          } else {
            neg_map[id0] = true;
            neg_map[id1] = true;
            neg.triangles.push_back({-1 * f1_idx - 1, id0, id1});
          }
        } else {
          if (f1_idx != f2_idx) {
            border_edges.push_back(std::pair<int, int>(f1_idx + 1, f2_idx + 1));
            neg_map[id2] = true;
            pos_map[id0] = true;
            pos_map[id1] = true;
            neg.triangles.push_back({id2, -1 * f2_idx - 1, -1 * f1_idx - 1});
            pos.triangles.push_back({id0, -1 * f1_idx - 1, -1 * f2_idx - 1});
            pos.triangles.push_back({-1 * f1_idx - 1, id0, id1});
          } else {
            pos_map[id0] = true;
            pos_map[id1] = true;
            pos.triangles.push_back({-1 * f1_idx - 1, id0, id1});
          }
        }
      } else if (f2 && f0 && !f1) {
        // f2
        add_edge_point(edge_map, border, pi2, id2, id0, idx);
        // f0
        add_edge_point(edge_map, border, pi0, id0, id1, idx);

        int f0_idx = edge_map[std::pair<int, int>(id0, id1)];
        int f2_idx = edge_map[std::pair<int, int>(id2, id0)];
        if (s0 == 1) {
          if (f0_idx != f2_idx) {
            border_edges.push_back(std::pair<int, int>(f0_idx + 1, f2_idx + 1));
            pos_map[id0] = true;
            neg_map[id1] = true;
            neg_map[id2] = true;
            pos.triangles.push_back({id0, -1 * f0_idx - 1, -1 * f2_idx - 1});
            neg.triangles.push_back({id1, -1 * f2_idx - 1, -1 * f0_idx - 1});
            neg.triangles.push_back({-1 * f2_idx - 1, id1, id2});
          } else {
            neg_map[id1] = true;
            neg_map[id2] = true;
            neg.triangles.push_back({-1 * f2_idx - 1, id1, id2});
          }
        } else {
          if (f2_idx != f0_idx) {
            border_edges.push_back(std::pair<int, int>(f2_idx + 1, f0_idx + 1));
            neg_map[id0] = true;
            pos_map[id1] = true;
            pos_map[id2] = true;
            neg.triangles.push_back({id0, -1 * f0_idx - 1, -1 * f2_idx - 1});
            pos.triangles.push_back({id1, -1 * f2_idx - 1, -1 * f0_idx - 1});
            pos.triangles.push_back({-1 * f2_idx - 1, id1, id2});
          } else {
            pos_map[id1] = true;
            pos_map[id2] = true;
            pos.triangles.push_back({-1 * f2_idx - 1, id1, id2});
          }
        }
      } else if (f0 && f1 && f2) {
        if (s0 == 0 || (s0 != 0 && s1 != 0 && s2 != 0 &&
                        same_point_detect(pi0, pi2))) // intersect at p0
        {
          // f2 = f0 = p0
          add_point(vertex_map, border, p0, id0, idx);
          edge_map[std::pair<int, int>(id0, id1)] = vertex_map[id0];
          edge_map[std::pair<int, int>(id1, id0)] = vertex_map[id0];
          edge_map[std::pair<int, int>(id2, id0)] = vertex_map[id0];
          edge_map[std::pair<int, int>(id0, id2)] = vertex_map[id0];

          // f1
          add_edge_point(edge_map, border, pi1, id1, id2, idx);
          int f1_idx = edge_map[std::pair<int, int>(id1, id2)];
          int f0_idx = vertex_map[id0];
          if (s1 == 1) {
            if (f1_idx != f0_idx) {
              border_edges.push_back(
                  std::pair<int, int>(f1_idx + 1, f0_idx + 1));
              pos_map[id1] = true;
              neg_map[id2] = true;
              pos.triangles.push_back({id1, -1 * f1_idx - 1, -1 * f0_idx - 1});
              neg.triangles.push_back({id2, -1 * f0_idx - 1, -1 * f1_idx - 1});
            }
          } else {
            if (f0_idx != f1_idx) {
              border_edges.push_back(
                  std::pair<int, int>(f0_idx + 1, f1_idx + 1));
              neg_map[id1] = true;
              pos_map[id2] = true;
              neg.triangles.push_back({id1, -1 * f1_idx - 1, -1 * f0_idx - 1});
              pos.triangles.push_back({id2, -1 * f0_idx - 1, -1 * f1_idx - 1});
            }
          }
        } else if (s1 == 0 || (s0 != 0 && s1 != 0 && s2 != 0 &&
                               same_point_detect(pi0, pi1))) // intersect at p1
        {
          // f0 = f1 = p1
          add_point(vertex_map, border, p1, id1, idx);
          edge_map[std::pair<int, int>(id0, id1)] = vertex_map[id1];
          edge_map[std::pair<int, int>(id1, id0)] = vertex_map[id1];
          edge_map[std::pair<int, int>(id1, id2)] = vertex_map[id1];
          edge_map[std::pair<int, int>(id2, id1)] = vertex_map[id1];

          // f2
          add_edge_point(edge_map, border, pi2, id2, id0, idx);
          int f1_idx = vertex_map[id1];
          int f2_idx = edge_map[std::pair<int, int>(id2, id0)];
          if (s0 == 1) {
            if (f1_idx != f2_idx) {
              border_edges.push_back(
                  std::pair<int, int>(f1_idx + 1, f2_idx + 1));
              pos_map[id0] = true;
              neg_map[id2] = true;
              pos.triangles.push_back({id0, -1 * f1_idx - 1, -1 * f2_idx - 1});
              neg.triangles.push_back({id2, -1 * f2_idx - 1, -1 * f1_idx - 1});
            }
          } else {
            if (f2_idx != f1_idx) {
              border_edges.push_back(
                  std::pair<int, int>(f2_idx + 1, f1_idx + 1));
              neg_map[id0] = true;
              pos_map[id2] = true;
              neg.triangles.push_back({id0, -1 * f1_idx - 1, -1 * f2_idx - 1});
              pos.triangles.push_back({id2, -1 * f2_idx - 1, -1 * f1_idx - 1});
            }
          }
        } else if (s2 == 0 || (s0 != 0 && s1 != 0 && s2 != 0 &&
                               same_point_detect(pi1, pi2))) // intersect at p2
        {
          // f1 = f2 = p2
          add_point(vertex_map, border, p2, id2, idx);
          edge_map[std::pair<int, int>(id1, id2)] = vertex_map[id2];
          edge_map[std::pair<int, int>(id2, id1)] = vertex_map[id2];
          edge_map[std::pair<int, int>(id2, id0)] = vertex_map[id2];
          edge_map[std::pair<int, int>(id0, id2)] = vertex_map[id2];

          // f0
          add_edge_point(edge_map, border, pi0, id0, id1, idx);
          int f0_idx = edge_map[std::pair<int, int>(id0, id1)];
          int f1_idx = vertex_map[id2];
          if (s0 == 1) {
            if (f0_idx != f1_idx) {
              border_edges.push_back(
                  std::pair<int, int>(f0_idx + 1, f1_idx + 1));
              pos_map[id0] = true;
              neg_map[id1] = true;
              pos.triangles.push_back({id0, -1 * f0_idx - 1, -1 * f1_idx - 1});
              neg.triangles.push_back({id1, -1 * f1_idx - 1, -1 * f0_idx - 1});
            }
          } else {
            if (f1_idx != f0_idx) {
              border_edges.push_back(
                  std::pair<int, int>(f1_idx + 1, f0_idx + 1));
              neg_map[id0] = true;
              pos_map[id1] = true;
              neg.triangles.push_back({id0, -1 * f0_idx - 1, -1 * f1_idx - 1});
              pos.triangles.push_back({id1, -1 * f1_idx - 1, -1 * f0_idx - 1});
            }
          }
        }
      }
      // cout << "added pos triangle: "
      //      << pos.triangles[pos.triangles.size() - 1][0] << " "
      //      << pos.triangles[pos.triangles.size() - 1][1] << " "
      //      << pos.triangles[pos.triangles.size() - 1][2] << endl;
      // cout << "added neg triangle: "
      //      << neg.triangles[neg.triangles.size() - 1][0] << " "
      //      << neg.triangles[neg.triangles.size() - 1][1] << " "
      //      << neg.triangles[neg.triangles.size() - 1][2] << endl;
    }
  }

  std::set<std::pair<int, int>> unique_edges;
  std::vector<std::pair<int, int>> deduped_edges;

  for (auto &e : border_edges) {
    // normalize order to treat (a,b) and (b,a) as the same
    auto norm = std::minmax(e.first, e.second);
    if (unique_edges.insert(norm).second)
      deduped_edges.push_back(e);
  }

  border_edges.swap(deduped_edges);

  if (border.size() > 2) {
    int oriN = (int)border.size();
    short flag = Triangulation(border, border_edges, border_triangles, plane);
    final_border = border;
    final_triangles = border_triangles;
  } else {
    final_border = border; // remember to fill final_border with border!
  }

  // original vertices in two parts
  double pos_x_min = INF, pos_x_max = -INF, pos_y_min = INF, pos_y_max = -INF,
         pos_z_min = INF, pos_z_max = -INF;
  double neg_x_min = INF, neg_x_max = -INF, neg_y_min = INF, neg_y_max = -INF,
         neg_z_min = INF, neg_z_max = -INF;

  int pos_idx = 0, neg_idx = 0;
  pos_proj = new int[N]();
  neg_proj = new int[N]();
  for (int i = 0; i < N; i++) {
    if (pos_map[i] == true) {
      pos.vertices.push_back(mesh.vertices[i]);
      pos_proj[i] = ++pos_idx; // 0 means not exist, so all plus 1

      pos_x_min = min(pos_x_min, mesh.vertices[i][0]);
      pos_x_max = max(pos_x_max, mesh.vertices[i][0]);
      pos_y_min = min(pos_y_min, mesh.vertices[i][1]);
      pos_y_max = max(pos_y_max, mesh.vertices[i][1]);
      pos_z_min = min(pos_z_min, mesh.vertices[i][2]);
      pos_z_max = max(pos_z_max, mesh.vertices[i][2]);
    }
    if (neg_map[i] == true) {
      neg.vertices.push_back(mesh.vertices[i]);
      neg_proj[i] = ++neg_idx;

      neg_x_min = min(neg_x_min, mesh.vertices[i][0]);
      neg_x_max = max(neg_x_max, mesh.vertices[i][0]);
      neg_y_min = min(neg_y_min, mesh.vertices[i][1]);
      neg_y_max = max(neg_y_max, mesh.vertices[i][1]);
      neg_z_min = min(neg_z_min, mesh.vertices[i][2]);
      neg_z_max = max(neg_z_max, mesh.vertices[i][2]);
    }
  }

  int pos_N = (int)pos.vertices.size(), neg_N = (int)neg.vertices.size();

  if (pos_N == 0 || neg_N == 0) {
    MeshList meshlist;
    meshlist.push_back(mesh);
    return meshlist;
  }

  pos.is_new = vector<bool>(pos_N + (int)final_border.size(), false);
  neg.is_new = vector<bool>(neg_N + (int)final_border.size(), false);

  // border vertices & triangles
  for (int i = 0; i < (int)final_border.size(); i++) {
    pos.vertices.push_back(final_border[i]);
    neg.vertices.push_back(final_border[i]);

    pos.is_new[pos_N + i] = true;
    neg.is_new[neg_N + i] = true;

    pos_x_min = min(pos_x_min, final_border[i][0]);
    pos_x_max = max(pos_x_max, final_border[i][0]);
    pos_y_min = min(pos_y_min, final_border[i][1]);
    pos_y_max = max(pos_y_max, final_border[i][1]);
    pos_z_min = min(pos_z_min, final_border[i][2]);
    pos_z_max = max(pos_z_max, final_border[i][2]);

    neg_x_min = min(neg_x_min, final_border[i][0]);
    neg_x_max = max(neg_x_max, final_border[i][0]);
    neg_y_min = min(neg_y_min, final_border[i][1]);
    neg_y_max = max(neg_y_max, final_border[i][1]);
    neg_z_min = min(neg_z_min, final_border[i][2]);
    neg_z_max = max(neg_z_max, final_border[i][2]);
  }

  // triangles
  for (int i = 0; i < (int)pos.triangles.size(); i++) {
    int f0, f1, f2;
    if (pos.triangles[i][0] >= 0)
      f0 = pos_proj[pos.triangles[i][0]] - 1;
    else
      f0 = -1 * pos.triangles[i][0] + pos_N - 1;
    if (pos.triangles[i][1] >= 0)
      f1 = pos_proj[pos.triangles[i][1]] - 1;
    else
      f1 = -1 * pos.triangles[i][1] + pos_N - 1;
    if (pos.triangles[i][2] >= 0)
      f2 = pos_proj[pos.triangles[i][2]] - 1;
    else
      f2 = -1 * pos.triangles[i][2] + pos_N - 1;

    pos.triangles[i] = {f0, f1, f2};
  }
  for (int i = 0; i < (int)neg.triangles.size(); i++) {
    int f0, f1, f2;
    if (neg.triangles[i][0] >= 0)
      f0 = neg_proj[neg.triangles[i][0]] - 1;
    else
      f0 = -1 * neg.triangles[i][0] + neg_N - 1;
    if (neg.triangles[i][1] >= 0)
      f1 = neg_proj[neg.triangles[i][1]] - 1;
    else
      f1 = -1 * neg.triangles[i][1] + neg_N - 1;
    if (neg.triangles[i][2] >= 0)
      f2 = neg_proj[neg.triangles[i][2]] - 1;
    else
      f2 = -1 * neg.triangles[i][2] + neg_N - 1;

    neg.triangles[i] = {f0, f1, f2};
  }

  for (int i = 0; i < (int)final_triangles.size(); i++) {
    pos.triangles.push_back({pos_N + final_triangles[i][0],
                             pos_N + final_triangles[i][1],
                             pos_N + final_triangles[i][2]});
    neg.triangles.push_back({neg_N + final_triangles[i][2],
                             neg_N + final_triangles[i][1],
                             neg_N + final_triangles[i][0]});
  }

  MeshList mesh_list;

  mesh_list.push_back(pos);
  mesh_list.push_back(neg);

  return mesh_list;
}

MeshList multiclip(const Mesh mesh, const vector<Plane> &planes) {
  MeshList mesh_list;
  mesh_list.push_back(mesh);
  for (const auto &plane : planes) {
    for (int i = mesh_list.size() - 1; i >= 0; i--) {
      Mesh m = mesh_list[i];
      mesh_list.erase(mesh_list.begin() + i);
      int *part1_map, *part2_map;
      MeshList clipped = neural_acd::clip(m, plane, part1_map, part2_map);
      delete[] part1_map;
      delete[] part2_map;
      for (auto &c : clipped) {
        if (c.triangles.empty() || c.vertices.empty())
          continue; // skip empty meshes
        // manifold_preprocess(c);
        mesh_list.push_back(c);
      }
    }
  }

  return mesh_list;
}
} // namespace neural_acd
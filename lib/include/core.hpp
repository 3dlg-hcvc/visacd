#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace neural_acd {

inline std::mt19937 random_engine(std::random_device{}());

#define INF std::numeric_limits<double>::max()
using Vec3D = std::array<double, 3>;

class Plane {
public:
  double a, b, c, d;
  bool pFlag;       // whether three point form exists
  Vec3D p0, p1, p2; // three point form
  short cut_side(Vec3D p0, Vec3D p1, Vec3D p2, Plane plane);
  short bool_side(Vec3D p);
  short side(Vec3D p, double eps = 1e-6);
  bool intersect_segment(Vec3D p1, Vec3D p2, Vec3D &pi, double eps = 1e-6);
  Plane() { pFlag = false; };
  Plane(double _a, double _b, double _c, double _d) {
    a = _a;
    b = _b;
    c = _c;
    d = _d;
    pFlag = false;
  }
};

class Mesh {
public:
  Vec3D pos;
  std::vector<Vec3D> vertices;
  std::vector<std::array<int, 3>> triangles;
  std::vector<Vec3D> cut_verts;
  std::vector<bool> is_new;
  std::vector<std::pair<unsigned int,unsigned int>> intersecting_edges;
  Mesh();
  void compute_ch(Mesh &convex, bool fix_normals = false) const;
  void compute_vch(Mesh &convex, bool fix_normals = false) const;
  void extract_point_set(std::vector<Vec3D> &samples,
                         std::vector<int> &sample_tri_ids, size_t resolution,
                         double base = 0.0, bool flag = false,
                         Plane plane = Plane(), bool one_per_tri = true);

  void extract_point_set(std::vector<Vec3D> &samples,
                         std::vector<int> &sample_tri_ids, size_t resolution) {
    extract_point_set(samples, sample_tri_ids, resolution, 0.0, false, Plane(),
                      false);
  };
  std::vector<double> normalize();
  void unnormalize(const std::vector<double> &orig_bbox);
  void normalize(std::vector<Vec3D> &points);
  void clear();
  void save_obj(const std::string &filename) const;
  Mesh copy();
};
using MeshList = std::vector<Mesh>;

class LoadingBar {
public:
  std::string message;
  int total_steps;
  int bar_length;
  LoadingBar(std::string message_, int total_steps_, int bar_length_ = 50)
      : message(std::move(message_)), total_steps(total_steps_),
        bar_length(bar_length_) {};
  void step();
  void finish();

private:
  int current_step = 0;
};

inline Vec3D operator+(const Vec3D &a, const Vec3D &b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
inline Vec3D operator-(const Vec3D &a, const Vec3D &b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
inline Vec3D operator*(const Vec3D &v, double scalar) {
  return {v[0] * scalar, v[1] * scalar, v[2] * scalar};
}
inline Vec3D operator/(const Vec3D &v, double scalar) {
  if (scalar == 0) {
    throw std::runtime_error("Division by zero in vector division");
  }
  return {v[0] / scalar, v[1] / scalar, v[2] / scalar};
}

inline double vector_length(const Vec3D &v) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

inline Vec3D normalize_vector(const Vec3D &v) {
  double length = vector_length(v);
  if (length == 0) {
    throw std::runtime_error("Cannot normalize a zero-length vector");
  }
  return {v[0] / length, v[1] / length, v[2] / length};
}

inline Vec3D slerp(const Vec3D &a, const Vec3D &b, double t) {
  double dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  dot = std::clamp(dot, -1.0, 1.0); // Ensure dot product is in valid range
  double theta = std::acos(dot) * t;
  Vec3D relative_vec = b - a * dot;
  relative_vec = normalize_vector(relative_vec);
  return a * std::cos(theta) + relative_vec * std::sin(theta);
}

inline double triangle_area(Vec3D p0, Vec3D p1, Vec3D p2) {
  return 0.5 * sqrt(pow(p1[0] * p0[1] - p2[0] * p0[1] - p0[0] * p1[1] +
                            p2[0] * p1[1] + p0[0] * p2[1] - p1[0] * p2[1],
                        2) +
                    pow(p1[0] * p0[2] - p2[0] * p0[2] - p0[0] * p1[2] +
                            p2[0] * p1[2] + p0[0] * p2[2] - p1[0] * p2[2],
                        2) +
                    pow(p1[1] * p0[2] - p2[1] * p0[2] - p0[1] * p1[2] +
                            p2[1] * p1[2] + p0[1] * p2[2] - p1[1] * p2[2],
                        2));
}

inline Vec3D cross_product(Vec3D v, Vec3D w) {
  Vec3D res;
  res[0] = v[1] * w[2] - v[2] * w[1];
  res[1] = v[2] * w[0] - v[0] * w[2];
  res[2] = v[0] * w[1] - v[1] * w[0];

  return res;
}

inline double dot(Vec3D v, Vec3D w) {
  return v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
}

inline Vec3D calc_face_normal(Vec3D p1, Vec3D p2, Vec3D p3) {
  Vec3D v, w, n, normal;
  v[0] = p2[0] - p1[0];
  v[1] = p2[1] - p1[1];
  v[2] = p2[2] - p1[2];
  w[0] = p3[0] - p1[0];
  w[1] = p3[1] - p1[1];
  w[2] = p3[2] - p1[2];

  n = cross_product(v, w);

  normal[0] = n[0] / sqrt(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2));
  normal[1] = n[1] / sqrt(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2));
  normal[2] = n[2] / sqrt(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2));

  return normal;
}

inline short Plane::cut_side(Vec3D p0, Vec3D p1, Vec3D p2, Plane plane) {
  Vec3D normal = calc_face_normal(p0, p1, p2);
  if (normal[0] * plane.a > 0 || normal[1] * plane.b > 0 ||
      normal[2] * plane.c > 0)
    return -1;
  return 1;
}

inline short Plane::bool_side(Vec3D p) {
  double res = p[0] * a + p[1] * b + p[2] * c + d;
  if (res > 0)
    return 1;
  else
    return -1;
}

inline short Plane::side(Vec3D p, double eps) {
  double res = p[0] * a + p[1] * b + p[2] * c + d;
  if (res > eps)
    return 1;
  else if (res < -1 * eps)
    return -1;
  return 0;
}

inline bool Plane::intersect_segment(Vec3D p1, Vec3D p2, Vec3D &pi,
                                     double eps) {
  pi[0] =
      (p1[0] * b * p2[1] + p1[0] * c * p2[2] + p1[0] * d - p2[0] * b * p1[1] -
       p2[0] * c * p1[2] - p2[0] * d) /
      (a * p2[0] - a * p1[0] + b * p2[1] - b * p1[1] + c * p2[2] - c * p1[2]);
  pi[1] =
      (a * p2[0] * p1[1] + c * p1[1] * p2[2] + p1[1] * d - a * p1[0] * p2[1] -
       c * p1[2] * p2[1] - p2[1] * d) /
      (a * p2[0] - a * p1[0] + b * p2[1] - b * p1[1] + c * p2[2] - c * p1[2]);
  pi[2] =
      (a * p2[0] * p1[2] + b * p2[1] * p1[2] + p1[2] * d - a * p1[0] * p2[2] -
       b * p1[1] * p2[2] - p2[2] * d) /
      (a * p2[0] - a * p1[0] + b * p2[1] - b * p1[1] + c * p2[2] - c * p1[2]);

  if (std::min(p1[0] - eps, p2[0] - eps) <= pi[0] &&
      pi[0] <= std::max(p1[0] + eps, p2[0] + eps) &&
      std::min(p1[1] - eps, p2[1] - eps) <= pi[1] &&
      pi[1] <= std::max(p1[1] + eps, p2[1] + eps) &&
      std::min(p1[2] - eps, p2[2] - eps) <= pi[2] &&
      pi[2] <= std::max(p1[2] + eps, p2[2] + eps))
    return true;
  return false;
}

void subdivide_edge(const Vec3D &v1, const Vec3D &v2,
                    std::vector<Vec3D> &new_vertices, int depth);

void extract_point_set(Mesh &convex1, Mesh &convex2,
                       std::vector<Vec3D> &samples,
                       std::vector<int> &sample_tri_id, size_t resolution);

void set_seed(unsigned int seed);

} // namespace neural_acd
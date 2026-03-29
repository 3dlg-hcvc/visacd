/*MIT License

Copyright (c) 2022 Xinyue Wei, Minghua Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once
#include <algorithm>
#include <assert.h>
#include <math.h>
#include <thread>
#include <time.h>
#include <typeinfo>

#include <iomanip>
#include <iostream>
#include <map>
#include <stdio.h>
#include <string.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "clip.hpp"
#include "core.hpp"
#include "nanoflann.hpp"
using namespace std;
using namespace nanoflann;

namespace neural_acd {

template <typename T> struct PointCloud {
  struct Point {
    T x, y, z;
  };

  std::vector<Point> pts;

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return pts.size(); }

  // Returns the dim'th component of the idx'th point in the class:
  // Since this is inlined and the "dim" argument is typically an immediate
  // value, the
  //  "if/else's" are actually solved at compile time.
  inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
    if (dim == 0)
      return pts[idx].x;
    else if (dim == 1)
      return pts[idx].y;
    else
      return pts[idx].z;
  }

  // Optional bounding-box computation: return false to default to a standard
  // bbox computation loop.
  //   Return true if the BBOX was already computed by the class and returned in
  //   "bb" so it can be avoided to redo it again. Look at bb.size() to find out
  //   the expected dimensionality (e.g. 2 or 3 for point clouds)
  template <class BBOX> bool kdtree_get_bbox(BBOX & /* bb */) const {
    return false;
  }
};

template <typename T> void vec2pc(PointCloud<T> &point, vector<Vec3D> V) {
  point.pts.resize(V.size());
  for (size_t i = 0; i < V.size(); i++) {
    point.pts[i].x = V[i][0];
    point.pts[i].y = V[i][1];
    point.pts[i].z = V[i][2];
  }
}

inline bool same_vector_dir(Vec3D v, Vec3D w) {
  if (v[0] * w[0] + v[1] * w[1] + v[2] * w[2] > 0)
    return true;
  return false;
}

double dist_point2point(Vec3D pt, Vec3D p) {
  return sqrt(pow(pt[0] - p[0], 2) + pow(pt[1] - p[1], 2) +
              pow(pt[2] - p[2], 2));
}

double dist_point2segment(Vec3D pt, Vec3D s0, Vec3D s1, bool flag = false) {
  // first we build a 3d triangle the compute the height, pt as the top point
  Vec3D BA, BC;
  BA[0] = pt[0] - s1[0];
  BA[1] = pt[1] - s1[1];
  BA[2] = pt[2] - s1[2];
  BC[0] = s0[0] - s1[0];
  BC[1] = s0[1] - s1[1];
  BC[2] = s0[2] - s1[2];

  // we calculate the projected vector along the segment
  double proj_dist = (BA[0] * BC[0] + BA[1] * BC[1] + BA[2] * BC[2]) /
                     (sqrt(pow(BC[0], 2) + pow(BC[1], 2) + pow(BC[2], 2)));
  if (flag) {
    Vec3D proj_pt;
    double len_BC = sqrt(pow(BC[0], 2) + pow(BC[1], 2) + pow(BC[2], 2));
    proj_pt[0] = s1[0] + proj_dist / len_BC * BC[0];
    proj_pt[1] = s1[1] + proj_dist / len_BC * BC[1];
    proj_pt[2] = s1[2] + proj_dist / len_BC * BC[2];
  }

  // we should make sure the projected point is within the segment, otherwise
  // not consider it if projected distance is negative or bigger than BC, it is
  // out
  double valAB = sqrt(pow(BA[0], 2) + pow(BA[1], 2) + pow(BA[2], 2));
  double valBC = sqrt(pow(BC[0], 2) + pow(BC[1], 2) + pow(BC[2], 2));
  if (proj_dist < 0 || proj_dist > valBC)
    return INF;
  return sqrt(pow(valAB, 2) - pow(proj_dist, 2));
}

bool point_in_triangle(Vec3D pt, Vec3D tri_pt0, Vec3D tri_pt1, Vec3D tri_pt2,
                       Vec3D normal) {
  // Compute vectors
  Vec3D v0 = tri_pt2 - tri_pt0;
  Vec3D v1 = tri_pt1 - tri_pt0;
  Vec3D v2 = pt - tri_pt0;

  // Compute dot products
  double dot00 = dot(v0, v0);
  double dot01 = dot(v0, v1);
  double dot02 = dot(v0, v2);
  double dot11 = dot(v1, v1);
  double dot12 = dot(v1, v2);

  // Compute barycentric coordinates
  double invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
  double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
  double v = (dot00 * dot12 - dot01 * dot02) * invDenom;

  // Check if point is in triangle
  return (u >= 0) && (v >= 0) && (u + v <= 1);
}

double dist_point2triangle(Vec3D pt, Vec3D tri_pt0, Vec3D tri_pt1,
                           Vec3D tri_pt2, bool flag = false) {
  // calculate the funciton of the plane, n = (a, b, c)
  double _a = (tri_pt1[1] - tri_pt0[1]) * (tri_pt2[2] - tri_pt0[2]) -
              (tri_pt1[2] - tri_pt0[2]) * (tri_pt2[1] - tri_pt0[1]);
  double _b = (tri_pt1[2] - tri_pt0[2]) * (tri_pt2[0] - tri_pt0[0]) -
              (tri_pt1[0] - tri_pt0[0]) * (tri_pt2[2] - tri_pt0[2]);
  double _c = (tri_pt1[0] - tri_pt0[0]) * (tri_pt2[1] - tri_pt0[1]) -
              (tri_pt1[1] - tri_pt0[1]) * (tri_pt2[0] - tri_pt0[0]);
  double a = _a / sqrt(pow(_a, 2) + pow(_b, 2) + pow(_c, 2));
  double b = _b / sqrt(pow(_a, 2) + pow(_b, 2) + pow(_c, 2));
  double c = _c / sqrt(pow(_a, 2) + pow(_b, 2) + pow(_c, 2));
  double d = 0 - (a * tri_pt0[0] + b * tri_pt0[1] + c * tri_pt0[2]);

  // distance can be calculated directly using the function, then we get the
  // projected point as well
  double dist = fabs(a * pt[0] + b * pt[1] + c * pt[2] + d) /
                sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2));

  Vec3D proj_pt;
  Plane p = Plane(a, b, c, d);
  short side = p.side(pt, 1e-8);
  if (side == 1) {
    proj_pt[0] = pt[0] - a * dist;
    proj_pt[1] = pt[1] - b * dist;
    proj_pt[2] = pt[2] - c * dist;
  } else if (side == -1) {
    proj_pt[0] = pt[0] + a * dist;
    proj_pt[1] = pt[1] + b * dist;
    proj_pt[2] = pt[2] + c * dist;
  } else {
    proj_pt = pt;
  }

  Vec3D normal = calc_face_normal(tri_pt0, tri_pt1, tri_pt2);

  // Check if projected point is inside triangle using barycentric coords
  if (point_in_triangle(proj_pt, tri_pt0, tri_pt1, tri_pt2, normal)) {
    return dist;
  } else // if not within the triangle, we calculate the distance to 3 edges and
         // 3 points and use the min
  {

    double dist_pt2AB = dist_point2segment(pt, tri_pt0, tri_pt1, flag);
    double dist_pt2BC = dist_point2segment(pt, tri_pt1, tri_pt2, flag);
    double dist_pt2CA = dist_point2segment(pt, tri_pt2, tri_pt0, flag);

    double dist_pt2A = dist_point2point(pt, tri_pt0);
    double dist_pt2B = dist_point2point(pt, tri_pt1);
    double dist_pt2C = dist_point2point(pt, tri_pt2);
    d = min(min(min(dist_pt2AB, dist_pt2BC), dist_pt2CA),
            min(min(dist_pt2A, dist_pt2B), dist_pt2C));

    return min(min(min(dist_pt2AB, dist_pt2BC), dist_pt2CA),
               min(min(dist_pt2A, dist_pt2B), dist_pt2C));
  }
}

double face_hausdorff_distance(Mesh &meshA, vector<Vec3D> &XA, vector<int> &idA,
                               Mesh &meshB, vector<Vec3D> &XB, vector<int> &idB,
                               bool flag = false) {
  int nA = XA.size();
  int nB = XB.size();
  double cmax = 0;

  PointCloud<double> cloudA, cloudB;
  vec2pc(cloudA, XA);
  vec2pc(cloudB, XB);

  typedef KDTreeSingleIndexAdaptor<
      L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>,
      3 /* dim */
      >
      my_kd_tree_t;

  my_kd_tree_t indexA(3 /*dim*/, cloudA,
                      KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
  my_kd_tree_t indexB(3 /*dim*/, cloudB,
                      KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
  indexA.buildIndex();
  indexB.buildIndex();

  for (int i = 0; i < nB; i++) {
    size_t num_results = 10;
    double query_pt[3] = {XB[i][0], XB[i][1], XB[i][2]};

    std::vector<size_t> ret_index(num_results);
    std::vector<double> out_dist_sqr(num_results);

    num_results = indexA.knnSearch(&query_pt[0], num_results, &ret_index[0],
                                   &out_dist_sqr[0]);

    double cmin = INF;
    for (int j = 0; j < (int)num_results; j++) {
      double distance;
      distance = dist_point2triangle(
          XB[i], meshA.vertices[meshA.triangles[idA[ret_index[j]]][0]],
          meshA.vertices[meshA.triangles[idA[ret_index[j]]][1]],
          meshA.vertices[meshA.triangles[idA[ret_index[j]]][2]]);
      if (distance < cmin) {
        cmin = distance;
        if (cmin < 1e-14)
          break;
      }
    }
    if (cmin > 10)
      cmin = sqrt(out_dist_sqr[0]);
    if (cmin > cmax && INF > cmin)
      cmax = cmin;
  }

  for (int i = 0; i < nA; i++) {
    size_t num_results = 10;

    double query_pt[3] = {XA[i][0], XA[i][1], XA[i][2]};

    std::vector<size_t> ret_index(num_results);
    std::vector<double> out_dist_sqr(num_results);

    num_results = indexB.knnSearch(&query_pt[0], num_results, &ret_index[0],
                                   &out_dist_sqr[0]);

    double cmin = INF;
    for (int j = 0; j < (int)num_results; j++) {
      double distance;
      distance = dist_point2triangle(
          XA[i], meshB.vertices[meshB.triangles[idB[ret_index[j]]][0]],
          meshB.vertices[meshB.triangles[idB[ret_index[j]]][1]],
          meshB.vertices[meshB.triangles[idB[ret_index[j]]][2]]);
      if (distance < cmin) {
        cmin = distance;
        if (cmin < 1e-14)
          break;
      }
    }
    if (cmin > 10)
      cmin = sqrt(out_dist_sqr[0]);
    if (cmin > cmax && INF > cmin)
      cmax = cmin;
  }

  return cmax;
}
} // namespace neural_acd
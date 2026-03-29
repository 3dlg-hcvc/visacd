#include "core.hpp"
#include <QuickHull.hpp>
#include <algorithm>
#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <btConvexHullComputer.h>
#include <cmath>
#include <iostream>
#include <libqhullcpp/Qhull.h>
#include <libqhullcpp/QhullFacetList.h>
#include <libqhullcpp/QhullPoints.h>
#include <libqhullcpp/QhullVertexSet.h>
#include <random>
#include <stdexcept>
#include <unordered_set>

using namespace std;

namespace neural_acd {

boost::random::sobol sobol_engine(2);
boost::uniform_01<double> uniform_dist;
boost::variate_generator<boost::random::sobol &, boost::uniform_01<double>>
    sobol_gen(sobol_engine, uniform_dist);

void set_seed(unsigned int seed) { random_engine.seed(seed); }

Mesh::Mesh() {
}


void cvx_fix_normals(Mesh &convex) {
  Vec3D center = {0, 0, 0};
  for (const auto &v : convex.vertices) {
    center = center + v;
  }
  center = center / convex.vertices.size();

  for (auto &tri : convex.triangles) {
    Vec3D v0 = convex.vertices[tri[0]];
    Vec3D v1 = convex.vertices[tri[1]];
    Vec3D v2 = convex.vertices[tri[2]];

    Vec3D edge1 = v1 - v0;
    Vec3D edge2 = v2 - v0;
    Vec3D normal = cross_product(edge1, edge2);

    // Vector from triangle vertex to center
    Vec3D to_center = center - v0;

    // If normal points toward center, flip the winding
    if (dot(normal, to_center) > 0) {
      std::swap(tri[1], tri[2]); // No const_cast needed
    }
  }
}

void Mesh::compute_ch(Mesh &convex, bool fix_normals) const {
  using namespace orgQhull;

  // Collect input points into a flat array
  std::vector<double> coords;
  coords.reserve(vertices.size() * 3);
  for (const auto &v : vertices) {
    coords.push_back(v[0]);
    coords.push_back(v[1]);
    coords.push_back(v[2]);
  }

  try {
    // Run Qhull: convex hull in 3D ("Qt" option disables triangulation of
    // coplanar facets)
    Qhull qh;
    qh.runQhull("convex_hull", 3, static_cast<int>(vertices.size()),
                coords.data(), "Qt Pp");

    for (coordT *  it = qh.pointCoordinateBegin(); it != qh.pointCoordinateEnd();
         it += 3) {
      convex.vertices.push_back({it[0], it[1], it[2]});
    }


    // Extract facets (triangles) from Qhull
    for (const QhullFacet &facet : qh.facetList()) {
      if (!facet.isGood())
        continue; // skip "bad" facets
      QhullVertexSet vset = facet.vertices();
      if (vset.size() == 3) {
        std::array<int, 3> tri;
        int j = 0;
        for (const QhullVertex &fv : vset) {
          tri[j++] = fv.point().id();
        }
        // Note: adjust winding order if needed
        convex.triangles.push_back({tri[0], tri[1], tri[2]});
      }
    }
  } catch (const std::exception &e) {
    // fallback: stable but slow algorithm
    save_obj("debug_input.obj");
    cout<<"Qhull failed, falling back to BulletConvexHullComputer"<<endl;
    // cout<<"Qhull failed: "<<e.what()<<", falling back to BulletConvexHullComputer"<<endl;
    compute_vch(convex, fix_normals);
  }

  if (fix_normals) {
    cvx_fix_normals(convex);
  }

}

void Mesh::compute_vch(Mesh &convex, bool fix_normals) const {
  btConvexHullComputer ch;
  ch.compute(vertices, -1.0, -1.0);
  // Mesh empty = Mesh();
  // for (const auto &v : vertices) {
  //   empty.vertices.push_back(v);
  // }
  // if (fix_normals)
  // empty.save_obj("debug_input.obj");
  for (int32_t v = 0; v < ch.vertices.size(); v++) {
    convex.vertices.push_back(
        {ch.vertices[v].getX(), ch.vertices[v].getY(), ch.vertices[v].getZ()});
  }
  const int32_t nt = ch.faces.size();
  for (int32_t t = 0; t < nt; ++t) {
    const btConvexHullComputer::Edge *sourceEdge = &(ch.edges[ch.faces[t]]);
    int32_t a = sourceEdge->getSourceVertex();
    int32_t b = sourceEdge->getTargetVertex();
    const btConvexHullComputer::Edge *edge = sourceEdge->getNextEdgeOfFace();
    int32_t c = edge->getTargetVertex();
    while (c != a) {
      convex.triangles.push_back({(int)a, (int)b, (int)c});
      edge = edge->getNextEdgeOfFace();
      b = c;
      c = edge->getTargetVertex();
    }
  }
  if (fix_normals)
  cvx_fix_normals(convex);
}

void Mesh::extract_point_set(vector<Vec3D> &samples,
                             vector<int> &sample_tri_ids, size_t resolution,
                             double base, bool flag, Plane plane,
                             bool one_per_tri) {

  if (triangles.empty() || vertices.empty()) {
    return;
  }

  double aObj = 0.0;

  vector<double> areas = vector<double>(triangles.size(), 0.0);
  for (size_t i = 0; i < triangles.size(); i++) {
    double area =
        triangle_area(vertices[triangles[i][0]], vertices[triangles[i][1]],
                      vertices[triangles[i][2]]);
    areas[i] = area;
    aObj += area;
  }

  if (base != 0)
    resolution = size_t(max(1000, int(resolution * (aObj / base))));

  discrete_distribution<size_t> triangle_index_generator(areas.begin(),
                                                         areas.end());

  uniform_real_distribution<double> uniform_dist(0.0, 1.0);

  unordered_set<size_t> sampled_tris;

  int sampled = 0;

  while (sampled < resolution) {

    size_t tidx = triangle_index_generator(random_engine);

    const auto &tri = triangles[tidx];

    if (flag && plane.side(vertices[tri[0]], 1e-3) == 0 &&
        plane.side(vertices[tri[1]], 1e-3) == 0 &&
        plane.side(vertices[tri[2]], 1e-3) == 0) {
      continue;
    }

    double a = uniform_dist(random_engine);
    double b = uniform_dist(random_engine);

    Vec3D v;
    v[0] = (1 - sqrt(a)) * vertices[tri[0]][0] +
           (sqrt(a) * (1 - b)) * vertices[tri[1]][0] +
           b * sqrt(a) * vertices[tri[2]][0];
    v[1] = (1 - sqrt(a)) * vertices[tri[0]][1] +
           (sqrt(a) * (1 - b)) * vertices[tri[1]][1] +
           b * sqrt(a) * vertices[tri[2]][1];
    v[2] = (1 - sqrt(a)) * vertices[tri[0]][2] +
           (sqrt(a) * (1 - b)) * vertices[tri[1]][2] +
           b * sqrt(a) * vertices[tri[2]][2];
    samples.push_back(v);
    sample_tri_ids.push_back(tidx);
    sampled_tris.insert(tidx);
    sampled++;
  }

  if (one_per_tri) {
    for (size_t i = 0; i < triangles.size(); i++) {
      if (sampled_tris.find(i) == sampled_tris.end()) {

        const auto &tri = triangles[i];

        double a = uniform_dist(random_engine);
        double b = uniform_dist(random_engine);

        Vec3D v;
        v[0] = (1 - sqrt(a)) * vertices[tri[0]][0] +
               (sqrt(a) * (1 - b)) * vertices[tri[1]][0] +
               b * sqrt(a) * vertices[tri[2]][0];
        v[1] = (1 - sqrt(a)) * vertices[tri[0]][1] +
               (sqrt(a) * (1 - b)) * vertices[tri[1]][1] +
               b * sqrt(a) * vertices[tri[2]][1];
        v[2] = (1 - sqrt(a)) * vertices[tri[0]][2] +
               (sqrt(a) * (1 - b)) * vertices[tri[1]][2] +
               b * sqrt(a) * vertices[tri[2]][2];

        samples.push_back(v);
        sample_tri_ids.push_back(i);
      }
    }
  }
}

void Mesh::clear() {
  vertices.clear();
  triangles.clear();
}

Mesh Mesh::copy() {
  Mesh new_mesh;
  new_mesh.pos = pos;
  new_mesh.vertices = vertices;
  new_mesh.triangles = triangles;
  new_mesh.cut_verts = cut_verts;
  return new_mesh;
}

vector<double> compute_bbox(const Mesh &mesh) {
  double x_min = INF, x_max = -INF, y_min = INF, y_max = -INF, z_min = INF,
         z_max = -INF;
  for (int i = 0; i < (int)mesh.vertices.size(); ++i) {
    x_min = min(x_min, mesh.vertices[i][0]);
    x_max = max(x_max, mesh.vertices[i][0]);
    y_min = min(y_min, mesh.vertices[i][1]);
    y_max = max(y_max, mesh.vertices[i][1]);
    z_min = min(z_min, mesh.vertices[i][2]);
    z_max = max(z_max, mesh.vertices[i][2]);
  }
  return vector<double>{x_min, x_max, y_min, y_max, z_min, z_max};
}

vector<double> Mesh::normalize(){
    double m_len;
    double m_Xmid, m_Ymid, m_Zmid;

    vector<double> bbox = compute_bbox(*this);

    double x_min = bbox[0], x_max = bbox[1], y_min = bbox[2], y_max = bbox[3], z_min = bbox[4], z_max = bbox[5];

    m_len = max(max(x_max - x_min, y_max - y_min), z_max - z_min);
    m_Xmid = (x_max + x_min) / 2;
    m_Ymid = (y_max + y_min) / 2;
    m_Zmid = (z_max + z_min) / 2;

    for (int i = 0; i < (int)vertices.size(); i++)
    {
        Vec3D tmp = {2.0 * (vertices[i][0] - m_Xmid) / m_len,
                      2.0 * (vertices[i][1] - m_Ymid) / m_len,
                      2.0 * (vertices[i][2] - m_Zmid) / m_len};
        vertices[i] = tmp;
    }
    double x_len = bbox[1] - bbox[0], y_len = bbox[3] - bbox[2], z_len = bbox[5] - bbox[4];
    bbox[0] = -x_len / m_len;
    bbox[1] = x_len / m_len;
    bbox[2] = -y_len / m_len;
    bbox[3] = y_len / m_len;
    bbox[4] = -z_len / m_len;
    bbox[5] = z_len / m_len;

    return vector<double> {x_min, x_max, y_min, y_max, z_min, z_max};
}

void Mesh::unnormalize(const vector<double> &_bbox){
  double m_len;
  double m_Xmid, m_Ymid, m_Zmid;
  double x_min = _bbox[0], x_max = _bbox[1], y_min = _bbox[2], y_max = _bbox[3], z_min = _bbox[4], z_max = _bbox[5];

  m_len = max(max(x_max - x_min, y_max - y_min), z_max - z_min);
  m_Xmid = (x_max + x_min) / 2;
  m_Ymid = (y_max + y_min) / 2;
  m_Zmid = (z_max + z_min) / 2;

  for (int i = 0; i < (int)vertices.size(); i++)
      vertices[i] = {vertices[i][0] / 2 * m_len + m_Xmid,
                    vertices[i][1] / 2 * m_len + m_Ymid,
                    vertices[i][2] / 2 * m_len + m_Zmid};
}

void Mesh::normalize(vector<Vec3D> &points) {
  if (vertices.empty())
    return;

  Vec3D mn = vertices[0], mx = vertices[0];

  // Compute bounding box
  for (const auto &v : vertices) {
    for (int i = 0; i < 3; i++) {
      if (v[i] < mn[i])
        mn[i] = v[i];
      if (v[i] > mx[i])
        mx[i] = v[i];
    }
  }

  // Compute center of bounding box
  Vec3D center = {(mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0,
                  (mn[2] + mx[2]) / 2.0};

  // Compute longest side length
  double scale = max({mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]});

  // Normalize: center and scale to fit in [-1, 1] along the longest axis
  for (auto &v : vertices) {
    v = (v - center) * 2 / scale;
  }
  for (auto &p : points) {
    p = (p - center) * 2 / scale;
  }
}

void Mesh::save_obj(const std::string &filename) const{
  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }
  for (const auto &v : vertices) {
    ofs << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
  }
  for (const auto &tri : triangles) {
    ofs << "f " << (tri[0] + 1) << " " << (tri[1] + 1) << " " << (tri[2] + 1)
        << "\n";
  }
  ofs.close();
}

void get_midpoint(const Vec3D &v1, const Vec3D &v2, Vec3D &mid) {
  mid[0] = (v1[0] + v2[0]) / 2.0;
  mid[1] = (v1[1] + v2[1]) / 2.0;
  mid[2] = (v1[2] + v2[2]) / 2.0;
}

void subdivide_edge(const Vec3D &v1, const Vec3D &v2,
                    vector<Vec3D> &new_vertices, int depth) {
  Vec3D mid;
  get_midpoint(v1, v2, mid);
  new_vertices.push_back(mid);
  if (depth == 0) {
    return;
  }
  subdivide_edge(v1, mid, new_vertices, depth - 1);
  subdivide_edge(mid, v2, new_vertices, depth - 1);
}

bool compute_overlap_face(Mesh &convex1, Mesh &convex2, Plane &plane) {
  bool flag;
  for (int i = 0; i < (int)convex1.triangles.size(); i++) {
    Plane p;
    Vec3D p1, p2, p3;
    p1 = convex1.vertices[convex1.triangles[i][0]];
    p2 = convex1.vertices[convex1.triangles[i][1]];
    p3 = convex1.vertices[convex1.triangles[i][2]];
    double a =
        (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1]);
    double b =
        (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2]);
    double c =
        (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]);
    p.a = a / sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2));
    p.b = b / sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2));
    p.c = c / sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2));
    p.d = 0 - (p.a * p1[0] + p.b * p1[1] + p.c * p1[2]);

    short side1 = 0;
    for (int j = 0; j < (int)convex1.vertices.size(); j++) {
      short s = p.side(convex1.vertices[j], 1e-8);
      if (s != 0) {
        side1 = s;
        flag = 1;
        break;
      }
    }

    for (int j = 0; j < (int)convex2.vertices.size(); j++) {
      short s = p.side(convex2.vertices[j], 1e-8);
      if (!flag || s == side1) {
        flag = 0;
        break;
      }
    }
    if (flag) {
      plane = p;
      return true;
    }
  }
  return false;
}

void extract_point_set(Mesh &convex1, Mesh &convex2, vector<Vec3D> &samples,
                       vector<int> &sample_tri_ids, size_t resolution) {
  vector<Vec3D> samples1, samples2;
  vector<int> sample_tri_ids1, sample_tri_ids2;
  double a1 = 0, a2 = 0;
  for (int i = 0; i < (int)convex1.triangles.size(); i++)
    a1 += triangle_area(convex1.vertices[convex1.triangles[i][0]],
                        convex1.vertices[convex1.triangles[i][1]],
                        convex1.vertices[convex1.triangles[i][2]]);
  for (int i = 0; i < (int)convex2.triangles.size(); i++)
    a2 += triangle_area(convex2.vertices[convex2.triangles[i][0]],
                        convex2.vertices[convex2.triangles[i][1]],
                        convex2.vertices[convex2.triangles[i][2]]);

  Plane overlap_plane;
  bool flag = compute_overlap_face(convex1, convex2, overlap_plane);

  convex1.extract_point_set(samples1, sample_tri_ids1,
                            size_t(a1 / (a1 + a2) * resolution), 1, flag,
                            overlap_plane);
  convex2.extract_point_set(samples2, sample_tri_ids2,
                            size_t(a2 / (a1 + a2) * resolution), 1, flag,
                            overlap_plane);

  samples.insert(samples.end(), samples1.begin(), samples1.end());
  samples.insert(samples.end(), samples2.begin(), samples2.end());

  sample_tri_ids.insert(sample_tri_ids.end(), sample_tri_ids1.begin(),
                        sample_tri_ids1.end());
  int N = (int)convex1.triangles.size();
  for (int i = 0; i < (int)sample_tri_ids2.size(); i++)
    sample_tri_ids.push_back(sample_tri_ids2[i] + N);
}

void LoadingBar::step() {
  string bar;
  bar += "\r" + message + " [";
  int pos = (current_step * bar_length) / total_steps;
  for (int i = 0; i < bar_length; ++i) {
    if (i < pos)
      bar += "=";
    else
      bar += " ";
  }
  bar += "] " + to_string(current_step) + "/" + to_string(total_steps);
  cout << bar;
  cout.flush();
  current_step++;
}

void LoadingBar::finish() {
  cout << "\r" + message + " [";
  for (int i = 0; i < bar_length; ++i) {
    if (i < bar_length)
      cout << "=";
    else
      cout << " ";
  }
  cout << "] " << total_steps << "/" << total_steps << endl;
}
} // namespace neural_acd
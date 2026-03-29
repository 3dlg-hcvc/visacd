#include <array>
#include <cmath>
#include <queue>
#include <support_surface.hpp>
#include <unordered_set>
#include <vector>

namespace neural_acd {

static inline double cosine_sim(const Vec3D &n1, const Vec3D &n2) {
  return dot(n1, n2) / (sqrt(dot(n1, n1)) * sqrt(dot(n2, n2)));
}

std::vector<Surface> extract_surfaces(const Mesh &mesh, double min_area) {
  const auto &tris = mesh.triangles;
  const auto &verts = mesh.vertices;
  const int T = tris.size();

  //-------------------------------------------------------
  // 1. Compute triangle normals
  //-------------------------------------------------------
  std::vector<Vec3D> normals(T);
  for (int i = 0; i < T; i++) {
    const auto &t = tris[i];
    normals[i] = calc_face_normal(verts[t[0]], verts[t[1]], verts[t[2]]);
  }

  //-------------------------------------------------------
  // 2. Build adjacency: triangles that share an edge
  //-------------------------------------------------------
  // Map: (minV, maxV) → list of triangle indices touching this edge
  std::unordered_map<long long, std::vector<int>> edge_map;
  edge_map.reserve(T * 3);

  auto encode = [&](int a, int b) {
    if (a > b)
      std::swap(a, b);
    return ((long long)a << 32) | (unsigned long long)b;
  };

  for (int i = 0; i < T; i++) {
    const auto &t = tris[i];
    long long e0 = encode(t[0], t[1]);
    long long e1 = encode(t[1], t[2]);
    long long e2 = encode(t[2], t[0]);
    edge_map[e0].push_back(i);
    edge_map[e1].push_back(i);
    edge_map[e2].push_back(i);
  }

  //-------------------------------------------------------
  // Build adjacency list for each triangle
  //-------------------------------------------------------
  std::vector<std::vector<int>> adj(T);
  for (auto &kv : edge_map) {
    const auto &list = kv.second;
    if (list.size() < 2)
      continue;
    for (int a : list)
      for (int b : list)
        if (a != b)
          adj[a].push_back(b);
  }

  //-------------------------------------------------------
  // 3. Region growing by normal similarity
  //-------------------------------------------------------
  std::vector<bool> used(T, false);
  std::vector<Surface> surfaces;

  for (int start = 0; start < T; start++) {
    if (used[start])
      continue;

    Surface surf;
    surf.triangle_ids.push_back(start);
    used[start] = true;

    Vec3D avg_normal = normals[start];

    std::queue<int> Q;
    Q.push(start);

    while (!Q.empty()) {
      int tid = Q.front();
      Q.pop();

      for (int nbr : adj[tid]) {
        if (used[nbr])
          continue;

        double sim = cosine_sim(avg_normal, normals[nbr]);
        if (sim > 0.999) {
          used[nbr] = true;
          surf.triangle_ids.push_back(nbr);
          // Update average normal
          avg_normal =
              (avg_normal * (surf.triangle_ids.size() - 1) + normals[nbr]) /
              surf.triangle_ids.size();
          Q.push(nbr);
        }
      }
    }

    surfaces.push_back(std::move(surf));
  }

  //-------------------------------------------------------
  // 4. Compute areas, filter small surfaces
  //-------------------------------------------------------
  std::vector<Surface> filtered;

  for (auto &S : surfaces) {
    double A = 0.0;
    for (int tid : S.triangle_ids) {
      const auto &t = tris[tid];
      A += triangle_area(verts[t[0]], verts[t[1]], verts[t[2]]);
    }
    S.area = A;

    if (A > min_area)
      filtered.push_back(S);
  }

  //if size > 40, pick top 40 largest surfaces
  if (filtered.size() > 40) {
    std::sort(filtered.begin(), filtered.end(),
              [](const Surface &a, const Surface &b) { return a.area > b.area; });
    filtered.resize(40);
  }

  //for each surface compute a plane
  for (auto &S : filtered) {
    // compute average normal and a point on the surface
    Vec3D avg_normal = {0.0, 0.0, 0.0};
    Vec3D point_on_surface = {0.0, 0.0, 0.0};
    for (int tid : S.triangle_ids) {
      const auto &t = tris[tid];
      Vec3D n = calc_face_normal(verts[t[0]], verts[t[1]], verts[t[2]]);
      avg_normal = avg_normal + n;
      point_on_surface = point_on_surface + 
                         (verts[t[0]] + verts[t[1]] + verts[t[2]]) / 3.0;
    }
    avg_normal = normalize_vector(avg_normal);
    point_on_surface = point_on_surface / S.triangle_ids.size();

    //slightly adjust point_on_surface along normal direction to avoid numerical issues
    point_on_surface = point_on_surface + avg_normal * 2e-2;

    double d = -dot(avg_normal, point_on_surface);
    S.plane = Plane(avg_normal[0], avg_normal[1], avg_normal[2], d);
  }

  // Remove duplicate planes
  std::vector<Surface> unique_surfaces;
  for (const auto &S : filtered) {
    bool duplicate = false;
    for (const auto &existing : unique_surfaces) {
      const auto &p1 = S.plane;
      const auto &p2 = existing.plane;

      double dot_prod = p1.a * p2.a + p1.b * p2.b + p1.c * p2.c + p1.d * p2.d;
      double mag1 =
          std::sqrt(p1.a * p1.a + p1.b * p1.b + p1.c * p1.c + p1.d * p1.d);
      double mag2 =
          std::sqrt(p2.a * p2.a + p2.b * p2.b + p2.c * p2.c + p2.d * p2.d);

      if (mag1 > 1e-9 && mag2 > 1e-9) {
        double cos_theta = std::abs(dot_prod) / (mag1 * mag2);
        if (cos_theta > 1.0 - 1e-4) {
          duplicate = true;
          break;
        }
      }
    }
    if (!duplicate) {
      unique_surfaces.push_back(S);
    }
  }

  return unique_surfaces;
}

} // namespace neural_acd
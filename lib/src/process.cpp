#include <CDT.h>
#include <CDTUtils.h>
#include <Eigen/Dense>
#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <clip.hpp>
#include <config.hpp>
#include <core.hpp>
#include <cost.hpp>
#include <intersections.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <optixUtils.hpp>
#include <optix_function_table_definition.h>
#include <postprocess.hpp>
#include <preprocess.hpp>
#include <process.hpp>
#include <vector>
#include <support_surface.hpp>

using namespace std;

namespace neural_acd {

  vector<Plane> support_surface_planes;

  vector<Plane>
  get_candidate_planes(vector<Vec3D> &vertices,
                       vector<pair<unsigned int, unsigned int>> &edges,
                       int num_planes)
  {
      vector<Plane> planes;

      if (edges.empty())
          return planes;

      uniform_int_distribution<> dis(0, edges.size() - 1);
      const double normal_eps = cos(5.0 * M_PI / 180.0); // ~5° angular tolerance
      const double dist_eps = 1e-3;                      // distance tolerance

      for (int i = 0; i < num_planes * 5 && (int)planes.size() < num_planes; ++i) {
          int idx = dis(random_engine);
          Vec3D p1 = vertices[edges[idx].first];
          Vec3D p2 = vertices[edges[idx].second];
          Vec3D n = p2 - p1; // normal vector

          if (vector_length(n) < 1e-6)
              continue;

          n = normalize_vector(n);
          Vec3D m = (p1 + p2) * 0.5; // midpoint
          double d = -(n[0] * m[0] + n[1] * m[1] + n[2] * m[2]);

          Plane candidate(n[0], n[1], n[2], d);

          // check if similar to an existing plane
          bool too_similar = false;
          for (const auto &p : planes) {
              double dot = fabs(n[0]*p.a + n[1]*p.b + n[2]*p.c); // |n·p.n|
              if (dot > normal_eps && fabs(d - p.d) < dist_eps) {
                  too_similar = true;
                  break;
              }
          }

          if (!too_similar)
              planes.push_back(candidate);
      }

      return planes;
  }

extern "C" void
classify_and_rate_planes(const float *planes,       // [numPlanes][4]
                         const float *points,       // [numPoints][3]
                         const unsigned int *edges, // [numEdges][2]
                         float *scores,             // [numPlanes]
                         int numPlanes, int numPoints, int numEdges);


bool decompose_iteration(Mesh &mesh, MeshList &parts, Mesh &cage,
                         OptixDeviceContext context) {

  int n_points = mesh.vertices.size();

  vector<Plane> planes =
      get_candidate_planes(mesh.vertices, mesh.intersecting_edges, 2500);

  int ss_offset = planes.size();

  for (auto &p : support_surface_planes) {
    planes.push_back(p);
  }

  if (planes.size() == 0 || mesh.intersecting_edges.size() == 0) {
    parts.push_back(mesh);
    return true;
  }

  // Prepare data for GPU
  vector<float> h_planes(planes.size() * 4);
  for (int i = 0; i < planes.size(); ++i) {
    h_planes[i * 4 + 0] = planes[i].a;
    h_planes[i * 4 + 1] = planes[i].b;
    h_planes[i * 4 + 2] = planes[i].c;
    h_planes[i * 4 + 3] = planes[i].d;
  }
  vector<float> h_points(n_points * 3);
  for (int i = 0; i < n_points; ++i) {
    h_points[i * 3 + 0] = mesh.vertices[i][0];
    h_points[i * 3 + 1] = mesh.vertices[i][1];
    h_points[i * 3 + 2] = mesh.vertices[i][2];
  }

  vector<unsigned int> h_edges(mesh.intersecting_edges.size() * 2);
  for (int i = 0; i < mesh.intersecting_edges.size(); ++i) {
    h_edges[i * 2 + 0] = mesh.intersecting_edges[i].first;
    h_edges[i * 2 + 1] = mesh.intersecting_edges[i].second;
  }

  vector<float> h_scores(planes.size(), 0);
  classify_and_rate_planes(h_planes.data(), h_points.data(), h_edges.data(),
                           h_scores.data(), planes.size(), n_points,
                           mesh.intersecting_edges.size());

  // increase all support surface scores to prioritize them
  for (int i = ss_offset; i < planes.size(); i++) {
    h_scores[i] *= config.support_surface_k;
  }

  int best_idx =
      max_element(h_scores.begin(), h_scores.end()) - h_scores.begin();
  Plane best_plane = planes[best_idx];

  int *part1_map, *part2_map;
  MeshList new_parts = clip(mesh, best_plane, part1_map, part2_map);

  if (new_parts.size() < 2) {
    // clipping failed
    delete[] part1_map;
    delete[] part2_map;
    parts.push_back(mesh);
    return false;
  }

  for (auto &edge : mesh.intersecting_edges) {
    int i1 = edge.first;
    int i2 = edge.second;

    if (part1_map[i1] && part1_map[i2]){
      pair<int, int> new_edge = make_pair(part1_map[i1]-1, part1_map[i2]-1);
      new_parts[0].intersecting_edges.push_back(new_edge);
    }
    if (part2_map[i1] && part2_map[i2]){
      pair<int, int> new_edge = make_pair(part2_map[i1]-1, part2_map[i2]-1);
      new_parts[1].intersecting_edges.push_back(new_edge);
    }
  }

  delete[] part1_map;
  delete[] part2_map;

  separate_disjoint(new_parts);

  for (auto &part : new_parts) {
    if (part.vertices.size() < 10)
      continue;

    cage = part.copy();
    manifold_preprocess(cage, 40, 0.02);

    vector<pair<unsigned int, unsigned int>> new_intersecting_edges = compute_intersection_matrix(part, cage, context);
    part.intersecting_edges.insert(part.intersecting_edges.end(),
                                   new_intersecting_edges.begin(),
                                   new_intersecting_edges.end());
    parts.push_back(part);
  }

  return false;
}

double compute_part_score(Mesh &part) {
  double score = 0.0;
  for (auto &e : part.intersecting_edges) {
    Vec3D v1 = part.vertices[e.first];
    Vec3D v2 = part.vertices[e.second];

    double len = vector_length(v2 - v1);
    score += len;
  }
  return score;
}

int get_part_with_highest_score(MeshList &parts) {
  double max_score = -1.0;
  int best_idx = -1;
  for (int i = 0; i < parts.size(); i++) {
    double score = compute_part_score(parts[i]);
    if (score > max_score) {
      max_score = score;
      best_idx = i;
    }
  }
  return best_idx;
}

int get_part_with_highest_concavity(MeshList &parts, double &max_concavity) {

  MeshList cvxs;
  for (auto &part : parts) {
    Mesh cvx;
    part.compute_ch(cvx, true);
    cvxs.push_back(cvx);
  }

  int best_idx = -1;
  for (int i = 0; i < parts.size(); i++) {
    double concavity = compute_h(parts[i], cvxs[i], 0.3, 3000, 42);
    if (concavity > max_concavity) {
      max_concavity = concavity;
      best_idx = i;
    }
  }
  return best_idx;
}

double compute_final_concavity(MeshList &parts, MeshList &cvxs) {
  double h = 0;
  for (int i = 0; i < parts.size(); i++) {
    double cur_h = compute_hb(parts[i], cvxs[i], 10000, 42);
    if (cur_h > h)
      h = cur_h;
  }
  return h;
}

ProcessResult process(Mesh mesh, double concavity, int num_parts) {

  auto log = [](const string &msg) {
    cout << "[visacd] " << msg << "\n";
    cout.flush();
  };

  support_surface_planes.clear();
  OptixDeviceContext context = createContext();
  vector<double> orig_bbox = mesh.normalize();

  log("Preprocessing mesh (" + to_string(mesh.vertices.size()) + " verts)...");
  Mesh original_mesh = mesh.copy();

  manifold_preprocess(mesh, 30, 0.55 / 30);

  if (mesh.vertices.size() > 15000){
    mesh = original_mesh.copy();
    manifold_preprocess(mesh, 20, 0.55 / 20);
  }
  log("Remeshed to " + to_string(mesh.vertices.size()) + " verts.");

  Mesh cage = mesh.copy();
  manifold_preprocess(cage, 40, 0.03);

  MeshList parts;

  if (config.use_support_surfaces) {
    vector<Surface> surfaces = extract_surfaces(mesh, config.support_surface_min_area);
    log("Detected " + to_string(surfaces.size()) + " support surface(s).");
    for (auto &S : surfaces)
      support_surface_planes.push_back(S.plane);
  }

  parts.push_back(mesh);
  separate_disjoint(parts);

  for (auto &part : parts) {
    vector<pair<unsigned int, unsigned int>> new_intersections =
        compute_intersection_matrix(part, cage, context);
    part.intersecting_edges.insert(part.intersecting_edges.end(),
                                   new_intersections.begin(),
                                   new_intersections.end());
  }

  log("Starting decomposition (max parts=" + to_string(num_parts) +
      ", concavity threshold=" + to_string(concavity) + ", mode=" + config.score_mode + ").");

  for (int i = 0; i < num_parts-1; i++) {
    int part_idx = -1;
    if (config.score_mode == "edge")
      part_idx = get_part_with_highest_score(parts);
    else if (config.score_mode == "concavity") {
      double max_concavity = -1;
      part_idx = get_part_with_highest_concavity(parts, max_concavity);
      if (max_concavity < concavity) {
        log("Step " + to_string(i+1) + ": concavity " +
            to_string(max_concavity).substr(0,6) + " < threshold, stopping.");
        break;
      }
    }

    if (part_idx == -1)
      break;

    log("Step " + to_string(i+1) + "/" + to_string(num_parts-1) +
        ": splitting part " + to_string(part_idx) +
        " (" + to_string(parts.size()) + " parts total).");

    Mesh part = parts[part_idx];
    parts.erase(parts.begin() + part_idx);
    bool flag = decompose_iteration(part, parts, cage, context);
    if (flag) {
      log("No more intersecting edges, stopping early.");
      break;
    }
  }

  log("Computing convex hulls for " + to_string(parts.size()) + " parts...");
  MeshList hulls;
  for (auto &part : parts) {
    Mesh hull;
    part.compute_ch(hull, true);
    hulls.push_back(hull);
  }

  double final_concavity = compute_final_concavity(parts, hulls);

  if (config.use_merging) {
    log("Merging parts...");
    multimerge_ch(parts, hulls, final_concavity, concavity);
  }

  optixDeviceContextDestroy(context);

  for (auto &hull : hulls)
    hull.unnormalize(orig_bbox);
  for (auto &part : parts)
    part.unnormalize(orig_bbox);

  MeshList &output = config.return_parts ? parts : hulls;
  int n = output.size();

  ostringstream summary;
  summary << "Done. parts=" << n
          << "  concavity=" << fixed << setprecision(4) << final_concavity;
  log(summary.str());

  return {output, final_concavity, n};
}

} // namespace neural_acd

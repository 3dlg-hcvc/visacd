#include <core.hpp>
#include <cost.hpp>
#include <fstream>
#include <hausdorff.hpp>
#include <iostream>

using namespace std;

namespace neural_acd {

void MergeMesh(Mesh &mesh1, Mesh &mesh2, Mesh &merge) {
  merge.vertices.insert(merge.vertices.end(), mesh1.vertices.begin(),
                        mesh1.vertices.end());
  merge.vertices.insert(merge.vertices.end(), mesh2.vertices.begin(),
                        mesh2.vertices.end());
  merge.triangles.insert(merge.triangles.end(), mesh1.triangles.begin(),
                         mesh1.triangles.end());
  int N = mesh1.vertices.size();
  for (int i = 0; i < (int)mesh2.triangles.size(); i++)
    merge.triangles.push_back({mesh2.triangles[i][0] + N,
                               mesh2.triangles[i][1] + N,
                               mesh2.triangles[i][2] + N});
}

double get_volume(Vec3D p1, Vec3D p2, Vec3D p3) {
  double v321 = p3[0] * p2[1] * p1[2];
  double v231 = p2[0] * p3[1] * p1[2];
  double v312 = p3[0] * p1[1] * p2[2];
  double v132 = p1[0] * p3[1] * p2[2];
  double v213 = p2[0] * p1[1] * p3[2];
  double v123 = p1[0] * p2[1] * p3[2];
  return (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123);
}
double get_mesh_volume(Mesh &mesh) {
  double volume = 0;
  for (int i = 0; i < (int)mesh.triangles.size(); i++) {
    int idx0 = mesh.triangles[i][0], idx1 = mesh.triangles[i][1],
        idx2 = mesh.triangles[i][2];
    volume += get_volume(mesh.vertices[idx0], mesh.vertices[idx1],
                         mesh.vertices[idx2]);
  }
  return volume;
}

double compute_rv(Mesh &cvx1, Mesh &cvx2, Mesh &cvxCH, double epsilon) {
  double v1, v2, v3;

  v1 = get_mesh_volume(cvx1);
  v2 = get_mesh_volume(cvx2);
  v3 = get_mesh_volume(cvxCH);

  // cout << "Volumes: " << v1 << ", " << v2 << ", " << v3 << endl;
  // cout << v1 + v2 - v3 << endl;
  double d = pow(3 * fabs(v1 + v2 - v3) / (4 * Pi), 1.0 / 3);

  return d;
}

double compute_rv(Mesh &tmesh1, Mesh &tmesh2, double epsilon) {
  double v1, v2;
  v1 = get_mesh_volume(tmesh1);
  v2 = get_mesh_volume(tmesh2);

  double d = pow(3 * fabs(v1 - v2) / (4 * Pi), 1.0 / 3);

  return d;
}

double compute_hb(Mesh &tmesh1, Mesh &tmesh2, unsigned int resolution,
                  bool flag) {
  vector<Vec3D> samples1, samples2;
  vector<int> sample_tri_ids1, sample_tri_ids2;

  tmesh1.extract_point_set(samples1, sample_tri_ids1, resolution, 1);
  tmesh2.extract_point_set(samples2, sample_tri_ids2, resolution, 1);

  if (!((int)samples1.size() > 0 && (int)samples2.size() > 0))
    return INF;

  double h;
  h = face_hausdorff_distance(tmesh1, samples1, sample_tri_ids1, tmesh2,
                              samples2, sample_tri_ids2);

  return h;
}

double compute_hb(Mesh &cvx1, Mesh &cvx2, Mesh &cvxCH,
                  unsigned int resolution) {
  if (cvx1.vertices.size() + cvx2.vertices.size() == cvxCH.vertices.size())
    return 0.0;
  Mesh cvx;
  vector<Vec3D> samples1, samples2;
  vector<int> sample_tri_ids1, sample_tri_ids2;
  MergeMesh(cvx1, cvx2, cvx);
  extract_point_set(cvx1, cvx2, samples1, sample_tri_ids1, resolution);
  cvxCH.extract_point_set(samples2, sample_tri_ids2, resolution, 1);

  if (!((int)samples1.size() > 0 && (int)samples2.size() > 0))
    return INF;

  double h = face_hausdorff_distance(cvx, samples1, sample_tri_ids1, cvxCH,
                                     samples2, sample_tri_ids2);

  return h;
}

double compute_h(Mesh &cvx1, Mesh &cvx2, Mesh &cvxCH, double k,
                 unsigned int resolution, double epsilon) {
  double h1 = compute_rv(cvx1, cvx2, cvxCH, epsilon);
  double h2 = compute_hb(cvx1, cvx2, cvxCH, resolution + 2000);


  return max(h1 * k, h2);
}

double compute_h(Mesh &tmesh1, Mesh &tmesh2, double k, unsigned int resolution,
                 double epsilon, bool flag) {
  double h1 = compute_rv(tmesh1, tmesh2, epsilon);
  double h2 = compute_hb(tmesh1, tmesh2, resolution, flag);

  // cout << "rv: " << h1 << ", hb: " << h2 << endl;
  return max(h1 * k, h2);
}

double mesh_dist(Mesh &ch1, Mesh &ch2) {
  vector<Vec3D> XA = ch1.vertices, XB = ch2.vertices;

  int nA = XA.size();

  PointCloud<double> cloudB;
  vec2pc(cloudB, XB);

  typedef KDTreeSingleIndexAdaptor<
      L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>,
      3 /* dim */
      >
      my_kd_tree_t;

  my_kd_tree_t indexB(3 /*dim*/, cloudB,
                      KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
  indexB.buildIndex();

  double minDist = INF;
  for (int i = 0; i < nA; i++) {
    size_t num_results = 1;

    double query_pt[3] = {XA[i][0], XA[i][1], XA[i][2]};

    vector<size_t> ret_index(num_results);
    vector<double> out_dist_sqr(num_results);

    num_results = indexB.knnSearch(&query_pt[0], num_results, &ret_index[0],
                                   &out_dist_sqr[0]);
    double dist = sqrt(out_dist_sqr[0]);
    minDist = min(minDist, dist);
  }

  return minDist;
}

} // namespace neural_acd
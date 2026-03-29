#include <algorithm>
#include <core.hpp>
#include <cstdio>
#include <iostream>
#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/util/Util.h>
#include <string>
#include <vector>

using namespace openvdb;
using namespace std;

namespace neural_acd {

void sdf_manifold(Mesh &input, Mesh &output, double scale, double level_set) {
  vector<Vec3s> points;
  vector<Vec3I> tris;
  vector<Vec4I> quads;

  for (unsigned int i = 0; i < input.vertices.size(); ++i) {
    points.push_back({(float)(input.vertices[i][0] * scale),
                      (float)(input.vertices[i][1] * scale),
                      (float)(input.vertices[i][2] * scale)});
  }
  for (unsigned int i = 0; i < input.triangles.size(); ++i) {
    tris.push_back({(unsigned int)input.triangles[i][0],
                    (unsigned int)input.triangles[i][1],
                    (unsigned int)input.triangles[i][2]});
  }

  math::Transform::Ptr xform = math::Transform::createLinearTransform();
  tools::QuadAndTriangleDataAdapter<Vec3s, Vec3I> mesh(points, tris);

  DoubleGrid::Ptr sgrid = tools::meshToSignedDistanceField<DoubleGrid>(
      *xform, points, tris, quads, level_set*scale+1.0, 3.0);

  vector<Vec3s> newPoints;
  vector<Vec3I> newTriangles;
  vector<Vec4I> newQuads;
  tools::volumeToMesh(*sgrid, newPoints, newTriangles, newQuads, level_set*scale);

  output.clear();
  for (unsigned int i = 0; i < newPoints.size(); ++i) {
    output.vertices.push_back({newPoints[i][0] / scale, newPoints[i][1] / scale,
                               newPoints[i][2] / scale});
  }
  for (unsigned int i = 0; i < newTriangles.size(); ++i) {
    output.triangles.push_back({(int)newTriangles[i][0],
                                (int)newTriangles[i][2],
                                (int)newTriangles[i][1]});
  }
  for (unsigned int i = 0; i < newQuads.size(); ++i) {
    output.triangles.push_back(
        {(int)newQuads[i][0], (int)newQuads[i][2], (int)newQuads[i][1]});
    output.triangles.push_back(
        {(int)newQuads[i][0], (int)newQuads[i][3], (int)newQuads[i][2]});
  }
}

void manifold_preprocess(Mesh &m, double scale, double level_set) {
  Mesh tmp = m;
  m.clear();
  sdf_manifold(tmp, m, scale, level_set);
}
} // namespace neural_acd
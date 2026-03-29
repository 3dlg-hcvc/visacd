#include <postprocess.hpp>
#include <core.hpp>
#include <cost.hpp>
#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <unordered_map>

using namespace std;

namespace neural_acd {

int32_t find_min_element(const std::vector<double> d, double *const m,
                         const int32_t begin, const int32_t end) {
  int32_t idx = -1;
  double min = (std::numeric_limits<double>::max)();
  for (size_t i = begin; i < size_t(end); ++i) {
    if (d[i] < min) {
      idx = i;
      min = d[i];
    }
  }

  *m = min;
  return idx;
}

void merge_ch(Mesh &ch1, Mesh &ch2, Mesh &ch) {
  Mesh merge;
  merge.vertices.insert(merge.vertices.end(), ch1.vertices.begin(),
                        ch1.vertices.end());
  merge.vertices.insert(merge.vertices.end(), ch2.vertices.begin(),
                        ch2.vertices.end());
  merge.triangles.insert(merge.triangles.end(), ch1.triangles.begin(),
                         ch1.triangles.end());
  for (int i = 0; i < (int)ch2.triangles.size(); i++)
    merge.triangles.push_back({int(ch2.triangles[i][0] + ch1.vertices.size()),
                               int(ch2.triangles[i][1] + ch1.vertices.size()),
                               int(ch2.triangles[i][2] + ch1.vertices.size())});
  merge.compute_ch(ch, true);
}

void print_cost_matrix(const vector<double> &matrix, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (j <= i) {
        size_t idx = (i * (i + 1)) >> 1;
        idx += j;
        if (matrix[idx] == INF)
          cout << "INF"
               << "\t";
        else
          cout << matrix[idx] << "\t";
      } else {
        cout << "-\t";
      }
    }
    cout << "\n";
  }
}

void multimerge_ch(MeshList &meshs, MeshList &cvxs, double current_concavity, double threshold) {
  size_t nConvexHulls = (size_t)cvxs.size();

  cout<<"Starting multimerge_ch with " << nConvexHulls << " convex hulls and threshold " << threshold << ".\n";

  if (nConvexHulls > 1) {
    int bound = ((((nConvexHulls - 1) * nConvexHulls)) >> 1);
    // Populate the cost matrix
    vector<double> costMatrix, precostMatrix;
    costMatrix.resize(bound);    // only keeps the top half of the matrix
    precostMatrix.resize(bound); // only keeps the top half of the matrix

    size_t p1, p2;
    for (int idx = 0; idx < bound; ++idx) {
      p1 = (int)(sqrt(8 * idx + 1) - 1) >>
           1; // compute nearest triangle number index
      int sum =
          (p1 * (p1 + 1)) >> 1; // compute nearest triangle number from index
      p2 = idx - sum;           // modular arithmetic from triangle number
      p1++;
      double dist = mesh_dist(cvxs[p1], cvxs[p2]);
      if (dist < threshold) {
        Mesh combinedCH;
        merge_ch(cvxs[p1], cvxs[p2], combinedCH);

        costMatrix[idx] =
            compute_h(cvxs[p1], cvxs[p2], combinedCH, 0.3, 10000, 42);
        precostMatrix[idx] = max(
            compute_h(meshs[p1], cvxs[p1], 0.3, 10000, 42),
            compute_h(meshs[p2], cvxs[p2], 0.3, 10000, 42));
      } else {
        costMatrix[idx] = INF;
      }
    }

    size_t costSize = (size_t)cvxs.size();

    while (true) {
      // Search for lowest cost
      double bestCost = INF;
      const int32_t addr = find_min_element(costMatrix, &bestCost, 0,
                                              (int32_t)costMatrix.size());
      if (addr < 0) {
        break;
      }

      // print_cost_matrix(costMatrix, costSize);


      
      // if dose not set max nConvexHull, stop the merging when bestCost is
      // larger than the threshold
      if (bestCost > threshold)
        break;
      if (bestCost+precostMatrix[addr] > current_concavity) // prevent increasing concavity
      {
        costMatrix[addr] = INF;
        continue;
      }
      

      const size_t addrI =
          (static_cast<int32_t>(sqrt(1 + (8 * addr))) - 1) >> 1;
      const size_t p1 = addrI + 1;
      const size_t p2 = addr - ((addrI * (addrI + 1)) >> 1);
      // printf("addr %ld, addrI %ld, p1 %ld, p2 %ld\n", addr, addrI, p1, p2);

      // Make the lowest cost row and column into a new hull
      Mesh cch;
      merge_ch(cvxs[p1], cvxs[p2], cch);
      cvxs[p2] = cch;

      std::swap(cvxs[p1], cvxs[cvxs.size() - 1]);
      cvxs.pop_back();

      cout<<"###########################################HULLS MERGED########################" << endl;

      costSize = costSize - 1;

      // Calculate costs versus the new hull
      size_t rowIdx = ((p2 - 1) * p2) >> 1;
      for (size_t i = 0; (i < p2); ++i) {
        double dist = mesh_dist(cvxs[p2], cvxs[i]);
        if (dist < threshold) {
          Mesh combinedCH;
          merge_ch(cvxs[p2], cvxs[i], combinedCH);
          costMatrix[rowIdx] =
              compute_h(cvxs[p2], cvxs[i], combinedCH, 0.3, 10000, 42);
          precostMatrix[rowIdx++] =
              max(precostMatrix[p2] + bestCost, precostMatrix[i]);
        } else
          costMatrix[rowIdx++] = INF;
      }

      rowIdx += p2;
      for (size_t i = p2 + 1; (i < costSize); ++i) {
        double dist = mesh_dist(cvxs[p2], cvxs[i]);
        if (dist < threshold) {
          Mesh combinedCH;
          merge_ch(cvxs[p2], cvxs[i], combinedCH);
          costMatrix[rowIdx] =
              compute_h(cvxs[p2], cvxs[i], combinedCH, 0.3, 10000, 42);
          precostMatrix[rowIdx] =
              max(precostMatrix[p2] + bestCost, precostMatrix[i]);
        } else
          costMatrix[rowIdx] = INF;
        rowIdx += i;
      }

      // Move the top column in to replace its space
      const size_t erase_idx = ((costSize - 1) * costSize) >> 1;
      if (p1 < costSize) {
        rowIdx = (addrI * p1) >> 1;
        size_t top_row = erase_idx;
        for (size_t i = 0; i < p1; ++i) {
          if (i != p2) {
            costMatrix[rowIdx] = costMatrix[top_row];
            precostMatrix[rowIdx] = precostMatrix[top_row];
          }
          ++rowIdx;
          ++top_row;
        }

        ++top_row;
        rowIdx += p1;
        for (size_t i = p1 + 1; i < costSize; ++i) {
          costMatrix[rowIdx] = costMatrix[top_row];
          precostMatrix[rowIdx] = precostMatrix[top_row++];
          rowIdx += i;
        }
      }
      costMatrix.resize(erase_idx);
      precostMatrix.resize(erase_idx);
    }
  }

}

MeshList separate_disjoint_step(Mesh &part) {
  if (part.triangles.empty()) {
    return {};
  }

  map<pair<int, int>, vector<int>> edge_map; // edge -> connected tris

  auto make_edge = [](int a, int b) {
    return std::pair<int, int>{std::min(a, b), std::max(a, b)};
  };

  // Build edge to triangle map
  for (int i = 0; i < part.triangles.size(); ++i) {
    const auto &tri = part.triangles[i];
    pair<int, int> e1 = make_edge(tri[0], tri[1]);
    pair<int, int> e2 = make_edge(tri[1], tri[2]);
    pair<int, int> e3 = make_edge(tri[0], tri[2]);

    edge_map[e1].push_back(i);
    edge_map[e2].push_back(i);
    edge_map[e3].push_back(i);
  }

  unordered_map<int, int> tri_to_part; // triangle index -> part number
  int part_num = 0;

  // Flood fill to find connected components
  for (int i = 0; i < part.triangles.size(); ++i) {
    if (tri_to_part.count(i)) {
      continue;
    }

    queue<int> q;
    q.push(i);
    tri_to_part[i] = part_num;

    while (!q.empty()) {
      int cur = q.front();
      q.pop();

      const auto &tri = part.triangles[cur];
      pair<int, int> edges[3] = {make_edge(tri[0], tri[1]),
                                 make_edge(tri[1], tri[2]),
                                 make_edge(tri[0], tri[2])};

      for (const auto &edge : edges) {
        for (int neighbor : edge_map[edge]) {
          if (neighbor != cur && !tri_to_part.count(neighbor)) {
            tri_to_part[neighbor] = part_num;
            q.push(neighbor);
          }
        }
      }
    }
    part_num++;
  }

  // Create new meshes for each part
  MeshList new_parts(part_num);
  vector<unordered_map<int, int>> vertex_remap(part_num); // global v -> part v

  for (int i = 0; i < part.triangles.size(); ++i) {
    int part_idx = tri_to_part[i];
    Mesh &current_part = new_parts[part_idx];
    const auto &tri = part.triangles[i];

    // Remap vertices
    array<int, 3> new_indices;
    for (int k = 0; k < 3; ++k) {
      int global_v = tri[k];
      auto &remap = vertex_remap[part_idx];

      if (!remap.count(global_v)) {
        remap[global_v] = current_part.vertices.size();
        current_part.vertices.push_back(part.vertices[global_v]);
        if (part.is_new.size() > 0)
          current_part.is_new.push_back(part.is_new[global_v]);
      }
      new_indices[k] = remap[global_v];
    }

    current_part.triangles.push_back(new_indices);

  }


  //update intersecting edges
  for (auto &edge : part.intersecting_edges) {
    int v1 = edge.first;
    int v2 = edge.second;

    int part_idx = -1;
    for (int p = 0; p < part_num; ++p) {
      auto &remap = vertex_remap[p];
      if (remap.count(v1) && remap.count(v2)) {
        part_idx = p;
        break;
      }
    }
    if (part_idx != -1) {
      auto &remap = vertex_remap[part_idx];
      new_parts[part_idx].intersecting_edges.push_back(
          {remap[v1], remap[v2]});
    }
  }




  return new_parts;
}

void separate_disjoint(MeshList &parts) {
  MeshList new_parts;
  for (auto &part : parts) {
    MeshList res = separate_disjoint_step(part);
    for (auto &new_part : res) {
      if (std::abs(get_mesh_volume(new_part)) < 1e-6)
        continue; // skip degenerate parts
      new_parts.push_back(new_part);
    }
  }
  parts.clear();
  parts.insert(parts.end(), new_parts.begin(), new_parts.end());
}


}
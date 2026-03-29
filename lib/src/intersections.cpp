#include <array>
#include <core.hpp>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <intersections.hpp>
#include <math.h>
#include <optixUtils.hpp>
#include <sstream>
#include <unordered_map>
#include <vector>

using namespace std;

namespace neural_acd {

// Payload struct for OptiX
struct Payload {
  int hit; // 1 if hit, 0 if not
};

// Helper: convert vector of vertices to float array
static void fill_float_array(const vector<array<double, 3>> &verts,
                             float *arr) {
  for (size_t i = 0; i < verts.size(); ++i) {
    arr[i * 3 + 0] = verts[i][0];
    arr[i * 3 + 1] = verts[i][1];
    arr[i * 3 + 2] = verts[i][2];
  }
}

// Helper: convert cage triangles to index array
static void fill_index_array(const vector<array<int, 3>> &tris, int *arr) {
  for (size_t i = 0; i < tris.size(); ++i) {
    arr[i * 3 + 0] = tris[i][0];
    arr[i * 3 + 1] = tris[i][1];
    arr[i * 3 + 2] = tris[i][2];
  }
}

inline __host__ __device__ float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float length(const float3 &v) {
  return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline __host__ __device__ float3 normalize(const float3 &v) {
  float len = length(v);
  if (len > 0.0f) {
    return make_float3(v.x / len, v.y / len, v.z / len);
  }
  return make_float3(0.0f, 0.0f, 0.0f);
}

string loadPTX(const string &filename) {
  ifstream file(filename, ios::in | ios::binary);
  if (!file) {
    throw runtime_error("Failed to open PTX file: " + filename);
  }
  ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

void linear_to_pair(long long k, long long n, long long &i, long long &j) {
  double kd = static_cast<double>(k);
  double nd = static_cast<double>(n);
  double t = sqrt(-8.0 * kd + 4.0 * nd * (nd - 1.0) - 7.0);
  i = static_cast<long long>(nd - 2.0 - floor(t / 2.0 - 0.5));
  j = static_cast<long long>(k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2);
}

vector<unsigned int> run_optix(vector<Vec3D> vertices, vector<bool> new_mask, Mesh &cage,
                               OptixDeviceContext context) {
  CUCHK(cudaDeviceSynchronize());

  const int n_points = static_cast<int>(vertices.size());
  const int n_tris = static_cast<int>(cage.triangles.size());
  const int n_verts = static_cast<int>(cage.vertices.size());

  // --- Host arrays ---
  vector<float> h_points(n_points * 3);
  fill_float_array(vertices, h_points.data());

  vector<unsigned int> h_new_mask(new_mask.size());
  for (int i = 0; i < new_mask.size(); i++) {
    h_new_mask[i] = new_mask[i] ? 1 : 0;
  }


  // Cage geometry in standard vertex/index form (OptiX triangles expect this):
  vector<float> h_vertices(n_verts * 3);
  for (int v = 0; v < n_verts; ++v) {
    h_vertices[3 * v + 0] = cage.vertices[v][0];
    h_vertices[3 * v + 1] = cage.vertices[v][1];
    h_vertices[3 * v + 2] = cage.vertices[v][2];
  }
  vector<uint3> h_indices(n_tris);
  for (int t = 0; t < n_tris; ++t) {
    h_indices[t] = make_uint3(static_cast<unsigned>(cage.triangles[t][0]),
                              static_cast<unsigned>(cage.triangles[t][1]),
                              static_cast<unsigned>(cage.triangles[t][2]));
  }
  // --- Device buffers ---
  float *d_points = nullptr;
  unsigned int *d_new_mask = nullptr;
  float *d_vertices = nullptr;
  uint3 *d_indices = nullptr;
  unsigned int *d_uM = nullptr;

  CUCHK(cudaMalloc((void **)&d_points, sizeof(float) * h_points.size()));
  CUCHK(cudaMalloc((void **)&d_new_mask, sizeof(unsigned int) * h_new_mask.size()));
  CUCHK(cudaMalloc((void **)&d_vertices, sizeof(float) * h_vertices.size()));
  CUCHK(cudaMalloc((void **)&d_indices, sizeof(uint3) * h_indices.size()));
  CUCHK(cudaMalloc((void **)&d_uM,
                   sizeof(unsigned int) * n_points * (n_points - 1) / 2));

  CUCHK(cudaMemcpy(d_points, h_points.data(), sizeof(float) * h_points.size(),
                   cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(d_new_mask, h_new_mask.data(),
                   sizeof(unsigned int) * h_new_mask.size(), cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(d_vertices, h_vertices.data(),
                   sizeof(float) * h_vertices.size(), cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(d_indices, h_indices.data(),
                   sizeof(uint3) * h_indices.size(), cudaMemcpyHostToDevice));
  CUCHK(cudaMemset(d_uM, 0,
                   sizeof(unsigned int) * n_points * (n_points - 1) / 2));

  // --- Build GAS (triangle mesh) ---
  OptixTraversableHandle gas_handle = 0;
  CUdeviceptr d_gas_output = 0;
  size_t gas_temp_size = 0, gas_output_size = 0;

  OptixBuildInput build_input = {};
  build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

  CUdeviceptr d_vertices_ptr = reinterpret_cast<CUdeviceptr>(d_vertices);
  CUdeviceptr d_indices_ptr = reinterpret_cast<CUdeviceptr>(d_indices);

  unsigned int triangle_input_flags[1] = {
      OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT}; // we’re using any-hit program, but
                                           // flag here doesn’t disable the SBT
                                           // any-hit; it's a per-primitive
                                           // flag. We'll leave it 0 actually.
  triangle_input_flags[0] = 0;

  build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  build_input.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
  build_input.triangleArray.numVertices = static_cast<unsigned int>(n_verts);
  build_input.triangleArray.vertexBuffers = &d_vertices_ptr;

  build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
  build_input.triangleArray.numIndexTriplets =
      static_cast<unsigned int>(n_tris);
  build_input.triangleArray.indexBuffer = d_indices_ptr;

  build_input.triangleArray.flags = triangle_input_flags;
  build_input.triangleArray.numSbtRecords = 1;

  OptixAccelBuildOptions accel_opts = {};
  accel_opts.buildFlags =
      OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  accel_opts.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes gas_sizes;
  OCHK(optixAccelComputeMemoryUsage(context, &accel_opts, &build_input, 1,
                                    &gas_sizes));

  CUdeviceptr d_temp;
  CUCHK(cudaMalloc((void **)&d_temp, gas_sizes.tempSizeInBytes));
  CUCHK(cudaMalloc((void **)&d_gas_output, gas_sizes.outputSizeInBytes));

  OCHK(optixAccelBuild(context, 0, &accel_opts, &build_input, 1, d_temp,
                       gas_sizes.tempSizeInBytes, d_gas_output,
                       gas_sizes.outputSizeInBytes, &gas_handle, nullptr, 0));

  CUCHK(cudaFree((void *)d_temp));

  // --- Module / Pipeline from PTX ---
  // Find ray_segments.ptx next to the .so at runtime, fall back to build dir
  auto get_ptx_path = []() -> string {
    Dl_info info;
    if (dladdr((void *)&loadPTX, &info) && info.dli_fname) {
      auto p = filesystem::path(info.dli_fname).parent_path() / "ray_segments.ptx";
      if (filesystem::exists(p))
        return p.string();
    }
    return string(PTX_DIR) + "/ray_segments.ptx";
  };
  string ptx = loadPTX(get_ptx_path());

  OptixModule module = 0;
  OptixPipeline pipeline = 0;
  OptixProgramGroup raygen_pg = 0, miss_pg = 0, hit_pg = 0;

  OptixModuleCompileOptions mopts = {};
  mopts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

  OptixPipelineCompileOptions pcomp = {};
  pcomp.usesMotionBlur = 0;
  pcomp.traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; // alias may be
                                                     // OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS
  pcomp.numPayloadValues = 1;   // we pass a single uint payload
  pcomp.numAttributeValues = 2; // default for triangles
  pcomp.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pcomp.pipelineLaunchParamsVariableName = nullptr;

  char log[2048];
  size_t log_size = sizeof(log);
  OCHK(optixModuleCreate(context, &mopts, &pcomp, ptx.c_str(), ptx.size(), log,
                         &log_size, &module));

  // Program groups
  OptixProgramGroupOptions pg_opts = {};
  OptixProgramGroupDesc rg_desc = {};
  rg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  rg_desc.raygen.module = module;
  rg_desc.raygen.entryFunctionName = "__raygen__rg";
  log_size = sizeof(log);
  OCHK(optixProgramGroupCreate(context, &rg_desc, 1, &pg_opts, log, &log_size,
                               &raygen_pg));

  OptixProgramGroupDesc ms_desc = {};
  ms_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  ms_desc.miss.module = module;
  ms_desc.miss.entryFunctionName = "__miss__ms";
  log_size = sizeof(log);
  OCHK(optixProgramGroupCreate(context, &ms_desc, 1, &pg_opts, log, &log_size,
                               &miss_pg));

  OptixProgramGroupDesc hg_desc = {};
  hg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hg_desc.hitgroup.moduleCH = nullptr; // no closest-hit
  hg_desc.hitgroup.entryFunctionNameCH = nullptr;
  hg_desc.hitgroup.moduleAH = module; // any-hit only
  hg_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
  hg_desc.hitgroup.moduleIS = nullptr; // built-in triangles
  hg_desc.hitgroup.entryFunctionNameIS = nullptr;
  log_size = sizeof(log);
  OCHK(optixProgramGroupCreate(context, &hg_desc, 1, &pg_opts, log, &log_size,
                               &hit_pg));

  // Link pipeline
  OptixProgramGroup groups[] = {raygen_pg, miss_pg, hit_pg};
  OptixPipelineLinkOptions link_opts = {};
  link_opts.maxTraceDepth = 1;
  log_size = sizeof(log);
  OCHK(optixPipelineCreate(context, &pcomp, &link_opts, groups, 3, log,
                           &log_size, &pipeline));

  // --- SBT ---
  SbtRecord<RayGenData> rg_rec = {};
  OCHK(optixSbtRecordPackHeader(raygen_pg, &rg_rec));
  rg_rec.data.points = d_points;
  rg_rec.data.new_mask = d_new_mask;
  rg_rec.data.n_points = n_points;
  rg_rec.data.has_mask = (new_mask.size() > 0) ? 1 : 0;
  rg_rec.data.uM = d_uM;
  rg_rec.data.gas = gas_handle;

  SbtRecord<MissData> ms_rec = {};
  OCHK(optixSbtRecordPackHeader(miss_pg, &ms_rec));

  SbtRecord<HitgroupData> hg_rec = {};
  OCHK(optixSbtRecordPackHeader(hit_pg, &hg_rec));
  hg_rec.data.vertices = d_vertices;
  hg_rec.data.indices = d_indices;

  CUdeviceptr d_rg_rec, d_ms_rec, d_hg_rec;
  CUCHK(cudaMalloc((void **)&d_rg_rec, sizeof(rg_rec)));
  CUCHK(cudaMalloc((void **)&d_ms_rec, sizeof(ms_rec)));
  CUCHK(cudaMalloc((void **)&d_hg_rec, sizeof(hg_rec)));

  CUCHK(cudaMemcpy((void *)d_rg_rec, &rg_rec, sizeof(rg_rec),
                   cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy((void *)d_ms_rec, &ms_rec, sizeof(ms_rec),
                   cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy((void *)d_hg_rec, &hg_rec, sizeof(hg_rec),
                   cudaMemcpyHostToDevice));

  OptixShaderBindingTable sbt = {};
  sbt.raygenRecord = d_rg_rec;
  sbt.missRecordBase = d_ms_rec;
  sbt.missRecordStrideInBytes = sizeof(SbtRecord<MissData>);
  sbt.missRecordCount = 1;
  sbt.hitgroupRecordBase = d_hg_rec;
  sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<HitgroupData>);
  sbt.hitgroupRecordCount = 1;

  unsigned int n_segments = n_points * (n_points - 1) / 2;

  // --- Launch: fully parallel (n_points x n_points) ---
  OCHK(optixLaunch(pipeline, 0 /* CUDA stream */, 0 /* params = nullptr */,
                   0 /* sizeof params */, &sbt, n_segments, 1, 1));
  CUCHK(cudaDeviceSynchronize());

  // --- Fetch result ---
  vector<unsigned int> h_uM(n_points * (n_points - 1) / 2);
  CUCHK(cudaMemcpy(h_uM.data(), d_uM, sizeof(unsigned int) * h_uM.size(),
                   cudaMemcpyDeviceToHost));

  // Cleanup OptiX objects
  OCHK(optixPipelineDestroy(pipeline));
  OCHK(optixProgramGroupDestroy(raygen_pg));
  OCHK(optixProgramGroupDestroy(miss_pg));
  OCHK(optixProgramGroupDestroy(hit_pg));
  OCHK(optixModuleDestroy(module));
  // GAS buffers can be kept if you reuse; free for now:
  CUCHK(cudaFree((void *)d_gas_output));

  // Cleanup buffers
  CUCHK(cudaFree(d_points));
  CUCHK(cudaFree(d_vertices));
  CUCHK(cudaFree(d_indices));
  CUCHK(cudaFree(d_uM));
  CUCHK(cudaFree((void *)d_rg_rec));
  CUCHK(cudaFree((void *)d_ms_rec));
  CUCHK(cudaFree((void *)d_hg_rec));

  return h_uM;
}

vector<pair<unsigned int, unsigned int>> compute_intersection_matrix(Mesh &mesh, Mesh &cage,
                                                 OptixDeviceContext &context) {

  cout<<"Running optix intersection test...\n";

  // mesh.is_new = vector<bool>(0, true);
  long long total_new = 0;
  for (auto flag : mesh.is_new) {
    if (flag)
      total_new++;
  }
  cout << "Total new vertices: " << total_new << "/"
       << mesh.vertices.size() << endl;

  vector<unsigned int> M = run_optix( mesh.vertices,mesh.is_new, cage, context);

  vector<unsigned int> self_intersect =
      run_optix(mesh.vertices, mesh.is_new, mesh, context);

  cout<<"Optix intersection test completed.\n";

  long long self_intersections = 0, cage_intersections = 0;

  int two = 0;
  for (int i = 0; i < mesh.vertices.size() * (mesh.vertices.size() - 1) / 2; i++) {
    if (self_intersect[i] == 1)
      self_intersections++;
    if (M[i] == 1)
      cage_intersections++;

    if (M[i] == 2 || self_intersect[i] == 2){
      two++;
    }
  }

  cout << "Total self-intersections: " << self_intersections << endl;
  cout << "Total cage intersections: " << cage_intersections << endl;
  cout << "Total not updated: " << two << endl;

  vector<pair<unsigned int, unsigned int>> intersecting_edges;
  for (int i = 0; i < mesh.vertices.size() * (mesh.vertices.size() - 1) / 2; i++) {
    if (M[i] == 2 || self_intersect[i] == 2) // wasn't updated
      continue;
    if (self_intersect[i])
      M[i] = 0;
    if (M[i]){
      long long x, y;
      linear_to_pair(i, mesh.vertices.size(), x, y);
      // x = vertex_map[x];
      // y = vertex_map[y];
      intersecting_edges.push_back({x, y});
    }
  }

  cout << "Total new intersecting edges: " << intersecting_edges.size() << endl;

  return intersecting_edges;

}

} // namespace neural_acd
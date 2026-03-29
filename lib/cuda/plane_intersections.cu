#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <iostream>

// Kernel
__global__ void plane_point_side_and_rate_kernel(
    const float *planes,        // [numPlanes * 4]
    const float *points,        // [numPoints * 3]
    const unsigned int *edges,  // [numEdges * 2]
    float *scores,              // [numPlanes]
    int numPlanes,
    int numPoints,
    int numEdges,
    float eps = 1e-6f
) {
    unsigned long long bidx = (unsigned long long)blockIdx.x;
    unsigned long long bdim = (unsigned long long)blockDim.x;
    unsigned long long tidx = (unsigned long long)threadIdx.x;
    unsigned long long tid = bidx * bdim + tidx;
    unsigned long long totalPairs = (unsigned long long)numPlanes * numEdges;
    if (tid >= totalPairs) return;

    int planeIdx = tid / numEdges;
    int edgeIdx  = tid % numEdges;

    // Load plane coefficients
    const float *pl = &planes[planeIdx * 4];

    // Load edge vertices
    unsigned int v1 = edges[edgeIdx * 2 + 0];
    unsigned int v2 = edges[edgeIdx * 2 + 1];

    if (v1 >= (unsigned int)numPoints || v2 >= (unsigned int)numPoints) {
        return; // Safety check to avoid OOB
    }

    // Load points
    float3 p1 = make_float3(points[3 * v1 + 0], points[3 * v1 + 1], points[3 * v1 + 2]);
    float3 p2 = make_float3(points[3 * v2 + 0], points[3 * v2 + 1], points[3 * v2 + 2]);

    // Evaluate plane equation
    float val1 = pl[0] * p1.x + pl[1] * p1.y + pl[2] * p1.z + pl[3];
    float val2 = pl[0] * p2.x + pl[1] * p2.y + pl[2] * p2.z + pl[3];

    int side1 = (val1 >  eps) - (val1 < -eps);  // +1, 0, -1
    int side2 = (val2 >  eps) - (val2 < -eps);

    // If both clearly on same side, ignore
    if (side1 != 0 && side1 == side2) {
        return;
    }

    // If either point is very close to plane, treat as same side
    if (side1 == 0 || side2 == 0) {
        return;
    }

    // Edge length
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float dz = p2.z - p1.z;
    float edge_len = sqrtf(dx * dx + dy * dy + dz * dz);

    // Atomic add to score
    atomicAdd(&scores[planeIdx], edge_len);
}


// Host wrapper
extern "C"
void classify_and_rate_planes(
    const float *planes,        // [numPlanes][4]
    const float *points,        // [numPoints][3]
    const unsigned int *edges,  // [numEdges][2]
    float *scores,              // [numPlanes]
    int numPlanes, int numPoints, int numEdges)
{
    size_t planes_bytes = sizeof(float) * 4 * numPlanes;
    size_t points_bytes = sizeof(float) * 3 * numPoints;
    size_t edges_bytes  = sizeof(unsigned int) * 2 * numEdges;
    size_t scores_bytes = sizeof(float) * numPlanes;

    float *d_planes;
    float *d_points;
    unsigned int *d_edges;
    float *d_scores;

    cudaMalloc(&d_planes, planes_bytes);
    cudaMalloc(&d_points, points_bytes);
    cudaMalloc(&d_edges, edges_bytes);
    cudaMalloc(&d_scores, scores_bytes);

    cudaMemcpy(d_planes, planes, planes_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_points, points, points_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, edges, edges_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_scores, 0, scores_bytes);

    unsigned long long totalPairs = (unsigned long long)numPlanes * numEdges;
    int block = 256;
    unsigned long long grid64 = (totalPairs + block - 1ULL) / block;
    if (grid64 > INT_MAX) {
        std::cerr << "Error: grid size too large!" << std::endl;
        cudaFree(d_planes);
        cudaFree(d_points);
        cudaFree(d_edges);
        cudaFree(d_scores);
        return;
    }
    int grid = (int)grid64;

    plane_point_side_and_rate_kernel<<<grid, block>>>(
        d_planes, d_points, d_edges,
        d_scores, numPlanes, numPoints, numEdges
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(scores, d_scores, scores_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_planes);
    cudaFree(d_points);
    cudaFree(d_edges);
    cudaFree(d_scores);
}

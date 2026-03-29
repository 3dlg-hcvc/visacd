#include <optix.h>
#include <cuda_runtime.h>

struct RayGenData
{
    const float *points; // [n_points*3]
    const unsigned int *new_mask;
    long long n_points;
    unsigned int has_mask;
    unsigned int *uM; //[n_points*(n_points-1)/2] // output adjacency matrix (upper triangular, row-major)
    OptixTraversableHandle gas;
};

struct HitgroupData
{
    const float *vertices; // float3 array
    const uint3 *indices;  // triangle indices
};

struct MissData
{
};

// --------------- Utilities ---------------
static __forceinline__ __device__ void set_payload_u32(unsigned int p0)
{
    optixSetPayload_0(p0);
}
static __forceinline__ __device__ unsigned int get_payload_u32()
{
    return optixGetPayload_0();
}
static __forceinline__ __device__ float3 ld3(const float *a, long long idx)
{
    const float *p = a + 3 * idx;
    return make_float3(p[0], p[1], p[2]);
}

static __forceinline__ __device__ void linear_to_pair(long long k, long long n, long long &i, long long &j)
{
    double kd = static_cast<double>(k);
    double nd = static_cast<double>(n);
    double t = sqrt(-8.0 * kd + 4.0 * nd * (nd - 1.0) - 7.0);
    i = static_cast<long long>(nd - 2.0 - floor(t / 2.0 - 0.5));
    j = static_cast<long long>(k + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2);
}

// --------------- Raygen ---------------
extern "C" __global__ void __raygen__rg()
{
    const RayGenData *rg = reinterpret_cast<const RayGenData *>(optixGetSbtDataPointer());

    unsigned long long launch_idx = optixGetLaunchIndex().x;

    long long i, j;
    linear_to_pair(launch_idx, rg->n_points, i, j);

    if (i >= rg->n_points || j >= rg->n_points)
        return;

    // If new is empty, this mesh is initial
    if (rg->has_mask)
    {
        // Skip if neither point is new
        if (rg->new_mask[i] == 0 && rg->new_mask[j] == 0)
        {
            rg->uM[launch_idx] = 2u;
            return;
        }
        // //Skip if both points are new (avoids bad intersecting edges)
        // if (rg->new_mask[i] == 1 && rg->new_mask[j] == 1){
        //     rg->uM[launch_idx] = 2u;
        //     return;
        // }
    }

    const float3 p = ld3(rg->points, i);
    const float3 q = ld3(rg->points, j);

    const float3 dir_raw = make_float3(q.x - p.x, q.y - p.y, q.z - p.z);
    const float seg_len = sqrtf(dir_raw.x * dir_raw.x + dir_raw.y * dir_raw.y + dir_raw.z * dir_raw.z);

    // Diagonal (i == j): no segment length; write 0 and skip.
    if (seg_len == 0.0f)
    {
        rg->uM[launch_idx] = 0u;
        return;
    }

    const float inv_len = 1.0f / seg_len;
    const float3 dir = make_float3(dir_raw.x * inv_len, dir_raw.y * inv_len, dir_raw.z * inv_len);

    unsigned int payload = 0u; // 0 = no hit, 1 = hit
    unsigned int p0 = payload;

    optixTrace(
        rg->gas,
        p, dir,
        0.001f,  // tmin - avoid self-intersection
        seg_len, // tmax: limit to segment
        0.0f,    // rayTime
        0xFF,    // visibility mask
        OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT |
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, // we only need first hit
        0,                                         // SBT offset
        1,                                         // SBT stride
        0,                                         // missSBTIndex
        p0);

    rg->uM[launch_idx] = p0;
}


// --------------- Miss: write 0 ---------------
extern "C" __global__ void __miss__ms()
{
    // payload remains 0
}

// --------------- Any-hit: write 1 & terminate ---------------
extern "C" __global__ void __anyhit__ah()
{
    set_payload_u32(1u);
    optixTerminateRay();
}

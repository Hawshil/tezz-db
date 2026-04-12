/**
 * @file gpu_asof.cu
 * @brief CUDA kernel implementing ASOF join via per-thread binary search.
 *
 * Each thread handles one left row and binary-searches the right timestamp
 * array for the rightmost entry with right_ts[j] <= left_ts[i].
 *
 * KNOWN LIMITATION: with use_key=true, right array must be pre-sorted by
 * (key, ts). Caller is responsible for pre-sorting. The CPU fallback in
 * AsofJoinNode handles the general (unsorted) case.
 */
#include "gpu_asof.cuh"

namespace gpudb {

__global__ void gpu_asof_kernel(
    const int64_t* left_ts,  int n_left,
    const int64_t* right_ts, int n_right,
    const int32_t* left_key, const int32_t* right_key,
    bool use_key, int64_t tol_ns, int* out_idx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_left) return;

    int64_t lt = left_ts[i];
    int32_t lk = use_key ? left_key[i] : 0;

    // Binary search for the rightmost j with right_ts[j] <= lt
    int lo = 0, hi = n_right - 1, best = -1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (right_ts[mid] <= lt) {
            // Candidate — check key match if required
            if (!use_key || right_key[mid] == lk)
                best = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    // Apply tolerance filter
    if (best != -1 && tol_ns > 0 && (lt - right_ts[best]) > tol_ns)
        best = -1;

    out_idx[i] = best;
}

void gpuAsofJoin(const std::int64_t* d_left_ts,  int n_left,
                 const std::int64_t* d_right_ts,  int n_right,
                 const std::int32_t* d_left_key,
                 const std::int32_t* d_right_key,
                 bool               use_key,
                 std::int64_t       tolerance_ns,
                 int*               d_out_right_idx)
{
    if (n_left == 0) return;

    const int BLK = 256;
    int grid = (n_left + BLK - 1) / BLK;
    gpu_asof_kernel<<<grid, BLK>>>(
        d_left_ts, n_left, d_right_ts, n_right,
        d_left_key, d_right_key, use_key, tolerance_ns,
        d_out_right_idx);
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace gpudb

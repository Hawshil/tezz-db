/**
 * @file hip_pipeline.hip
 * @brief HIP port of gpu_pipeline.cu — double-buffered async pipeline.
 *
 * CUDA → HIP mapping used here:
 *   cudaStream_t      → hipStream_t
 *   cudaEvent_t       → hipEvent_t
 *   cudaHostAlloc     → hipHostMalloc
 *   cudaStreamCreate  → hipStreamCreate
 *   cudaEventCreate   → hipEventCreate
 *   cudaEventRecord   → hipEventRecord
 *   cudaEventElapsedTime → hipEventElapsedTime
 *   cudaMemcpyAsync   → hipMemcpyAsync
 *
 * Consumer RDNA (RX series) vs Datacenter CDNA (MI series)
 * ─────────────────────────────────────────────────────────
 * • RDNA GPUs have smaller VRAM (8-24 GB) and PCIe Gen4 x16 (~32 GB/s).
 *   500M rows (6 GB) may exceed VRAM → chunked pipeline is essential.
 * • CDNA/MI series has Infinity Fabric with ~900 GB/s aggregate bandwidth
 *   and 32-128 GB HBM. The pipeline will be compute-bound, not PCIe-bound.
 * • RDNA runs wavefronts of 32 (wave32 mode) by default; use -mwavefrontsize64
 *   to force wave64 if needed for consistency with CDNA kernels.
 */
#include "hip_utils.h"
#include "hip_ops.h"
#include <algorithm>
#include <cstdio>
#include <vector>

namespace gpudb {

// Declared in hip_groupby.hip
extern __global__ void hip_groupby_sum_kernel(
    const int* keys, const double* vals, int n,
    int* ht_keys, double* ht_vals, int ht_cap);

void hipRunAsyncPipeline(const double* h_vals, const int* h_keys,
                         int total_rows, int chunk_size, int num_groups) {
    int num_chunks = (total_rows + chunk_size - 1) / chunk_size;
    int ht_cap = num_groups * 4;

    hipStream_t streams[2];
    HIP_CHECK(hipStreamCreate(&streams[0]));
    HIP_CHECK(hipStreamCreate(&streams[1]));

    HipBuffer<double> d_vals[2] = { HipBuffer<double>(chunk_size),
                                     HipBuffer<double>(chunk_size) };
    HipBuffer<int>    d_keys[2] = { HipBuffer<int>(chunk_size),
                                     HipBuffer<int>(chunk_size) };

    HipBuffer<int>    d_ht_keys(ht_cap);
    HipBuffer<double> d_ht_vals(ht_cap);
    d_ht_keys.fill(0xFF);
    d_ht_vals.memsetZero();

    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    float total_copy_ms = 0, total_kern_ms = 0;
    HIP_CHECK(hipEventRecord(ev_start));

    for (int c = 0; c < num_chunks; ++c) {
        int s = c % 2;
        int off = c * chunk_size;
        int n = std::min(chunk_size, total_rows - off);
        int blk = 256, grd = std::min((n + blk - 1) / blk, 2048);

        hipEvent_t ec0, ec1, ek0, ek1;
        HIP_CHECK(hipEventCreate(&ec0)); HIP_CHECK(hipEventCreate(&ec1));
        HIP_CHECK(hipEventCreate(&ek0)); HIP_CHECK(hipEventCreate(&ek1));

        HIP_CHECK(hipEventRecord(ec0, streams[s]));
        HIP_CHECK(hipMemcpyAsync(d_vals[s].data(), h_vals + off,
                                  n * sizeof(double), hipMemcpyHostToDevice, streams[s]));
        HIP_CHECK(hipMemcpyAsync(d_keys[s].data(), h_keys + off,
                                  n * sizeof(int), hipMemcpyHostToDevice, streams[s]));
        HIP_CHECK(hipEventRecord(ec1, streams[s]));

        HIP_CHECK(hipEventRecord(ek0, streams[s]));
        hip_groupby_sum_kernel<<<grd, blk, 0, streams[s]>>>(
            d_keys[s].data(), d_vals[s].data(), n,
            d_ht_keys.data(), d_ht_vals.data(), ht_cap);
        HIP_CHECK(hipEventRecord(ek1, streams[s]));

        HIP_CHECK(hipEventSynchronize(ec1));
        HIP_CHECK(hipEventSynchronize(ek1));
        float cpy, krn;
        HIP_CHECK(hipEventElapsedTime(&cpy, ec0, ec1));
        HIP_CHECK(hipEventElapsedTime(&krn, ek0, ek1));
        total_copy_ms += cpy; total_kern_ms += krn;

        hipEventDestroy(ec0); hipEventDestroy(ec1);
        hipEventDestroy(ek0); hipEventDestroy(ek1);
    }

    HIP_CHECK(hipEventRecord(ev_stop));
    HIP_CHECK(hipEventSynchronize(ev_stop));
    float wall_ms;
    HIP_CHECK(hipEventElapsedTime(&wall_ms, ev_start, ev_stop));

    std::vector<int> hk(ht_cap); std::vector<double> hv(ht_cap);
    HIP_CHECK(hipMemcpy(hk.data(), d_ht_keys.data(), ht_cap*4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hv.data(), d_ht_vals.data(), ht_cap*8, hipMemcpyDeviceToHost));
    int groups_found = 0;
    for (int i = 0; i < ht_cap; ++i) if (hk[i] != -1) ++groups_found;

    double data_gb = (double)total_rows * (sizeof(double)+sizeof(int)) / 1e9;
    std::printf("  ┌── HIP Async Pipeline Report ─────────────────┐\n");
    std::printf("  │  Chunks:  %d × %dM rows                      \n",
                num_chunks, chunk_size / 1000000);
    std::printf("  │  Σ PCIe copy time : %8.1f ms                \n", total_copy_ms);
    std::printf("  │  Σ Kernel time    : %8.1f ms                \n", total_kern_ms);
    std::printf("  │  Wall-clock time  : %8.1f ms                \n", wall_ms);
    std::printf("  │  PCIe utilisation : %5.1f%%                   \n",
                total_copy_ms / wall_ms * 100.0);
    std::printf("  │  Effective BW     : %5.1f GB/s               \n",
                data_gb / (wall_ms / 1000.0));
    std::printf("  │  Groups found     : %d                       \n", groups_found);
    std::printf("  └───────────────────────────────────────────────┘\n");
    if (total_kern_ms > total_copy_ms)
        std::printf("  → Pipeline is COMPUTE-BOUND.\n");
    else
        std::printf("  → Pipeline is PCIe-BOUND.\n");

    for (int s = 0; s < 2; ++s) HIP_CHECK(hipStreamDestroy(streams[s]));
    hipEventDestroy(ev_start); hipEventDestroy(ev_stop);
}

} // namespace gpudb

/**
 * @file hip_utils.h
 * @brief HIP equivalents of cuda_utils.cuh — RAII buffers, context, error check.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 *  CUDA → HIP API Mapping (key calls used in this project)
 * ═══════════════════════════════════════════════════════════════════════════════
 *  cudaMalloc          → hipMalloc              (identical semantics)
 *  cudaFree            → hipFree
 *  cudaMemcpy          → hipMemcpy
 *  cudaMemcpyAsync     → hipMemcpyAsync
 *  cudaMemset          → hipMemset
 *  cudaHostAlloc       → hipHostMalloc          (note: different name)
 *  cudaFreeHost        → hipHostFree            (note: different name)
 *  cudaStream_t        → hipStream_t
 *  cudaEvent_t         → hipEvent_t
 *  cudaStreamCreate    → hipStreamCreate
 *  cudaEventCreate     → hipEventCreate
 *  cudaEventRecord     → hipEventRecord
 *  cudaEventElapsedTime→ hipEventElapsedTime
 *  cudaDeviceSynchronize → hipDeviceSynchronize
 *  cudaGetDeviceProperties → hipGetDeviceProperties
 *  cudaGetErrorString  → hipGetErrorString
 *  __syncthreads()     → __syncthreads()        (identical)
 *  atomicCAS           → atomicCAS              (identical)
 *  atomicAdd           → atomicAdd              (identical on GFX9+)
 *  __shfl_down_sync(mask, val, off) → __shfl_down(val, off, warpSize)
 *      NOTE: HIP has no explicit mask — all lanes always participate.
 *  thrust::exclusive_scan → rocprim::exclusive_scan  (temp-storage pattern)
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 *  Wavefront (64) vs Warp (32) — Kernel Impact
 * ═══════════════════════════════════════════════════════════════════════════════
 *  AMD CDNA (MI100/MI200/MI300):  wavefront = 64 threads
 *  AMD RDNA (RX 6000/7000/9000):  wavefront = 32 (native), 64 (wave64 mode)
 *
 *  Effects on kernel code:
 *  1. Shared-memory reductions must reduce down to wavefront size (64, not 32)
 *     before switching to shuffle instructions.
 *  2. __shfl_down width defaults to warpSize (64 on CDNA), so the final
 *     reduction loop needs 6 steps (64→32→16→8→4→2→1) instead of 5.
 *  3. Shared memory usage per block doubles for the final-wavefront load
 *     (e.g., need ≥128 elements if blockDim=256 and wavefront=64).
 *  4. Occupancy: each wavefront uses 64 VGPRs minimum, so register pressure
 *     is higher. Use fewer registers per thread for better occupancy.
 *  5. Use warpSize built-in (runtime) or __AMDGCN_WAVEFRONT_SIZE (compile-time)
 *     to write portable code across CDNA and RDNA.
 */
#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <utility>

namespace gpudb {

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        std::fprintf(stderr, "HIP error at %s:%d — %s\n", \
                     __FILE__, __LINE__, hipGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// ── HipBuffer<T> — RAII device memory ───────────────────────────────────────
template<typename T>
class HipBuffer {
public:
    HipBuffer() = default;
    explicit HipBuffer(std::size_t n) : size_(n) {
        HIP_CHECK(hipMalloc(&ptr_, n * sizeof(T)));
    }
    ~HipBuffer() { if (ptr_) hipFree(ptr_); }
    HipBuffer(const HipBuffer&) = delete;
    HipBuffer& operator=(const HipBuffer&) = delete;
    HipBuffer(HipBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_) {
        o.ptr_ = nullptr; o.size_ = 0;
    }
    HipBuffer& operator=(HipBuffer&& o) noexcept {
        if (this != &o) { if (ptr_) hipFree(ptr_);
            ptr_ = o.ptr_; size_ = o.size_; o.ptr_ = nullptr; o.size_ = 0; }
        return *this;
    }
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return size_; }
    std::size_t bytes() const { return size_ * sizeof(T); }
    void memsetZero() { HIP_CHECK(hipMemset(ptr_, 0, bytes())); }
    void fill(int v)  { HIP_CHECK(hipMemset(ptr_, v, bytes())); }
private:
    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

// ── HipPinnedBuffer<T> — RAII pinned host memory ────────────────────────────
template<typename T>
class HipPinnedBuffer {
public:
    HipPinnedBuffer() = default;
    explicit HipPinnedBuffer(std::size_t n) : size_(n) {
        HIP_CHECK(hipHostMalloc(&ptr_, n * sizeof(T), hipHostMallocDefault));
    }
    ~HipPinnedBuffer() { if (ptr_) hipHostFree(ptr_); }
    HipPinnedBuffer(const HipPinnedBuffer&) = delete;
    HipPinnedBuffer& operator=(const HipPinnedBuffer&) = delete;
    HipPinnedBuffer(HipPinnedBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_) {
        o.ptr_ = nullptr; o.size_ = 0;
    }
    HipPinnedBuffer& operator=(HipPinnedBuffer&& o) noexcept {
        if (this != &o) { if (ptr_) hipHostFree(ptr_);
            ptr_ = o.ptr_; size_ = o.size_; o.ptr_ = nullptr; o.size_ = 0; }
        return *this;
    }
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return size_; }
private:
    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

// ── HipContext ──────────────────────────────────────────────────────────────
class HipContext {
public:
    explicit HipContext(int device = 0) : device_(device) {
        HIP_CHECK(hipSetDevice(device));
        HIP_CHECK(hipGetDeviceProperties(&props_, device));
    }
    void printDeviceInfo() const {
        std::printf("\n  ┌──────────────────────────────────────────────┐\n");
        std::printf("  │  AMD GPU: %-34s│\n", props_.name);
        std::printf("  ├──────────────────────────────────────────────┤\n");
        std::printf("  │  GCN arch name     : %-23s│\n", props_.gcnArchName);
        std::printf("  │  Compute units     : %-22d │\n", props_.multiProcessorCount);
        std::printf("  │  Wavefront size    : %-22d │\n", props_.warpSize);
        std::printf("  │  Total VRAM        : %-6.0f MB             │\n", props_.totalGlobalMem/1e6);
        std::printf("  │  Max threads/block : %-22d │\n", props_.maxThreadsPerBlock);
        std::printf("  │  Clock rate        : %-6d MHz            │\n", props_.clockRate/1000);
        std::printf("  └──────────────────────────────────────────────┘\n\n");
    }
    template<typename T>
    void copyToDevice(HipBuffer<T>& dst, const T* src, std::size_t n,
                      hipStream_t s = 0) const {
        HIP_CHECK(hipMemcpyAsync(dst.data(), src, n*sizeof(T), hipMemcpyHostToDevice, s));
    }
    template<typename T>
    void copyToHost(T* dst, const HipBuffer<T>& src, std::size_t n,
                    hipStream_t s = 0) const {
        HIP_CHECK(hipMemcpyAsync(dst, src.data(), n*sizeof(T), hipMemcpyDeviceToHost, s));
    }
    int cus() const { return props_.multiProcessorCount; }
    int wavefrontSize() const { return props_.warpSize; }
private:
    hipDeviceProp_t props_{};
    int device_;
};

} // namespace gpudb

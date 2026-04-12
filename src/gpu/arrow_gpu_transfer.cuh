/**
 * @file arrow_gpu_transfer.cuh
 * @brief Zero-copy and UVM strategies for Arrow buffer → GPU transfer.
 *
 * Three Transfer Strategies
 * ─────────────────────────
 * (a) Explicit cudaMemcpy: copy Arrow buffer to device VRAM.
 *     - Best for compute-heavy kernels (filter, join) where kernel time >> copy.
 *     - Data lives in fast HBM/GDDR; full memory bandwidth available.
 *
 * (b) Zero-copy from pinned host: cudaHostRegister Arrow buffer, GPU reads
 *     directly over PCIe. No copy at all.
 *     - Best for large datasets that exceed VRAM, or light-read kernels.
 *     - Limited by PCIe bandwidth (~32 GB/s Gen4 vs ~900 GB/s HBM).
 *     - Wins when dataset >> VRAM or kernel accesses each element only once.
 *
 * (c) UVM with prefetch: cudaMallocManaged + cudaMemPrefetchAsync moves pages
 *     on demand. Driver migrates hot pages to GPU transparently.
 *     - Best for irregular access patterns or when dataset size is unknown.
 *     - Overhead from page faults on first access; prefetch mitigates this.
 *     - Wins when access pattern is unpredictable or mixed CPU/GPU.
 *
 * When each wins (rule of thumb):
 *   Data < VRAM && compute-heavy     → (a) explicit copy
 *   Data > VRAM || single-pass scan  → (b) zero-copy
 *   Mixed CPU/GPU or unknown pattern → (c) UVM
 */
#pragma once

#include <cstring>
#include "cuda_utils.cuh"
#include <cstddef>

namespace gpudb {

// Strategy (a): Explicit copy to VRAM
template<typename T>
GpuBuffer<T> arrowToDevice_Copy(const T* arrow_ptr, std::size_t n) {
    GpuBuffer<T> d(n);
    CUDA_CHECK(cudaMemcpy(d.data(), arrow_ptr, n * sizeof(T),
                           cudaMemcpyHostToDevice));
    return d;
}

// Strategy (b): Zero-copy — register Arrow buffer as pinned, return device ptr
template<typename T>
T* arrowToDevice_ZeroCopy(T* arrow_ptr, std::size_t n) {
    CUDA_CHECK(cudaHostRegister(arrow_ptr, n * sizeof(T),
                                 cudaHostRegisterDefault));
    T* d_ptr = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_ptr, arrow_ptr, 0));
    return d_ptr;  // GPU reads directly over PCIe
}

inline void arrowUnregister(void* ptr) {
    CUDA_CHECK(cudaHostUnregister(ptr));
}

// Strategy (c): UVM with prefetch
template<typename T>
T* arrowToDevice_UVM(const T* arrow_ptr, std::size_t n, int device = 0) {
    T* uvm_ptr = nullptr;
    CUDA_CHECK(cudaMallocManaged(&uvm_ptr, n * sizeof(T)));
    std::memcpy(uvm_ptr, arrow_ptr, n * sizeof(T));
    // CUDA 13.0+: cudaMemPrefetchAsync requires cudaMemLocation struct
    cudaMemLocation location = {};
    location.type = cudaMemLocationTypeDevice;
    location.id = device;
    CUDA_CHECK(cudaMemPrefetchAsync(uvm_ptr, n * sizeof(T), location, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    return uvm_ptr;
}

inline void arrowFreeUVM(void* ptr) { CUDA_CHECK(cudaFree(ptr)); }

// ── Benchmark all three strategies ──────────────────────────────────────────
void benchArrowTransferStrategies(const double* h_data, std::size_t n);

} // namespace gpudb

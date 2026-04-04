/**
 * @file cuda_utils.cuh
 * @brief Core CUDA utilities: error checking, RAII memory buffers, GPU context.
 */
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <utility>

namespace gpudb {

// ── CUDA_CHECK macro ────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(err));                                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// ── GpuBuffer<T> — RAII device memory ───────────────────────────────────────
template <typename T> class GpuBuffer {
public:
  GpuBuffer() = default;
  explicit GpuBuffer(std::size_t n) : size_(n) {
    CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
  }
  ~GpuBuffer() {
    if (ptr_)
      cudaFree(ptr_);
  }
  GpuBuffer(const GpuBuffer &) = delete;
  GpuBuffer &operator=(const GpuBuffer &) = delete;
  GpuBuffer(GpuBuffer &&o) noexcept : ptr_(o.ptr_), size_(o.size_) {
    o.ptr_ = nullptr;
    o.size_ = 0;
  }
  GpuBuffer &operator=(GpuBuffer &&o) noexcept {
    if (this != &o) {
      if (ptr_)
        cudaFree(ptr_);
      ptr_ = o.ptr_;
      size_ = o.size_;
      o.ptr_ = nullptr;
      o.size_ = 0;
    }
    return *this;
  }
  T *data() { return ptr_; }
  const T *data() const { return ptr_; }
  std::size_t size() const { return size_; }
  std::size_t bytes() const { return size_ * sizeof(T); }
  void memsetZero() { CUDA_CHECK(cudaMemset(ptr_, 0, bytes())); }
  void fill(int byte_val) { CUDA_CHECK(cudaMemset(ptr_, byte_val, bytes())); }

private:
  T *ptr_ = nullptr;
  std::size_t size_ = 0;
};

// ── GpuPinnedBuffer<T> — RAII pinned host memory ────────────────────────────
template <typename T> class GpuPinnedBuffer {
public:
  GpuPinnedBuffer() = default;
  explicit GpuPinnedBuffer(std::size_t n) : size_(n) {
    CUDA_CHECK(cudaHostAlloc(&ptr_, n * sizeof(T), cudaHostAllocDefault));
  }
  ~GpuPinnedBuffer() {
    if (ptr_)
      cudaFreeHost(ptr_);
  }
  GpuPinnedBuffer(const GpuPinnedBuffer &) = delete;
  GpuPinnedBuffer &operator=(const GpuPinnedBuffer &) = delete;
  GpuPinnedBuffer(GpuPinnedBuffer &&o) noexcept : ptr_(o.ptr_), size_(o.size_) {
    o.ptr_ = nullptr;
    o.size_ = 0;
  }
  GpuPinnedBuffer &operator=(GpuPinnedBuffer &&o) noexcept {
    if (this != &o) {
      if (ptr_)
        cudaFreeHost(ptr_);
      ptr_ = o.ptr_;
      size_ = o.size_;
      o.ptr_ = nullptr;
      o.size_ = 0;
    }
    return *this;
  }
  T *data() { return ptr_; }
  const T *data() const { return ptr_; }
  std::size_t size() const { return size_; }

private:
  T *ptr_ = nullptr;
  std::size_t size_ = 0;
};

// ── GpuContext ──────────────────────────────────────────────────────────────
class GpuContext {
public:
  explicit GpuContext(int device = 0) : device_(device) {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaGetDeviceProperties(&props_, device));
  }
  void printDeviceInfo() const {
    std::printf("\n  ┌─────────────────────────────────────────────┐\n");
    std::printf("  │  GPU: %-37s│\n", props_.name);
    std::printf("  ├─────────────────────────────────────────────┤\n");
    std::printf("  │  Compute capability : %d.%-20d│\n", props_.major,
                props_.minor);
    std::printf("  │  Total VRAM         : %-6.0f MB            │\n",
                props_.totalGlobalMem / 1e6);
    std::printf("  │  Streaming MPs      : %-22d│\n",
                props_.multiProcessorCount);
    std::printf("  │  Warp size          : %-22d│\n", props_.warpSize);
    std::printf("  │  Max threads/block  : %-22d│\n",
                props_.maxThreadsPerBlock);
    std::printf("  └─────────────────────────────────────────────┘\n\n");
  }
  template <typename T>
  void copyToDevice(GpuBuffer<T> &dst, const T *src, std::size_t n,
                    cudaStream_t s = 0) const {
    CUDA_CHECK(cudaMemcpyAsync(dst.data(), src, n * sizeof(T),
                               cudaMemcpyHostToDevice, s));
  }
  template <typename T>
  void copyToHost(T *dst, const GpuBuffer<T> &src, std::size_t n,
                  cudaStream_t s = 0) const {
    CUDA_CHECK(cudaMemcpyAsync(dst, src.data(), n * sizeof(T),
                               cudaMemcpyDeviceToHost, s));
  }
  int sms() const { return props_.multiProcessorCount; }
  int warpSize() const { return props_.warpSize; }

private:
  cudaDeviceProp props_{};
  int device_;
};

} // namespace gpudb

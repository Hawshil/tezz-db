/**
 * @file sycl_utils.h
 * @brief Intel oneAPI/SYCL utilities — device selection, USM RAII buffers.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 *  SYCL Buffer/Accessor Model vs CUDA Explicit cudaMalloc Model
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 *  CUDA model:
 *    double* d_ptr;
 *    cudaMalloc(&d_ptr, n * sizeof(double));       // explicit alloc
 *    cudaMemcpy(d_ptr, h_ptr, n*8, H2D);           // explicit copy
 *    kernel<<<G,B>>>(d_ptr, n);                     // launch
 *    cudaMemcpy(h_ptr, d_ptr, n*8, D2H);           // explicit copy back
 *    cudaFree(d_ptr);                               // explicit free
 *
 *  SYCL buffer/accessor model (idiomatic):
 *    sycl::buffer<double,1> buf(h_ptr, sycl::range<1>(n));
 *    q.submit([&](sycl::handler& h) {
 *      auto acc = buf.get_access<sycl::access::mode::read_write>(h);
 *      h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
 *        acc[i] *= 2.0;
 *      });
 *    });
 *    // Data automatically copied back when buffer goes out of scope.
 *    // Runtime manages all H↔D transfers. No manual memcpy.
 *
 *  SYCL USM model (CUDA-like, used in this port for easier migration):
 *    double* d = sycl::malloc_device<double>(n, q);
 *    q.memcpy(d, h_ptr, n*8).wait();
 *    q.parallel_for(n, [=](auto i){ d[i] *= 2.0; }).wait();
 *    q.memcpy(h_ptr, d, n*8).wait();
 *    sycl::free(d, q);
 *
 *  Key difference: buffer/accessor lets the runtime optimise data movement,
 *  while USM gives explicit pointer control like CUDA. We use USM here
 *  because it maps 1:1 to the CUDA/HIP code structure.
 * ═══════════════════════════════════════════════════════════════════════════════
 */
#pragma once
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>

namespace gpudb {

// ── Device selector: prefer Intel GPU ───────────────────────────────────────
inline sycl::queue makeGpuQueue() {
    try {
        return sycl::queue{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    } catch (...) {
        std::fprintf(stderr, "No GPU found — falling back to default device.\n");
        return sycl::queue{sycl::default_selector_v};
    }
}

inline void listAllDevices() {
    for (auto& p : sycl::platform::get_platforms()) {
        for (auto& d : p.get_devices()) {
            std::printf("  [%s] %s — %s\n",
                d.is_gpu() ? "GPU" : d.is_cpu() ? "CPU" : "ACC",
                d.get_info<sycl::info::device::name>().c_str(),
                p.get_info<sycl::info::platform::name>().c_str());
        }
    }
}

inline void printDeviceInfo(const sycl::queue& q) {
    auto d = q.get_device();
    std::printf("\n  ┌──────────────────────────────────────────────────┐\n");
    std::printf("  │  Intel GPU: %-37s│\n",
        d.get_info<sycl::info::device::name>().c_str());
    std::printf("  ├──────────────────────────────────────────────────┤\n");
    std::printf("  │  Max compute units  : %-25d│\n",
        d.get_info<sycl::info::device::max_compute_units>());
    std::printf("  │  Max work-group size: %-25zu│\n",
        d.get_info<sycl::info::device::max_work_group_size>());
    std::printf("  │  Global memory      : %-6.0f MB               │\n",
        d.get_info<sycl::info::device::global_mem_size>() / 1e6);
    std::printf("  │  Local memory       : %-6zu KB               │\n",
        d.get_info<sycl::info::device::local_mem_size>() / 1024);
    std::printf("  │  Sub-group sizes    :");
    for (auto sz : d.get_info<sycl::info::device::sub_group_sizes>())
        std::printf(" %zu", sz);
    std::printf("\n  └──────────────────────────────────────────────────┘\n\n");
}

// ── SyclBuffer<T> — RAII USM device memory ──────────────────────────────────
template<typename T>
class SyclBuffer {
public:
    SyclBuffer() = default;
    SyclBuffer(std::size_t n, sycl::queue& q) : q_(&q), size_(n) {
        ptr_ = sycl::malloc_device<T>(n, q);
        if (!ptr_) { std::fprintf(stderr, "sycl::malloc_device failed\n"); std::exit(1); }
    }
    ~SyclBuffer() { if (ptr_ && q_) sycl::free(ptr_, *q_); }
    SyclBuffer(const SyclBuffer&) = delete;
    SyclBuffer& operator=(const SyclBuffer&) = delete;
    SyclBuffer(SyclBuffer&& o) noexcept : q_(o.q_), ptr_(o.ptr_), size_(o.size_) {
        o.ptr_ = nullptr; o.size_ = 0;
    }
    SyclBuffer& operator=(SyclBuffer&& o) noexcept {
        if (this != &o) { if (ptr_ && q_) sycl::free(ptr_, *q_);
            q_ = o.q_; ptr_ = o.ptr_; size_ = o.size_;
            o.ptr_ = nullptr; o.size_ = 0; }
        return *this;
    }
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return size_; }
    void memsetZero() { q_->memset(ptr_, 0, size_ * sizeof(T)).wait(); }
    void fill(int v)  { q_->memset(ptr_, v, size_ * sizeof(T)).wait(); }
    void copyFrom(const T* h, std::size_t n) { q_->memcpy(ptr_, h, n*sizeof(T)).wait(); }
    void copyTo(T* h, std::size_t n) const   { q_->memcpy(h, ptr_, n*sizeof(T)).wait(); }
private:
    sycl::queue* q_ = nullptr;
    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

// ── Vector-add verification kernel ──────────────────────────────────────────
inline bool verifySetup(sycl::queue& q) {
    const int N = 1024;
    auto* a = sycl::malloc_shared<float>(N, q);
    auto* b = sycl::malloc_shared<float>(N, q);
    auto* c = sycl::malloc_shared<float>(N, q);
    for (int i = 0; i < N; ++i) { a[i] = (float)i; b[i] = (float)(i*2); }

    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        c[i] = a[i] + b[i];
    }).wait();

    bool ok = true;
    for (int i = 0; i < N; ++i) if (c[i] != a[i] + b[i]) { ok = false; break; }

    sycl::free(a, q); sycl::free(b, q); sycl::free(c, q);
    std::printf("  Vector-add verification: %s\n", ok ? "PASS ✓" : "FAIL ✗");
    return ok;
}

} // namespace gpudb

/**
 * @file gpu_backend.h
 * @brief Abstract GPU backend interface + factory for runtime backend selection.
 *
 * This is the key portability layer: QueryPlanner uses GpuBackend without
 * knowing whether it's running on NVIDIA, AMD, or Intel hardware.
 * Compile with -DUSE_CUDA, -DUSE_HIP, or -DUSE_SYCL to enable backends.
 */
#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <cstdio>

namespace gpudb {

class GpuBackend {
public:
    virtual ~GpuBackend() = default;
    virtual std::string name() const = 0;
    virtual std::string deviceName() const = 0;

    // Memory
    virtual void* allocate(std::size_t bytes) = 0;
    virtual void  deallocate(void* ptr) = 0;
    virtual void  copyToDevice(void* dst, const void* src, std::size_t bytes) = 0;
    virtual void  copyToHost(void* dst, const void* src, std::size_t bytes) = 0;
    virtual void  memsetZero(void* ptr, std::size_t bytes) = 0;

    // Operators
    virtual int   filter(const double* d_in, int n, double threshold,
                         int op, double* d_out) = 0;
    virtual double sum(const double* d_in, int n) = 0;
    virtual int   groupBySum(const int* d_keys, const double* d_vals, int n,
                             int* h_out_keys, double* h_out_sums, int ht_cap) = 0;
    virtual int   hashJoin(const int* d_build, int nb, const int* d_probe, int np,
                           int* d_out_b, int* d_out_p, int ht_cap) = 0;
    virtual void  sync() = 0;
};

// ── CUDA Backend (compiled only with -DUSE_CUDA) ────────────────────────────
#ifdef USE_CUDA
class CudaBackend : public GpuBackend {
public:
    std::string name() const override { return "CUDA"; }
    std::string deviceName() const override;
    void* allocate(std::size_t bytes) override;
    void  deallocate(void* ptr) override;
    void  copyToDevice(void* dst, const void* src, std::size_t bytes) override;
    void  copyToHost(void* dst, const void* src, std::size_t bytes) override;
    void  memsetZero(void* ptr, std::size_t bytes) override;
    int   filter(const double* d_in, int n, double threshold,
                 int op, double* d_out) override;
    double sum(const double* d_in, int n) override;
    int   groupBySum(const int* d_keys, const double* d_vals, int n,
                     int* h_out_keys, double* h_out_sums, int ht_cap) override;
    int   hashJoin(const int* d_build, int nb, const int* d_probe, int np,
                   int* d_out_b, int* d_out_p, int ht_cap) override;
    void  sync() override;
};
#endif

// ── HIP Backend (compiled only with -DUSE_HIP) ─────────────────────────────
#ifdef USE_HIP
class HipBackend : public GpuBackend {
public:
    std::string name() const override { return "HIP/ROCm"; }
    std::string deviceName() const override;
    void* allocate(std::size_t bytes) override;
    void  deallocate(void* ptr) override;
    void  copyToDevice(void* dst, const void* src, std::size_t bytes) override;
    void  copyToHost(void* dst, const void* src, std::size_t bytes) override;
    void  memsetZero(void* ptr, std::size_t bytes) override;
    int   filter(const double* d_in, int n, double threshold,
                 int op, double* d_out) override;
    double sum(const double* d_in, int n) override;
    int   groupBySum(const int* d_keys, const double* d_vals, int n,
                     int* h_out_keys, double* h_out_sums, int ht_cap) override;
    int   hashJoin(const int* d_build, int nb, const int* d_probe, int np,
                   int* d_out_b, int* d_out_p, int ht_cap) override;
    void  sync() override;
};
#endif

// ── SYCL Backend (compiled only with -DUSE_SYCL) ───────────────────────────
#ifdef USE_SYCL
class SyclBackend : public GpuBackend {
public:
    SyclBackend();
    std::string name() const override { return "SYCL/oneAPI"; }
    std::string deviceName() const override;
    void* allocate(std::size_t bytes) override;
    void  deallocate(void* ptr) override;
    void  copyToDevice(void* dst, const void* src, std::size_t bytes) override;
    void  copyToHost(void* dst, const void* src, std::size_t bytes) override;
    void  memsetZero(void* ptr, std::size_t bytes) override;
    int   filter(const double* d_in, int n, double threshold,
                 int op, double* d_out) override;
    double sum(const double* d_in, int n) override;
    int   groupBySum(const int* d_keys, const double* d_vals, int n,
                     int* h_out_keys, double* h_out_sums, int ht_cap) override;
    int   hashJoin(const int* d_build, int nb, const int* d_probe, int np,
                   int* d_out_b, int* d_out_p, int ht_cap) override;
    void  sync() override;
private:
    void* q_ = nullptr;  // opaque sycl::queue*
};
#endif

// ── BackendFactory ──────────────────────────────────────────────────────────
class BackendFactory {
public:
    static std::unique_ptr<GpuBackend> createBest() {
#ifdef USE_CUDA
        std::printf("  [BackendFactory] CUDA backend selected\n");
        return std::make_unique<CudaBackend>();
#elif defined(USE_HIP)
        std::printf("  [BackendFactory] HIP/ROCm backend selected\n");
        return std::make_unique<HipBackend>();
#elif defined(USE_SYCL)
        std::printf("  [BackendFactory] SYCL/oneAPI backend selected\n");
        return std::make_unique<SyclBackend>();
#else
        std::printf("  [BackendFactory] No GPU backend available — CPU only\n");
        return nullptr;
#endif
    }

    static std::vector<std::string> availableBackends() {
        std::vector<std::string> v;
#ifdef USE_CUDA
        v.push_back("CUDA");
#endif
#ifdef USE_HIP
        v.push_back("HIP/ROCm");
#endif
#ifdef USE_SYCL
        v.push_back("SYCL/oneAPI");
#endif
        return v;
    }
};

} // namespace gpudb

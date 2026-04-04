/**
 * @file cuda_backend.cu
 * @brief CudaBackend implementation — wraps existing CUDA kernels behind
 *        the GpuBackend interface for vendor-agnostic query execution.
 */
#ifdef USE_CUDA
#include "backend/gpu_backend.h"
#include "gpu/cuda_utils.cuh"
#include "gpu/gpu_ops.cuh"
#include <cuda_runtime.h>

namespace gpudb {

std::string CudaBackend::deviceName() const {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    return std::string(p.name);
}
void* CudaBackend::allocate(std::size_t bytes) {
    void* p = nullptr; CUDA_CHECK(cudaMalloc(&p, bytes)); return p;
}
void CudaBackend::deallocate(void* p) { cudaFree(p); }
void CudaBackend::copyToDevice(void* d, const void* h, std::size_t b) {
    CUDA_CHECK(cudaMemcpy(d, h, b, cudaMemcpyHostToDevice));
}
void CudaBackend::copyToHost(void* h, const void* d, std::size_t b) {
    CUDA_CHECK(cudaMemcpy(h, d, b, cudaMemcpyDeviceToHost));
}
void CudaBackend::memsetZero(void* p, std::size_t b) {
    CUDA_CHECK(cudaMemset(p, 0, b));
}
int CudaBackend::filter(const double* d_in, int n, double t, int op, double* d_out) {
    return gpuFilter(d_in, n, t, (CompareOp)op, d_out);
}
double CudaBackend::sum(const double* d_in, int n) { return gpuSum(d_in, n); }
int CudaBackend::groupBySum(const int* dk, const double* dv, int n,
                             int* hk, double* hs, int cap) {
    return gpuGroupBySum(dk, dv, n, hk, hs, cap);
}
int CudaBackend::hashJoin(const int* db, int nb, const int* dp, int np,
                           int* ob, int* op, int cap) {
    return gpuHashJoin(db, nb, dp, np, ob, op, cap);
}
void CudaBackend::sync() { CUDA_CHECK(cudaDeviceSynchronize()); }

} // namespace gpudb
#endif

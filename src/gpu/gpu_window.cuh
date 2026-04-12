/**
 * @file gpu_window.cuh
 * @brief GPU-accelerated window function declarations: SMA, EMA, RollingStd.
 */
#pragma once
#include "cuda_utils.cuh"
#include <cstdint>

namespace gpudb {

/// Compute SMA of d_in[0..n-1] with window w into d_out[0..n-1].
/// d_out[i] = mean of d_in[max(0,i-w+1)..i].
void gpuSMA(const double* d_in, int n, int w, double* d_out);

/// Compute EMA with smoothing factor alpha = 2/(w+1).
void gpuEMA(const double* d_in, int n, int w, double* d_out);

/// Compute rolling population stddev with window w.
void gpuRollingStd(const double* d_in, int n, int w, double* d_out);

} // namespace gpudb

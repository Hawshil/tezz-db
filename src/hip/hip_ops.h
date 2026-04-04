/**
 * @file hip_ops.h
 * @brief Declarations for all HIP GPU operations (AMD ROCm port).
 */
#pragma once
#include "hip_utils.h"

namespace gpudb {

enum class HipCompareOp { GT=0, GTE, LT, LTE, EQ, NEQ };

int  hipFilter(const double* d_in, int n, double threshold,
               HipCompareOp op, double* d_out);

double       hipSum  (const double* d_in, int n);
std::int64_t hipCount(const int*    d_in, int n);

int hipGroupBySum(const int* d_keys, const double* d_vals, int n,
                  int* h_out_keys, double* h_out_sums, int ht_cap);

int hipHashJoin(const int* d_build, int n_build,
                const int* d_probe, int n_probe,
                int* d_match_build, int* d_match_probe, int ht_cap);

void hipRunAsyncPipeline(const double* h_vals, const int* h_keys,
                         int total_rows, int chunk_size, int num_groups);

} // namespace gpudb

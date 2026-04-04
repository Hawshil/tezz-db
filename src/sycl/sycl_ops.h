/**
 * @file sycl_ops.h
 * @brief Declarations for all SYCL GPU operations (Intel oneAPI port).
 */
#pragma once
#include "sycl_utils.h"

namespace gpudb {

enum class SyclCompareOp { GT=0, GTE, LT, LTE, EQ, NEQ };

int  syclFilter(sycl::queue& q, const double* d_in, int n,
                double threshold, SyclCompareOp op, double* d_out);

double       syclSum  (sycl::queue& q, const double* d_in, int n);
double       syclSumManual(sycl::queue& q, const double* d_in, int n);
std::int64_t syclCount(sycl::queue& q, const int* d_in, int n);

int syclGroupBySum(sycl::queue& q, const int* d_keys, const double* d_vals, int n,
                   int* h_out_keys, double* h_out_sums, int ht_cap);

int syclHashJoin(sycl::queue& q, const int* d_build, int n_build,
                 const int* d_probe, int n_probe,
                 int* d_match_build, int* d_match_probe, int ht_cap);

} // namespace gpudb

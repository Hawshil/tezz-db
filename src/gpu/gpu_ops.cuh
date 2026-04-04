/**
 * @file gpu_ops.cuh
 * @brief Declarations for all GPU database operations.
 */
#pragma once
#include "cuda_utils.cuh"

namespace gpudb {

// ── Filter (stream compaction) ──────────────────────────────────────────────
enum class CompareOp { GT=0, GTE, LT, LTE, EQ, NEQ };
int gpuFilter(const double* d_in, int n, double threshold,
              CompareOp op, double* d_out);

// ── Aggregation ─────────────────────────────────────────────────────────────
double       gpuSum  (const double* d_in, int n);
std::int64_t gpuCount(const int*    d_in, int n);

// ── GROUP BY + SUM ──────────────────────────────────────────────────────────
int gpuGroupBySum(const int* d_keys, const double* d_vals, int n,
                  int* h_out_keys, double* h_out_sums, int ht_capacity);

// ── Hash Join ───────────────────────────────────────────────────────────────
int gpuHashJoin(const int* d_build, int n_build,
                const int* d_probe, int n_probe,
                int* d_match_build, int* d_match_probe, int ht_capacity);

// ── Async Pipeline ──────────────────────────────────────────────────────────
void runAsyncPipeline(const double* h_vals, const int* h_keys,
                      int total_rows, int chunk_size, int num_groups);

// ── Compressed data ops ─────────────────────────────────────────────────────
void gpuDictGroupBySum(const int* d_codes, const double* d_vals, int n,
                       double* h_sums, int num_groups);
void gpuRleAggSum(const int* d_rle_vals, const int* d_rle_lens, int num_runs,
                  const double* d_vals, double* h_group_sums, int num_groups);

// ── GPU Gather (late materialization) ───────────────────────────────────────
void gpuGather(const double* d_src, const int* d_indices, int sel_count,
              double* d_out);

// ── CUDA Graph benchmark ────────────────────────────────────────────────────
void benchGraphExecution(const int* h_keys, const double* h_vals, int n,
                         int num_groups, int num_queries);

// ── Arrow transfer strategies ───────────────────────────────────────────────
void benchArrowTransferStrategies(const double* h_data, std::size_t n);

} // namespace gpudb

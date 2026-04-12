/**
 * @file gpu_asof.cuh
 * @brief GPU-accelerated ASOF join via per-thread binary search.
 */
#pragma once
#include "cuda_utils.cuh"
#include <cstdint>

namespace gpudb {

/**
 * @brief GPU ASOF join: for each left row i, find the largest right index j
 *        such that right_ts[j] <= left_ts[i] (and optionally keys match).
 *
 * @param d_left_ts      Sorted ascending int64 timestamps (device, n_left).
 * @param n_left         Number of left rows.
 * @param d_right_ts     Sorted ascending int64 timestamps (device, n_right).
 * @param n_right        Number of right rows.
 * @param d_left_key     Equality key for left (device, n_left). Ignored if !use_key.
 * @param d_right_key    Equality key for right (device, n_right). Ignored if !use_key.
 * @param use_key        If true, require key match in addition to timestamp.
 * @param tolerance_ns   Max allowed gap (left_ts - right_ts). 0 = no limit.
 * @param d_out_right_idx Output: matched right index per left row, or -1 (device, n_left).
 *
 * KNOWN LIMITATION: with use_key=true, right array must be pre-sorted by
 * (key, ts). Caller is responsible. The CPU fallback in AsofJoinNode
 * handles the general case.
 */
void gpuAsofJoin(const std::int64_t* d_left_ts,  int n_left,
                 const std::int64_t* d_right_ts,  int n_right,
                 const std::int32_t* d_left_key,
                 const std::int32_t* d_right_key,
                 bool               use_key,
                 std::int64_t       tolerance_ns,
                 int*               d_out_right_idx);

} // namespace gpudb

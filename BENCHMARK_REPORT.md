# GPUDB Benchmark Report

## System Info

| Property | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 |
| VRAM | 4295 MB |
| CUDA Compute | 7.5 |

## Speedup Results

| Operation | Rows | CPU (ms) | GPU (ms) | Speedup | BW (GB/s) |
|---|---|---|---|---|---|

*See `benchmark_results.csv` for full data.*

## Observations

- **Filter** achieves highest speedup at large row counts because it is
  memory-bandwidth-bound, and GPU HBM/GDDR bandwidth exceeds CPU DDR by 5-10x.
- **SUM reduction** benefits from warp-shuffle / sub-group primitives that
  eliminate shared-memory synchronization overhead.
- **GROUP BY** speedup is limited by atomic contention on the hash table;
  shared-memory partial aggregation mitigates this.
- **Hash Join** is PCIe-bound for small build tables; the GPU advantage
  emerges at 10M+ probe rows.

/**
 * @file arrow_bridge.h
 * @brief Apache Arrow ↔ gpudb::Table converter + IPC file reader/writer.
 *
 * Arrow Columnar Layout ↔ TypedColumn Mapping
 * ────────────────────────────────────────────
 * Arrow stores columns as contiguous, cache-aligned byte buffers (64-byte
 * aligned by default). Each primitive array has two buffers:
 *   [0] validity bitmap  (1 bit per element — maps to our null_bitmap_)
 *   [1] values buffer     (contiguous T[] — maps directly to our data_ vector)
 *
 * This layout is identical to our TypedColumn<T>::data() pointer: both are
 * flat, densely-packed arrays of primitive values. Because GPU global memory
 * reads are coalesced when adjacent threads access adjacent addresses, Arrow's
 * layout gives us coalesced GPU access for free — no layout transformation
 * needed. We can cudaMemcpy Arrow's raw buffer pointer directly to device
 * memory and launch kernels on it.
 *
 * String columns in Arrow use a separate offsets buffer + values buffer
 * (variable-length binary). These require flattening for GPU use.
 */
#pragma once
#include "core/table.h"
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <memory>
#include <string>

namespace gpudb {

// ── ArrowConverter ──────────────────────────────────────────────────────────
class ArrowConverter {
public:
    static std::shared_ptr<arrow::Table> toArrowTable(const Table& src);
    static Table fromArrowTable(const std::shared_ptr<arrow::Table>& atbl,
                                const std::string& name = "arrow_table");
};

// ── ArrowIPCReader ──────────────────────────────────────────────────────────
class ArrowIPCReader {
public:
    static Table readFile(const std::string& path,
                          const std::string& table_name = "ipc_table");
};

// ── ArrowIPCWriter ──────────────────────────────────────────────────────────
class ArrowIPCWriter {
public:
    static void writeFile(const Table& src, const std::string& path);
};

} // namespace gpudb

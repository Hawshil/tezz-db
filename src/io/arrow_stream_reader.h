/**
 * @file arrow_stream_reader.h
 * @brief Incremental Arrow IPC stream reader that feeds arriving
 *        RecordBatches into a RingTable.
 *
 * Supports:
 *   - readFile()      — one-shot read from an IPC stream file.
 *   - readStream()    — blocking read from an Arrow InputStream (socket/pipe).
 *   - startPolling()  — background thread that re-reads a file periodically
 *                       (simulates a live tick feed from a rotating file).
 *
 * All Arrow-dependent code is guarded by GPUDB_HAS_ARROW.
 */
#pragma once

#ifdef GPUDB_HAS_ARROW
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#endif

#include "../core/ring_table.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace gpudb {

class ArrowStreamReader {
public:
    /// Called synchronously on the reader thread for each arriving batch.
    using BatchCallback = std::function<void(std::size_t rows_appended)>;

    // ── One-shot file read ──────────────────────────────────────────────────

    /**
     * @brief Read all RecordBatches from an Arrow IPC stream file.
     *
     * @param path       Path to the IPC stream file.
     * @param target     RingTable to append batches into.
     * @param on_batch   Optional callback invoked after each batch.
     */
    static void readFile(const std::string& path,
                         RingTable& target,
                         BatchCallback on_batch = {}) {
#ifdef GPUDB_HAS_ARROW
        auto file_result = arrow::io::ReadableFile::Open(path);
        if (!file_result.ok())
            throw std::runtime_error(
                "ArrowStreamReader::readFile — cannot open: " + path +
                " (" + file_result.status().ToString() + ")");

        auto stream = *file_result;
        readStream(stream, target, on_batch);
#else
        (void)path; (void)target; (void)on_batch;
        throw std::runtime_error(
            "ArrowStreamReader::readFile — Arrow support not compiled "
            "(GPUDB_HAS_ARROW=0)");
#endif
    }

    // ── Blocking stream read ────────────────────────────────────────────────

#ifdef GPUDB_HAS_ARROW
    /**
     * @brief Read from an open Arrow IPC stream until EOF or stop().
     *
     * @param stream    Arrow InputStream (file, socket, pipe, …).
     * @param target    RingTable to append into.
     * @param on_batch  Optional per-batch callback.
     */
    static void readStream(
            std::shared_ptr<arrow::io::InputStream> stream,
            RingTable& target,
            BatchCallback on_batch = {}) {
        auto reader_result =
            arrow::ipc::RecordBatchStreamReader::Open(stream);
        if (!reader_result.ok())
            throw std::runtime_error(
                "ArrowStreamReader::readStream — cannot open IPC reader ("
                + reader_result.status().ToString() + ")");

        auto reader = *reader_result;
        while (true) {
            std::shared_ptr<arrow::RecordBatch> batch;
            auto status = reader->ReadNext(&batch);
            if (!status.ok())
                throw std::runtime_error(
                    "ArrowStreamReader::readStream — read error: "
                    + status.ToString());
            if (!batch) break;  // EOF
            processRecordBatch(batch, target, on_batch);
        }
    }
#endif

    // ── Polling mode ────────────────────────────────────────────────────────

    /**
     * @brief Launch a background thread that re-reads the file every
     *        interval_ms milliseconds (simulates a live tick feed).
     */
    void startPolling(const std::string& path,
                      RingTable& target,
                      int interval_ms = 100,
                      BatchCallback on_batch = {}) {
        running_ = true;
        thread_ = std::thread([this, path, &target, interval_ms, on_batch]() {
            while (running_) {
                try {
                    readFile(path, target, on_batch);
                } catch (const std::exception&) {
                    // File may not exist yet or be mid-rotation — retry.
                }
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(interval_ms));
            }
        });
    }

    /// Signal the polling thread to stop and join it.
    void stop() {
        running_ = false;
        if (thread_.joinable())
            thread_.join();
    }

    ~ArrowStreamReader() {
        stop();
    }

private:
    std::atomic<bool> running_{false};
    std::thread thread_;

    // ── Batch processing ────────────────────────────────────────────────────

#ifdef GPUDB_HAS_ARROW
    /**
     * @brief Convert an Arrow RecordBatch into raw byte buffers and call
     *        target.appendBatch().
     */
    static void processRecordBatch(
            const std::shared_ptr<arrow::RecordBatch>& batch,
            RingTable& target,
            BatchCallback& cb) {
        const int num_cols = batch->num_columns();
        const std::int64_t num_rows = batch->num_rows();
        const auto& types = target.columnTypes();

        std::vector<std::vector<std::byte>> column_batches(num_cols);

        for (int i = 0; i < num_cols; ++i) {
            auto col = batch->column(i);
            switch (types[i]) {
                case DataType::INT32: {
                    auto typed = std::static_pointer_cast<arrow::Int32Array>(col);
                    std::size_t bytes = num_rows * sizeof(std::int32_t);
                    column_batches[i].resize(bytes);
                    std::memcpy(column_batches[i].data(),
                                typed->raw_values(), bytes);
                    break;
                }
                case DataType::INT64: {
                    auto typed = std::static_pointer_cast<arrow::Int64Array>(col);
                    std::size_t bytes = num_rows * sizeof(std::int64_t);
                    column_batches[i].resize(bytes);
                    std::memcpy(column_batches[i].data(),
                                typed->raw_values(), bytes);
                    break;
                }
                case DataType::FLOAT64: {
                    auto typed = std::static_pointer_cast<arrow::DoubleArray>(col);
                    std::size_t bytes = num_rows * sizeof(double);
                    column_batches[i].resize(bytes);
                    std::memcpy(column_batches[i].data(),
                                typed->raw_values(), bytes);
                    break;
                }
                case DataType::STRING: {
                    // String columns: push values one-by-one through the
                    // RingColumn<string> API (appendBatch byte buffer is
                    // not used for strings).
                    auto typed = std::static_pointer_cast<arrow::StringArray>(col);
                    auto& rc = std::get<RingColumn<std::string>>(
                                   const_cast<RingColumnVariant&>(target.getCol(i)));
                    for (std::int64_t r = 0; r < num_rows; ++r)
                        rc.push(typed->GetString(r));
                    break;
                }
            }
        }

        // Append the non-string columns via the batch API.
        target.appendBatch(column_batches, static_cast<std::size_t>(num_rows));

        if (cb) cb(static_cast<std::size_t>(num_rows));
    }
#endif
};

}  // namespace gpudb

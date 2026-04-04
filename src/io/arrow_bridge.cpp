/**
 * @file arrow_bridge.cpp
 * @brief Arrow ↔ Table conversion + IPC file I/O implementation.
 */
#include "arrow_bridge.h"
#include <arrow/builder.h>
#include <arrow/type.h>
#include <stdexcept>

namespace gpudb {

// ═══════════════════════════════════════════════════════════════════════════
// toArrowTable
// ═══════════════════════════════════════════════════════════════════════════

std::shared_ptr<arrow::Table> ArrowConverter::toArrowTable(const Table& src) {
    std::vector<std::shared_ptr<arrow::Field>>  fields;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;

    for (std::size_t c = 0; c < src.numCols(); ++c) {
        const auto& name = src.columnNames()[c];
        const auto& col  = src.getColumn(c);

        std::visit([&](const auto& typed) {
            using ColT = std::decay_t<decltype(typed)>;
            using ValT = typename ColT::value_type;

            if constexpr (std::is_same_v<ValT, std::int32_t>) {
                arrow::Int32Builder b;
                (void)b.AppendValues(typed.data(), typed.size());
                std::shared_ptr<arrow::Array> arr; (void)b.Finish(&arr);
                fields.push_back(arrow::field(name, arrow::int32()));
                columns.push_back(std::make_shared<arrow::ChunkedArray>(arr));
            } else if constexpr (std::is_same_v<ValT, std::int64_t>) {
                arrow::Int64Builder b;
                (void)b.AppendValues(typed.data(), typed.size());
                std::shared_ptr<arrow::Array> arr; (void)b.Finish(&arr);
                fields.push_back(arrow::field(name, arrow::int64()));
                columns.push_back(std::make_shared<arrow::ChunkedArray>(arr));
            } else if constexpr (std::is_same_v<ValT, double>) {
                arrow::DoubleBuilder b;
                (void)b.AppendValues(typed.data(), typed.size());
                std::shared_ptr<arrow::Array> arr; (void)b.Finish(&arr);
                fields.push_back(arrow::field(name, arrow::float64()));
                columns.push_back(std::make_shared<arrow::ChunkedArray>(arr));
            } else if constexpr (std::is_same_v<ValT, std::string>) {
                arrow::StringBuilder b;
                for (std::size_t i = 0; i < typed.size(); ++i)
                    (void)b.Append(typed[i]);
                std::shared_ptr<arrow::Array> arr; (void)b.Finish(&arr);
                fields.push_back(arrow::field(name, arrow::utf8()));
                columns.push_back(std::make_shared<arrow::ChunkedArray>(arr));
            }
        }, col);
    }

    auto schema = arrow::schema(fields);
    return arrow::Table::Make(schema, columns);
}

// ═══════════════════════════════════════════════════════════════════════════
// fromArrowTable
// ═══════════════════════════════════════════════════════════════════════════

Table ArrowConverter::fromArrowTable(const std::shared_ptr<arrow::Table>& atbl,
                                     const std::string& name) {
    std::vector<std::string> col_names;
    std::vector<ColumnVariant> columns;
    auto schema = atbl->schema();

    for (int c = 0; c < schema->num_fields(); ++c) {
        col_names.push_back(schema->field(c)->name());
        auto chunk = atbl->column(c)->chunk(0);  // assume single chunk

        switch (schema->field(c)->type()->id()) {
        case arrow::Type::INT32: {
            auto arr = std::static_pointer_cast<arrow::Int32Array>(chunk);
            Int32Column col;
            col.reserve(arr->length());
            for (int64_t i = 0; i < arr->length(); ++i) col.pushValue(arr->Value(i));
            columns.emplace_back(std::move(col));
            break;
        }
        case arrow::Type::INT64: {
            auto arr = std::static_pointer_cast<arrow::Int64Array>(chunk);
            Int64Column col;
            col.reserve(arr->length());
            for (int64_t i = 0; i < arr->length(); ++i) col.pushValue(arr->Value(i));
            columns.emplace_back(std::move(col));
            break;
        }
        case arrow::Type::DOUBLE: {
            auto arr = std::static_pointer_cast<arrow::DoubleArray>(chunk);
            Float64Column col;
            col.reserve(arr->length());
            for (int64_t i = 0; i < arr->length(); ++i) col.pushValue(arr->Value(i));
            columns.emplace_back(std::move(col));
            break;
        }
        case arrow::Type::STRING: {
            auto arr = std::static_pointer_cast<arrow::StringArray>(chunk);
            StringColumn col;
            col.reserve(arr->length());
            for (int64_t i = 0; i < arr->length(); ++i)
                col.pushValue(arr->GetString(i));
            columns.emplace_back(std::move(col));
            break;
        }
        default:
            throw std::runtime_error("Unsupported Arrow type: " +
                                     schema->field(c)->type()->ToString());
        }
    }
    return Table(name, std::move(col_names), std::move(columns));
}

// ═══════════════════════════════════════════════════════════════════════════
// IPC Reader
// ═══════════════════════════════════════════════════════════════════════════

Table ArrowIPCReader::readFile(const std::string& path,
                               const std::string& table_name) {
    auto file_result = arrow::io::ReadableFile::Open(path);
    if (!file_result.ok())
        throw std::runtime_error("Cannot open Arrow IPC file: " + path);

    auto reader_result = arrow::ipc::RecordBatchFileReader::Open(*file_result);
    if (!reader_result.ok())
        throw std::runtime_error("Cannot create IPC reader: " +
                                 reader_result.status().ToString());
    auto reader = *reader_result;

    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    for (int i = 0; i < reader->num_record_batches(); ++i) {
        auto batch_result = reader->ReadRecordBatch(i);
        if (batch_result.ok()) batches.push_back(*batch_result);
    }

    auto table_result = arrow::Table::FromRecordBatches(batches);
    if (!table_result.ok())
        throw std::runtime_error("Cannot assemble table: " +
                                 table_result.status().ToString());

    return ArrowConverter::fromArrowTable(*table_result, table_name);
}

// ═══════════════════════════════════════════════════════════════════════════
// IPC Writer
// ═══════════════════════════════════════════════════════════════════════════

void ArrowIPCWriter::writeFile(const Table& src, const std::string& path) {
    auto atbl = ArrowConverter::toArrowTable(src);

    auto file_result = arrow::io::FileOutputStream::Open(path);
    if (!file_result.ok())
        throw std::runtime_error("Cannot create output file: " + path);

    auto writer_result = arrow::ipc::MakeFileWriter(*file_result, atbl->schema());
    if (!writer_result.ok())
        throw std::runtime_error("Cannot create IPC writer");

    auto writer = *writer_result;
    auto rb = arrow::TableBatchReader(*atbl);
    std::shared_ptr<arrow::RecordBatch> batch;
    while (rb.ReadNext(&batch).ok() && batch)
        (void)writer->WriteRecordBatch(*batch);
    (void)writer->Close();
}

} // namespace gpudb

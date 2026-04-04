/**
 * @file csv_reader.h
 * @brief CSV file reader that populates a columnar Table according to a Schema.
 *
 * Design goals:
 *   1. Schema-driven parsing — each cell is type-converted according to the
 *      corresponding ColumnDef's DataType.
 *   2. True columnar population — data is accumulated into per-column vectors,
 *      NOT stored row-by-row.
 *   3. RFC-4180-style quoting — handles quoted fields containing commas,
 *      newlines, and escaped quotes ("").
 *   4. Nullable support — empty or missing fields in nullable columns are
 *      stored as null (via the TypedColumn null bitmap).
 *   5. Error reporting — parse errors carry the exact (row, column) position.
 */

#pragma once

#include "../core/schema.h"
#include "../core/table.h"

#include <cstddef>
#include <string>
#include <vector>

namespace gpudb {

// ─────────────────────────────────────────────────────────────────────────────
// ParseError — describes a single cell-level parse failure
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Records one parse error with its location in the CSV file.
 */
struct ParseError {
    std::size_t row;        ///< 1-based row number (after the header).
    std::size_t col;        ///< 0-based column index.
    std::string col_name;   ///< Column name from the schema.
    std::string cell_value; ///< The raw cell text that failed to parse.
    std::string message;    ///< Human-readable error description.
};

// ─────────────────────────────────────────────────────────────────────────────
// CSVReaderOptions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Configuration knobs for the CSV reader.
 */
struct CSVReaderOptions {
    char   delimiter     = ',';    ///< Field delimiter (default: comma).
    char   quote_char    = '"';    ///< Quote character for escaping.
    bool   has_header    = true;   ///< If true, first row is header (skipped as data).
    bool   strict        = false;  ///< If true, abort on the first parse error.
    std::size_t max_errors = 100;  ///< Stop collecting errors after this many.
};

// ─────────────────────────────────────────────────────────────────────────────
// CSVReader
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Reads a CSV file into a columnar Table according to a Schema.
 *
 * Usage:
 * @code
 *   Schema schema = SchemaBuilder()
 *       .addColumn("id",   DataType::INT64)
 *       .addColumn("name", DataType::STRING, true)
 *       .build();
 *
 *   CSVReader reader("data.csv", schema);
 *   Table table = reader.read("my_table");
 *
 *   if (reader.hasErrors()) {
 *       for (const auto& err : reader.errors()) {
 *           std::cerr << err.message << "\n";
 *       }
 *   }
 * @endcode
 */
class CSVReader {
public:
    /**
     * @brief Construct a CSVReader for the given file and schema.
     *
     * @param file_path  Path to the CSV file.
     * @param schema     Schema describing the expected column types.
     * @param options    Optional configuration overrides.
     */
    CSVReader(std::string file_path, Schema schema,
              CSVReaderOptions options = {});

    /**
     * @brief Read the entire CSV file and return a populated Table.
     *
     * @param table_name  The logical name to assign to the Table.
     * @return A Table containing all successfully parsed rows.
     *
     * @throws std::runtime_error if the file cannot be opened.
     * @throws std::runtime_error if strict mode is on and a parse error occurs.
     */
    [[nodiscard]] Table read(const std::string& table_name);

    // -- Error reporting -----------------------------------------------------

    /** Whether any parse errors occurred during the last read(). */
    [[nodiscard]] bool hasErrors() const noexcept;

    /** The list of parse errors from the last read(). */
    [[nodiscard]] const std::vector<ParseError>& errors() const noexcept;

    /** Total number of data rows successfully parsed. */
    [[nodiscard]] std::size_t rowsParsed() const noexcept;

private:
    std::string       file_path_;
    Schema            schema_;
    CSVReaderOptions  options_;

    std::vector<ParseError> errors_;
    std::size_t             rows_parsed_ = 0;

    // ── Internal helpers ────────────────────────────────────────────────────

    /**
     * @brief Split a single CSV line into fields, respecting quoting.
     *
     * Handles:
     *   - Quoted fields containing the delimiter.
     *   - Escaped quotes ("" inside a quoted field → single ").
     *   - Newlines inside quoted fields (caller must supply the full
     *     logical line, potentially spanning multiple physical lines).
     */
    std::vector<std::string> parseLine(const std::string& line) const;

    /**
     * @brief Convert a raw cell string to the appropriate type and push
     *        it into the correct typed column.
     *
     * @param cell       Raw cell text (already unquoted).
     * @param col_idx    Schema column index.
     * @param row_num    1-based row number (for error reporting).
     * @param columns    The per-column ColumnVariant vector being populated.
     *
     * @return true on success, false if a parse error was recorded.
     */
    bool convertAndPush(const std::string& cell, std::size_t col_idx,
                        std::size_t row_num, std::vector<ColumnVariant>& columns);

    /** Record a parse error (respecting max_errors). */
    void recordError(std::size_t row, std::size_t col,
                     const std::string& cell, const std::string& msg);

    /**
     * @brief Count unescaped quote characters in a line.
     *
     * Used to detect whether a logical CSV record spans multiple physical
     * lines (an odd count means the record is incomplete).
     */
    std::size_t countUnescapedQuotes(const std::string& line) const;
};

}  // namespace gpudb

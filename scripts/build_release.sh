#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# GPUDB — Linux/macOS build and release packaging script
# Usage:
#   chmod +x scripts/build_release.sh
#   ./scripts/build_release.sh 1.0.0           # full build
#   ./scripts/build_release.sh 1.0.0 --no-gpu  # CPU-only build
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

VERSION="${1:?Usage: $0 <version> [--no-gpu]}"
NO_GPU="${2:-}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  GPUDB Release Build — v${VERSION}                        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Clean and Build ──────────────────────────────────────────────────
echo "▸ Step 1: Clean and Build"

rm -rf build
mkdir build

USE_CUDA="ON"
if [[ "$NO_GPU" == "--no-gpu" ]] || ! command -v nvcc &>/dev/null; then
    USE_CUDA="OFF"
    echo "  Building CPU-only (CUDA not available or --no-gpu set)"
fi

echo "  Configuring CMake (USE_CUDA=$USE_CUDA) ..."
cmake -S . -B build \
    -DUSE_CUDA="$USE_CUDA" \
    -DBUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release

echo "  Compiling ..."
cmake --build build --config Release --parallel "$(nproc 2>/dev/null || echo 4)"
echo "  ✓ Build succeeded."

# ── Step 2: Verify and Generate ──────────────────────────────────────────────
echo ""
echo "▸ Step 2: Verify Executables"

FOUND=0
for exe in build/gpudb_demo build/gpudb_bench build/gpudb_cli build/gpudb_gpu_bench; do
    if [[ -f "$exe" ]]; then
        echo "  ✓ $(basename $exe)"
        FOUND=$((FOUND + 1))
    fi
done

if [[ $FOUND -eq 0 ]]; then
    echo "  ✗ No executables found!" && exit 1
fi

# Generate benchmark results if report tool exists
if [[ -f "build/gpudb_report" ]]; then
    echo "  Running performance report ..."
    ./build/gpudb_report || true
fi

# ── Step 3: Package ──────────────────────────────────────────────────────────
echo ""
echo "▸ Step 3: Package Release"

OS="linux"
ARCH="$(uname -m)"
ARCHIVE="cuda-gpu-db-v${VERSION}-${OS}-${ARCH}.tar.gz"
STAGING="release-staging"

rm -rf "$STAGING" "$ARCHIVE"
mkdir "$STAGING"

for exe in build/gpudb_demo build/gpudb_bench build/gpudb_cli build/gpudb_gpu_bench; do
    [[ -f "$exe" ]] && cp "$exe" "$STAGING/"
done
[[ -f README.md ]]            && cp README.md "$STAGING/"
[[ -f LICENSE ]]              && cp LICENSE "$STAGING/"
[[ -f release_notes.md ]]     && cp release_notes.md "$STAGING/"
[[ -f BENCHMARK_REPORT.md ]]  && cp BENCHMARK_REPORT.md "$STAGING/"
[[ -f benchmark_results.csv ]] && cp benchmark_results.csv "$STAGING/"

tar czf "$ARCHIVE" -C "$STAGING" .
rm -rf "$STAGING"

SIZE=$(du -h "$ARCHIVE" | cut -f1)
echo "  ✓ Created $ARCHIVE ($SIZE)"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  ✓ Release v${VERSION} packaged!                         ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "To publish:"
echo "  git tag v${VERSION}"
echo "  git push origin v${VERSION}"
echo "  # GitHub Actions will auto-create the release."

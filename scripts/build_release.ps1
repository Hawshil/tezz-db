<#
.SYNOPSIS
    Build GPUDB for release and package all artifacts into a .zip archive.

.DESCRIPTION
    Step 1: Clean build directory, configure CMake, compile Release build.
    Step 2: Verify executables, generate benchmark results.
    Step 3: Package everything into cuda-gpu-db-v<VERSION>-<OS>-<ARCH>.zip

.PARAMETER Version
    Release version tag, e.g. "1.0.0"

.PARAMETER SkipBenchmark
    Skip running the performance report generator (useful if no GPU present).

.EXAMPLE
    .\scripts\build_release.ps1 -Version "1.0.0"
    .\scripts\build_release.ps1 -Version "1.1.0" -SkipBenchmark
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$Version,

    [switch]$SkipBenchmark
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path "$ProjectRoot\CMakeLists.txt")) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
if (-not (Test-Path "$ProjectRoot\CMakeLists.txt")) {
    $ProjectRoot = $PSScriptRoot
}

# Resolve to the actual project root
Push-Location $ProjectRoot

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  GPUDB Release Build — v$Version                         ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Clean and Build ──────────────────────────────────────────────────

Write-Host "▸ Step 1: Clean and Build" -ForegroundColor Yellow

$BuildDir = "build"

if (Test-Path $BuildDir) {
    Write-Host "  Removing existing $BuildDir/ ..."
    Remove-Item -Recurse -Force $BuildDir
}

Write-Host "  Creating $BuildDir/ ..."
New-Item -ItemType Directory -Path $BuildDir | Out-Null

Write-Host "  Running CMake configuration (Release, CUDA=ON) ..."
cmake -S . -B $BuildDir `
    -DUSE_CUDA=ON `
    -DBUILD_TESTS=OFF `
    -DCMAKE_BUILD_TYPE=Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ CMake configuration failed." -ForegroundColor Red
    Write-Host "  Retrying without CUDA ..." -ForegroundColor Yellow
    cmake -S . -B $BuildDir `
        -DUSE_CUDA=OFF `
        -DBUILD_TESTS=OFF `
        -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ✗ CMake configuration failed (CPU-only too)." -ForegroundColor Red
        Pop-Location; exit 1
    }
}

Write-Host "  Compiling ..."
cmake --build $BuildDir --config Release --parallel
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ Build failed." -ForegroundColor Red
    Pop-Location; exit 1
}
Write-Host "  ✓ Build succeeded." -ForegroundColor Green

# ── Step 2: Verify and Generate Artifacts ────────────────────────────────────

Write-Host ""
Write-Host "▸ Step 2: Generate Artifacts" -ForegroundColor Yellow

# Find executables (handle both multi-config and single-config generators)
$SearchPaths = @("$BuildDir", "$BuildDir\Release", "$BuildDir\Debug")
$Benchmark = $null
$Cli = $null
$Demo = $null
$Report = $null

foreach ($sp in $SearchPaths) {
    if (-not $Benchmark -and (Test-Path "$sp\gpudb_bench.exe")) { $Benchmark = "$sp\gpudb_bench.exe" }
    if (-not $Benchmark -and (Test-Path "$sp\gpudb_gpu_bench.exe")) { $Benchmark = "$sp\gpudb_gpu_bench.exe" }
    if (-not $Cli -and (Test-Path "$sp\gpudb_cli.exe")) { $Cli = "$sp\gpudb_cli.exe" }
    if (-not $Demo -and (Test-Path "$sp\gpudb_demo.exe")) { $Demo = "$sp\gpudb_demo.exe" }
    if (-not $Report -and (Test-Path "$sp\gpudb_report.exe")) { $Report = "$sp\gpudb_report.exe" }
}

# Check CPU benchmark at minimum
if (-not $Benchmark -and -not $Demo) {
    Write-Host "  ✗ No executables found in $BuildDir!" -ForegroundColor Red
    Write-Host "  Searched: $($SearchPaths -join ', ')" -ForegroundColor Red
    Pop-Location; exit 1
}

Write-Host "  Found executables:"
if ($Demo)      { Write-Host "    • gpudb_demo:      $Demo" -ForegroundColor Gray }
if ($Benchmark) { Write-Host "    • gpudb_bench:     $Benchmark" -ForegroundColor Gray }
if ($Cli)       { Write-Host "    • gpudb_cli:       $Cli" -ForegroundColor Gray }
if ($Report)    { Write-Host "    • gpudb_report:    $Report" -ForegroundColor Gray }

# Generate benchmark results
$BenchResultsFile = "benchmark_results.csv"
if (-not $SkipBenchmark -and $Report) {
    Write-Host "  Running performance report generator ..."
    & $Report
    if (Test-Path "BENCHMARK_REPORT.md") {
        Write-Host "  ✓ Generated BENCHMARK_REPORT.md" -ForegroundColor Green
    }
    if (Test-Path $BenchResultsFile) {
        Write-Host "  ✓ Generated $BenchResultsFile" -ForegroundColor Green
    }
} elseif (-not $SkipBenchmark -and $Benchmark) {
    Write-Host "  Running CPU benchmark ..."
    & $Benchmark | Out-File -FilePath "benchmark_output.txt" -Encoding UTF8
    $BenchResultsFile = "benchmark_output.txt"
    Write-Host "  ✓ Generated benchmark_output.txt" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Skipping benchmark (no executable or -SkipBenchmark)" -ForegroundColor Yellow
}

# ── Step 3: Package Release Archive ──────────────────────────────────────────

Write-Host ""
Write-Host "▸ Step 3: Package Release Archive" -ForegroundColor Yellow

# Determine OS/arch tag
$OS = "windows"
$Arch = "x64"
$ArchiveName = "cuda-gpu-db-v$Version-$OS-$Arch.zip"
$StagingDir = "release-staging"

if (Test-Path $StagingDir) { Remove-Item -Recurse -Force $StagingDir }
New-Item -ItemType Directory -Path $StagingDir | Out-Null

# Copy executables
if ($Benchmark) { Copy-Item $Benchmark -Destination $StagingDir }
if ($Cli)       { Copy-Item $Cli -Destination $StagingDir }
if ($Demo)      { Copy-Item $Demo -Destination $StagingDir }

# Copy docs
if (Test-Path "README.md")           { Copy-Item "README.md" -Destination $StagingDir }
if (Test-Path "LICENSE")             { Copy-Item "LICENSE" -Destination $StagingDir }
if (Test-Path "BENCHMARK_REPORT.md") { Copy-Item "BENCHMARK_REPORT.md" -Destination $StagingDir }
if (Test-Path $BenchResultsFile)     { Copy-Item $BenchResultsFile -Destination $StagingDir }
if (Test-Path "release_notes.md")    { Copy-Item "release_notes.md" -Destination $StagingDir }

# Create zip
if (Test-Path $ArchiveName) { Remove-Item $ArchiveName }
Compress-Archive -Path "$StagingDir\*" -DestinationPath $ArchiveName -CompressionLevel Optimal
Remove-Item -Recurse -Force $StagingDir

$Size = [math]::Round((Get-Item $ArchiveName).Length / 1MB, 2)
Write-Host "  ✓ Created $ArchiveName ($Size MB)" -ForegroundColor Green

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  Release v$Version packaged successfully!                 ║" -ForegroundColor Green
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  git tag v$Version"
Write-Host "  git push origin v$Version"
Write-Host "  # → GitHub Actions will auto-build and create the release"
Write-Host ""

Pop-Location

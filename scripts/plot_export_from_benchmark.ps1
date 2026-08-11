<#
.SYNOPSIS
    Builds the paper plots, tables, and MuJoCo waypoint export from a benchmark folder.

.DESCRIPTION
    Thin PowerShell wrapper around `gcr paper-report`, kept for the documented
    Windows workflow. It runs solution selection, the paper report, and the
    waypoint export in that order; the export reordering flags live in
    src/gcr/pipelines.py (EXPORT_REORDER_FLAGS).

.EXAMPLE
    .\scripts\plot_export_from_benchmark.ps1 -BenchmarkDir ".\results\paper_benchmark_1000"
#>
param(
    [string]$BenchmarkDir = ".\results\benchmark_rerun",
    [string]$PaperOutputDir = ".\results\paper_report\generated",
    [string]$ExportDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$Arguments = @(
    "-m", "gcr", "paper-report", $BenchmarkDir,
    "--paper-output-dir", $PaperOutputDir
)
if (-not [string]::IsNullOrWhiteSpace($ExportDir)) {
    $Arguments += @("--export-dir", $ExportDir)
}

Push-Location $Root
try {
    python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "gcr paper-report failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

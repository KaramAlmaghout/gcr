param(
    [string]$BenchmarkDir = ".\results\benchmark_rerun",
    [string]$PaperOutputDir = ".\results\paper_report\generated",
    [string]$ExportDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Resolve-In-Package([string]$PathText) {
    if ([System.IO.Path]::IsPathRooted($PathText)) {
        return $PathText
    }
    return (Join-Path $Root $PathText)
}

$BenchmarkPath = Resolve-In-Package $BenchmarkDir
$PaperPath = Resolve-In-Package $PaperOutputDir
if ([string]::IsNullOrWhiteSpace($ExportDir)) {
    $ExportPath = Join-Path $BenchmarkPath "mujoco_selected_waypoints"
} else {
    $ExportPath = Resolve-In-Package $ExportDir
}

Push-Location $Root
try {
    python .\plot_cpp_solution_extremes.py $BenchmarkPath

    python .\generate_paper_results.py $BenchmarkPath --output-dir $PaperPath

    python .\export_cpp_selected_waypoints.py $BenchmarkPath $ExportPath `
      --endpoint-reorder `
      --allow-terminal-reversal `
      --endpoint-reorder-cost-ratio 0.75 `
      --endpoint-reorder-tangent-margin 0.05
}
finally {
    Pop-Location
}

Write-Host "Paper plots/tables: $PaperPath"
Write-Host "MuJoCo waypoint export: $ExportPath"

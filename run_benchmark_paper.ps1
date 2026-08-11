param(
    [string]$OutputDir = ".\results\benchmark_rerun",
    [int]$Runs = 1000,
    [int]$MaxIterations = 100000,
    [double]$ScenarioTimeLimit = 120.0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutputPath = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir
} else {
    Join-Path $Root $OutputDir
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputPath) | Out-Null

Push-Location (Join-Path $Root "cpp")
try {
    if (-not (Test-Path ".\build\gcr_benchmark.exe")) {
        .\build_msvc.bat
    }

    .\build\gcr_benchmark.exe $OutputPath `
      --scenario 2 `
      --scenario 3 `
      --scenario 5 `
      --scenario 7 `
      --scenario 8 `
      --scenario 9 `
      --scenario 10 `
      --scenario 11 `
      --scenario 12 `
      --scenario 14 `
      --runs $Runs `
      --max-iterations $MaxIterations `
      --scenario-time-limit $ScenarioTimeLimit `
      --endpoint-reorder `
      --allow-terminal-reversal `
      --endpoint-reorder-cost-ratio 0.75 `
      --endpoint-reorder-tangent-margin 0.05 `
      --robot-execution-feasibility `
      --robot-tcp-clearance 0.035 `
      --robot-obstacle-clearance 0.003 `
      --robot-swept-tcp-clearance 0.015 `
      --save-all-success-paths
}
finally {
    Pop-Location
}

Write-Host "Benchmark output: $OutputPath"

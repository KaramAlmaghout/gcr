<#
.SYNOPSIS
    Runs the C++ GCR benchmark with the paper planner configuration.

.DESCRIPTION
    Thin PowerShell wrapper around `gcr benchmark`, kept for the documented
    Windows workflow. The planner flags themselves live in
    src/gcr/cpp_benchmark.py (PAPER_PLANNER_FLAGS), so both entry points always
    run the same configuration.

.EXAMPLE
    .\scripts\run_benchmark_paper.ps1

.EXAMPLE
    .\scripts\run_benchmark_paper.ps1 -OutputDir ".\results\benchmark_rerun\smoke" -Runs 1 -MaxIterations 5000 -ScenarioTimeLimit 15
#>
param(
    [string]$OutputDir = ".\results\benchmark_rerun",
    [int]$Runs = 1000,
    [int]$MaxIterations = 100000,
    [double]$ScenarioTimeLimit = 120.0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Push-Location $Root
try {
    python -m gcr benchmark $OutputDir `
      --runs $Runs `
      --max-iterations $MaxIterations `
      --scenario-time-limit $ScenarioTimeLimit
    if ($LASTEXITCODE -ne 0) {
        throw "gcr benchmark failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

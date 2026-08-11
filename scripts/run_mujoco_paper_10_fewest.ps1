<#
.SYNOPSIS
    Replays the ten fewest-configuration solutions in MuJoCo.

.DESCRIPTION
    Thin PowerShell wrapper around `gcr paper-replay`, kept for the documented
    Windows workflow. The replay flags live in src/gcr/pipelines.py
    (REPLAY_FLAGS). Without -Interactive it records four-corner videos and
    settled cable-shape errors; with -Interactive it opens the viewer instead.

.EXAMPLE
    .\scripts\run_mujoco_paper_10_fewest.ps1 -Interactive
#>
param(
    [string]$BenchmarkDir = ".\results\paper_benchmark_1000",
    [string]$OutputDir = ".\results\replay_outputs",
    [int]$StepsPerWaypoint = 300,
    [double]$FinalErrorWait = 10.0,
    [double]$VideoMaxSeconds = 60.0,
    [switch]$Interactive
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$Arguments = @(
    "-m", "gcr", "paper-replay",
    "--benchmark-dir", $BenchmarkDir,
    "--output-dir", $OutputDir,
    "--steps-per-waypoint", "$StepsPerWaypoint",
    "--final-error-wait", "$FinalErrorWait",
    "--video-max-seconds", "$VideoMaxSeconds"
)
if ($Interactive) {
    $Arguments += "--interactive"
}

Push-Location $Root
try {
    python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "gcr paper-replay failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

param(
    [string]$BenchmarkDir = ".\results\paper_benchmark_1000",
    [string]$OutputDir = ".\results\replay_outputs",
    [int]$StepsPerWaypoint = 300,
    [double]$FinalErrorWait = 10.0,
    [double]$VideoMaxSeconds = 60.0,
    [switch]$Interactive
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
$OutPath = Resolve-In-Package $OutputDir
$WaypointDir = Join-Path $BenchmarkPath "mujoco_selected_waypoints"
$ErrorDir = Join-Path $OutPath "final_errors"
$VideoDir = Join-Path $OutPath "videos"
New-Item -ItemType Directory -Force -Path $ErrorDir | Out-Null
New-Item -ItemType Directory -Force -Path $VideoDir | Out-Null

$WaypointFiles = Get-ChildItem -LiteralPath $WaypointDir -Filter "*_fewest_configs_waypoints.csv" | Sort-Object Name
if ($WaypointFiles.Count -eq 0) {
    throw "No *_fewest_configs_waypoints.csv files found in $WaypointDir"
}

Push-Location $Root
try {
    foreach ($Waypoint in $WaypointFiles) {
        $Obstacle = Join-Path $WaypointDir ($Waypoint.BaseName -replace "_waypoints$", "_obstacles.csv")
        if (-not (Test-Path $Obstacle)) {
            throw "Missing obstacle CSV for $($Waypoint.Name): $Obstacle"
        }

        $FinalErrorJson = Join-Path $ErrorDir ($Waypoint.BaseName + "_final_error.json")
        $VideoPath = Join-Path $VideoDir ($Waypoint.BaseName + "_four_corners.mp4")

        $Args = @(
            ".\mujoco_exec\run_csv.py",
            "--csv", $Waypoint.FullName,
            "--obstacles", $Obstacle,
            "--endpoint-filter", "choose-global-swap",
            "--yaw-mode", "simultaneous",
            "--steps-per-waypoint", "$StepsPerWaypoint",
            "--cable-attachment-mode", "site_weld",
            "--cable-weld-torquescale", "6.0",
            "--final-error-wait", "$FinalErrorWait",
            "--final-error-json", $FinalErrorJson,
            "--tcp-schedule", "auto-stagger",
            "--side-correction-moving-robot", "right"
        )

        if (-not $Interactive) {
            $Args += @(
                "--no-viewer",
                "--video-four-corners",
                "--video-max-seconds", "$VideoMaxSeconds",
                "--video-path", $VideoPath
            )
        }

        Write-Host "Running MuJoCo replay: $($Waypoint.Name)"
        python @Args
    }
}
finally {
    Pop-Location
}

Write-Host "Final error JSON files: $ErrorDir"
if (-not $Interactive) {
    Write-Host "Four-corner MP4 videos: $VideoDir"
}

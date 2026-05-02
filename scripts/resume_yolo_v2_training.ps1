param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$RunName = "yolo11s_airborne_drone_vs_bird_v2",
    [string]$DatasetName = "airborne_yolo_v2",
    [string]$Python = "",
    [string]$TrainingRoot = "",
    [int]$Workers = 8
)

$ErrorActionPreference = "Stop"

if (-not $Python) {
    $pythonCandidates = @(
        (Join-Path $RepoRoot ".venv_train\Scripts\python.exe"),
        (Join-Path $RepoRoot ".venv\Scripts\python.exe")
    )
    $Python = ($pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1)
}

if (-not $Python) {
    throw "Training Python not found. Create .venv_train first, then install requirements."
}

if (-not $TrainingRoot) {
    $rootCandidates = @($RepoRoot)
    $worktreesDir = Join-Path $RepoRoot ".claude\worktrees"
    if (Test-Path -LiteralPath $worktreesDir) {
        $rootCandidates += Get-ChildItem -LiteralPath $worktreesDir -Directory | ForEach-Object { $_.FullName }
    }

    $TrainingRoot = $rootCandidates |
        Where-Object {
            (Test-Path -LiteralPath (Join-Path $_ "data\training\$DatasetName\data.yaml")) -and
            (Test-Path -LiteralPath (Join-Path $_ "data\training\runs\$RunName\weights\last.pt"))
        } |
        Sort-Object {
            (Get-Item -LiteralPath (Join-Path $_ "data\training\runs\$RunName\weights\last.pt")).LastWriteTime
        } -Descending |
        Select-Object -First 1
}

if (-not $TrainingRoot) {
    throw "Could not find data\training\$DatasetName\data.yaml and data\training\runs\$RunName\weights\last.pt under repo or .claude worktrees."
}

$RunDir = Join-Path $TrainingRoot "data\training\runs\$RunName"
$Checkpoint = Join-Path $RunDir "weights\last.pt"
$Log = Join-Path $RunDir ("resume_training_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "Checkpoint not found: $Checkpoint"
}

$env:YOLO_CONFIG_DIR = $TrainingRoot
$env:PYTHONUNBUFFERED = "1"
Set-Location $TrainingRoot

Start-Transcript -Path $Log -Force
try {
    Write-Host "SkyScouter YOLO v2 resume"
    Write-Host "Training root: $TrainingRoot"
    Write-Host "Python: $Python"
    Write-Host "Checkpoint: $Checkpoint"
    Write-Host "Log: $Log"
    Write-Host "Workers: $Workers"
    $checkpointRelative = "data\training\runs\$RunName\weights\last.pt"
    & $Python -c "from ultralytics import YOLO; YOLO(r'$checkpointRelative').train(resume=True, workers=$Workers)"
}
finally {
    Stop-Transcript
}

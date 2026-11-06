param(
    [int]$MaxEvalsPerMethod = 220,
    [int]$SequenceExactPerRegion = 18,
    [int]$SequenceRobustEvals = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"


function Invoke-PythonLogged {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Python,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$LogPath
    )

    # Python libraries may write harmless RuntimeWarning messages to stderr.
    # Windows PowerShell can convert such stderr lines into ErrorRecord objects;
    # with ErrorActionPreference=Stop that incorrectly terminates the pipeline.
    # Temporarily continue, stream every line to screen and log, then use the
    # native process exit code as the only success/failure criterion.
    Set-Content -Path $LogPath -Value "" -Encoding UTF8

    $OldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"

        & $Python @Arguments 2>&1 |
            ForEach-Object {
                $Line = $_.ToString()
                Write-Host $Line
                Add-Content -Path $LogPath -Value $Line -Encoding UTF8
            }

        $ExitCode = [int]$LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $OldPreference
    }

    return $ExitCode
}

$PY = "D:\Soft\tools\anaconda\envs\spatialvla\python.exe"
$PROJECT = "D:\Project\wrs-sealp"
$LAYOUT = "$PROJECT\sealp\examples\layout"
$OUT = "$LAYOUT\_output"
$LOGROOT = "$PROJECT\logs"
$IK = "$PROJECT\hint_cache\tower_ik_hint_r1row.npz"

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = "$LOGROOT\traditional_overnight_$Stamp"
New-Item -ItemType Directory -Force $RunDir | Out-Null
New-Item -ItemType Directory -Force $OUT | Out-Null

$SequenceName = "tower_sequence_basin_overnight_$Stamp"
$GlobalName = "tower_global_overnight_$Stamp"
$SequenceReport = "$OUT\${SequenceName}_basin_report.json"
$SequenceLog = "$RunDir\01_sequence_basin.log"
$GlobalLog = "$RunDir\02_global_baseline.log"
$Summary = "$RunDir\run_manifest.json"

Write-Host "============================================================"
Write-Host "Traditional overnight benchmark"
Write-Host "Run directory : $RunDir"
Write-Host "IK hint       : $IK"
Write-Host "Budget/method : $MaxEvalsPerMethod"
Write-Host "============================================================"

if (-not (Test-Path $PY)) { throw "Python not found: $PY" }
if (-not (Test-Path $IK)) { throw "Merged IK hint not found: $IK" }
if (-not (Test-Path "$LAYOUT\find_optimal_initial_layout_tower_sequence_basin.py")) {
    throw "Sequence-basin script missing in $LAYOUT"
}

# Prevent AC sleep/hibernate where permissions allow it. Display timeout is untouched.
try {
    powercfg /change standby-timeout-ac 0 | Out-Null
    powercfg /change hibernate-timeout-ac 0 | Out-Null
    Write-Host "[OK] AC sleep and hibernate disabled for the overnight run."
}
catch {
    Write-Warning "Could not change power settings. Manually set sleep to Never."
}

$Common = @(
    "--asmdef", "$PROJECT\sealp\assembly_sequence\_demo_output\topdown_tower.asmdef",
    "--config", "$PROJECT\sealp\config\sample_config.yaml",
    "--grasp-dir", "$PROJECT\sealp\examples\grasp\tower_grasp",
    "--output-dir", $OUT,
    "--seed", "0",
    "--cdprim-type", "box",
    "--planner-obstacle-mode", "staging_aware",
    "--w-grasp", "0.25",
    "--w-manip", "0.50",
    "--w-dist", "0.05",
    "--w-rot", "0.20"
)

$Failures = @()
$Started = Get-Date

# ------------------------------------------------------------
# 1) Proposed traditional method:
#    balanced per-region exact evaluation + local refinement
#    + full-grasp Top-5 certification + 1 cm perturbation tests.
# ------------------------------------------------------------
try {
    Write-Host "`n[START] Sequence Basin formal overnight run"
    $SequenceArgs = @(
        "-m", "sealp.examples.layout.find_optimal_initial_layout_tower_sequence_basin"
    ) + $Common + @(
        "--output-name", $SequenceName,
        "--n-samples", "20",
        "--basin-ik-hint", $IK,
        "--basin-beam-width", "48",
        "--basin-part-candidates", "20",
        "--basin-exact-proposals", ([string](3 * $SequenceExactPerRegion)),
        "--basin-exact-per-region", ([string]$SequenceExactPerRegion),
        "--basin-elite", "3",
        "--basin-margin-cap", "0.12",
        "--basin-margin-weight", "0.45",
        "--basin-hint-weight", "0.40",
        "--basin-distance-weight", "0.15",
        "--basin-nms-cells", "2",
        "--basin-fallback-explore", "30",
        "--basin-certify-topk", "5",
        "--basin-certify-grasp-cap", "0",
        "--basin-robust-evals", ([string]$SequenceRobustEvals),
        "--basin-robust-sigma", "0.01",
        "--basin-report-json", $SequenceReport,
        "--global-refine-steps", "0.03,0.015,0.008",
        "--global-refine-rounds", "2",
        "--global-max-evals", ([string]$MaxEvalsPerMethod)
    )
    $SequenceExitCode = Invoke-PythonLogged `
        -Python $PY `
        -Arguments $SequenceArgs `
        -LogPath $SequenceLog

    if ($SequenceExitCode -ne 0) {
        throw "Sequence Basin exited with code $SequenceExitCode"
    }
    Write-Host "[DONE] Sequence Basin"
}
catch {
    $Failures += "sequence_basin: $($_.Exception.Message)"
    Write-Warning $_.Exception.Message
}

# ------------------------------------------------------------
# 2) Existing traditional global-pattern baseline at same budget.
# ------------------------------------------------------------
try {
    Write-Host "`n[START] Global + Pattern Search baseline"
    $GlobalArgs = @(
        "-m", "sealp.examples.layout.find_optimal_initial_layout_tower_global"
    ) + $Common + @(
        "--output-name", $GlobalName,
        "--n-samples", "80",
        "--global-explore", "80",
        "--global-elite", "3",
        "--global-refine-steps", "0.03,0.015,0.008",
        "--global-refine-rounds", "2",
        "--global-max-evals", ([string]$MaxEvalsPerMethod)
    )
    $GlobalExitCode = Invoke-PythonLogged `
        -Python $PY `
        -Arguments $GlobalArgs `
        -LogPath $GlobalLog

    if ($GlobalExitCode -ne 0) {
        throw "Global baseline exited with code $GlobalExitCode"
    }
    Write-Host "[DONE] Global baseline"
}
catch {
    $Failures += "global_baseline: $($_.Exception.Message)"
    Write-Warning $_.Exception.Message
}

$Finished = Get-Date
$Manifest = [ordered]@{
    started_at = $Started.ToString("o")
    finished_at = $Finished.ToString("o")
    duration_hours = [math]::Round(($Finished - $Started).TotalHours, 3)
    max_evals_per_method = $MaxEvalsPerMethod
    sequence_exact_per_region = $SequenceExactPerRegion
    sequence_robust_evals = $SequenceRobustEvals
    ik_hint = $IK
    sequence_output_name = $SequenceName
    global_output_name = $GlobalName
    sequence_report = $SequenceReport
    sequence_log = $SequenceLog
    global_log = $GlobalLog
    failures = $Failures
}
$Manifest | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $Summary

Write-Host "`n============================================================"
Write-Host "Overnight benchmark finished"
Write-Host "Manifest: $Summary"
Write-Host "Sequence report: $SequenceReport"
Write-Host "Logs: $RunDir"
if ($Failures.Count -gt 0) {
    Write-Warning ("Failures: " + ($Failures -join " | "))
}
else {
    Write-Host "[OK] Both methods completed."
}
Write-Host "============================================================"

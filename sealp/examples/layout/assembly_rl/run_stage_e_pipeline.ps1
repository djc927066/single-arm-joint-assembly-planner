param(
    [switch]$ForceRebuildIk,
    [switch]$ForceRecollect,
    [switch]$ForceRetrain
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PY = "D:\Soft\tools\anaconda\envs\spatialvla\python.exe"
$PROJECT = "D:\Project\wrs-sealp"
$RL_DST = $PSScriptRoot

$ASMDEF = "$PROJECT\sealp\assembly_sequence\_demo_output\topdown_tower.asmdef"
$CONFIG = "$PROJECT\sealp\config\sample_config.yaml"
$GRASP_DIR = "$PROJECT\sealp\examples\grasp\tower_grasp"
$GRASP_HINT = "$PROJECT\hint_cache\tower_grasp_hint.npz"

$IK_R1C0 = "$PROJECT\hint_cache\tower_ik_hint_r1c0.npz"
$IK_R1C0_JSON = "$PROJECT\hint_cache\tower_ik_hint_r1c0.json"
$IK_R1C1 = "$PROJECT\hint_cache\tower_ik_hint_r1c1.npz"
$IK_R1C2 = "$PROJECT\hint_cache\tower_ik_hint_r1c2.npz"
$IK_R1C2_JSON = "$PROJECT\hint_cache\tower_ik_hint_r1c2.json"
$IK_MERGED = "$PROJECT\hint_cache\tower_ik_hint_r1row.npz"
$IK_MERGED_JSON = "$PROJECT\hint_cache\tower_ik_hint_r1row.json"

$BASE_DATASET = "$PROJECT\datasets\totem_r1c1_arm_bc_v1"
$MULTI_DATASET = "$PROJECT\datasets\totem_r1row_arm_bc_v2"
$CHECKPOINT_DIR = "$PROJECT\checkpoints\arm_bc_stage_e_r1row_overfit"
$BEST_CHECKPOINT = "$CHECKPOINT_DIR\arm_bc_best.pt"

$LOG_DIR = "$PROJECT\logs"
New-Item -ItemType Directory -Force $LOG_DIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$TRANSCRIPT = "$LOG_DIR\stage_e_pipeline_$STAMP.log"

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string[]]$Arguments
    )

    Write-Host ""
    Write-Host ("=" * 90)
    Write-Host "[START] $Name"
    Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host ("=" * 90)

    & $PY @Arguments
    $Code = $LASTEXITCODE

    if ($Code -ne 0) {
        throw "$Name failed with exit code $Code"
    }

    Write-Host "[DONE]  $Name"
    Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

function Test-DatasetCompleted {
    param([string]$DatasetDir)

    $IndexPath = Join-Path $DatasetDir "index.json"
    if (-not (Test-Path $IndexPath)) {
        return $false
    }

    try {
        $Index = Get-Content $IndexPath -Raw -Encoding UTF8 | ConvertFrom-Json
        return ($Index.completed_target -eq $true)
    }
    catch {
        return $false
    }
}

function Remove-OutputSafely {
    param(
        [string]$Path,
        [string]$Description
    )

    if (Test-Path $Path) {
        Write-Host "[REMOVE] $Description : $Path"
        Remove-Item $Path -Recurse -Force
    }
}

Start-Transcript -Path $TRANSCRIPT -Force

try {
    Write-Host "Stage-E sequential pipeline"
    Write-Host "Python      : $PY"
    Write-Host "Project     : $PROJECT"
    Write-Host "Transcript  : $TRANSCRIPT"
    Write-Host "Script dir  : $RL_DST"

    $ExpectedRlDir = "$PROJECT\sealp\examples\layout\assembly_rl"
    if ([System.IO.Path]::GetFullPath($RL_DST).TrimEnd('\') -ne
        [System.IO.Path]::GetFullPath($ExpectedRlDir).TrimEnd('\')) {
        throw @"
This script must be stored in:
$ExpectedRlDir

Current script directory:
$RL_DST
"@
    }

    # The PowerShell script and all Stage-E Python modules are stored in the
    # same assembly_rl directory. Verify them in place; no copying is needed.
    $StageEFiles = @(
        "precompute_ik_hint.py",
        "merge_ik_hint_caches.py",
        "collect_multiregion_arm_demonstrations.py",
        "inspect_multiregion_arm_demo_dataset.py"
    )

    foreach ($FileName in $StageEFiles) {
        $InstalledFile = Join-Path $RL_DST $FileName
        if (-not (Test-Path $InstalledFile)) {
            throw "Missing Stage-E helper file: $InstalledFile"
        }
    }

    Write-Host "[OK] Stage-E helper modules found in $RL_DST"

    foreach ($RequiredPath in @(
        $PY,
        $ASMDEF,
        $CONFIG,
        $GRASP_DIR,
        $GRASP_HINT,
        $IK_R1C1,
        $BASE_DATASET
    )) {
        if (-not (Test-Path $RequiredPath)) {
            throw "Required path does not exist: $RequiredPath"
        }
    }

    # -------------------------------------------------------------------------
    # Step 1: r1_c0 IK cache
    # -------------------------------------------------------------------------
    if ($ForceRebuildIk) {
        Remove-OutputSafely $IK_R1C0 "old r1_c0 IK NPZ"
        Remove-OutputSafely $IK_R1C0_JSON "old r1_c0 IK JSON"
    }

    if ((Test-Path $IK_R1C0) -and (Test-Path $IK_R1C0_JSON)) {
        Write-Host "[SKIP] r1_c0 IK cache already exists."
    }
    else {
        Invoke-PythonStep "Generate r1_c0 IK Hint" @(
            "-m", "sealp.examples.layout.assembly_rl.precompute_ik_hint",
            "--asmdef", $ASMDEF,
            "--config", $CONFIG,
            "--grasp-dir", $GRASP_DIR,
            "--regions", "r1_c0",
            "--resolution", "0.02",
            "--max-poses", "16",
            "--max-grasps", "32",
            "--grid-stride", "2",
            "--cdprim-type", "box",
            "--planner-obstacle-mode", "staging_aware",
            "--seed", "0",
            "--output-npz", $IK_R1C0,
            "--output-json", $IK_R1C0_JSON
        )
    }

    # -------------------------------------------------------------------------
    # Step 2: r1_c2 IK cache
    # -------------------------------------------------------------------------
    if ($ForceRebuildIk) {
        Remove-OutputSafely $IK_R1C2 "old r1_c2 IK NPZ"
        Remove-OutputSafely $IK_R1C2_JSON "old r1_c2 IK JSON"
    }

    if ((Test-Path $IK_R1C2) -and (Test-Path $IK_R1C2_JSON)) {
        Write-Host "[SKIP] r1_c2 IK cache already exists."
    }
    else {
        Invoke-PythonStep "Generate r1_c2 IK Hint" @(
            "-m", "sealp.examples.layout.assembly_rl.precompute_ik_hint",
            "--asmdef", $ASMDEF,
            "--config", $CONFIG,
            "--grasp-dir", $GRASP_DIR,
            "--regions", "r1_c2",
            "--resolution", "0.02",
            "--max-poses", "16",
            "--max-grasps", "32",
            "--grid-stride", "2",
            "--cdprim-type", "box",
            "--planner-obstacle-mode", "staging_aware",
            "--seed", "0",
            "--output-npz", $IK_R1C2,
            "--output-json", $IK_R1C2_JSON
        )
    }

    # -------------------------------------------------------------------------
    # Step 3: merge three IK caches. This is fast, so always rerun it.
    # -------------------------------------------------------------------------
    Invoke-PythonStep "Merge r1_c0, r1_c1 and r1_c2 IK Hints" @(
        "-m", "sealp.examples.layout.assembly_rl.merge_ik_hint_caches",
        "--inputs", $IK_R1C0, $IK_R1C1, $IK_R1C2,
        "--output-npz", $IK_MERGED,
        "--output-json", $IK_MERGED_JSON
    )

    # Validate merged region order and tensor shape.
    Invoke-PythonStep "Validate merged IK Hint" @(
        "-c",
        "import numpy as np; p=r'$IK_MERGED'; z=np.load(p,allow_pickle=True); regions=[str(x) for x in z['region_ids'].tolist()]; assert regions==['r1_c0','r1_c1','r1_c2'], regions; assert z['scores'].shape==(3,6,16,54,24), z['scores'].shape; assert np.allclose(z['scores'],np.maximum(z['left_scores'],z['right_scores']),atol=1e-3,equal_nan=True); print('[OK] merged regions=',regions,'scores=',z['scores'].shape)"
    )

    # -------------------------------------------------------------------------
    # Step 4: collect r1_c0/r1_c2 demonstrations and copy existing r1_c1 data.
    # -------------------------------------------------------------------------
    if ($ForceRecollect) {
        Remove-OutputSafely $MULTI_DATASET "old multi-region dataset"
    }

    if (Test-DatasetCompleted $MULTI_DATASET) {
        Write-Host "[SKIP] Multi-region dataset is already marked completed."
    }
    else {
        if ((Test-Path $MULTI_DATASET) -and
            ((Get-ChildItem $MULTI_DATASET -Force | Measure-Object).Count -gt 0)) {
            throw @"
The multi-region dataset directory exists but is incomplete:
$MULTI_DATASET

Run this script again with -ForceRecollect to delete it and recollect from scratch.
"@
        }

        Invoke-PythonStep "Collect multi-region strict-arm demonstrations" @(
            "-m", "sealp.examples.layout.assembly_rl.collect_multiregion_arm_demonstrations",
            "--asmdef", $ASMDEF,
            "--config", $CONFIG,
            "--grasp-dir", $GRASP_DIR,
            "--grasp-hint", $GRASP_HINT,
            "--ik-hint", $IK_MERGED,
            "--regions", "r1_c0", "r1_c2",
            "--base-dataset", $BASE_DATASET,
            "--output-dir", $MULTI_DATASET,
            "--target-successes-per-region", "3",
            "--max-attempted-per-region", "7",
            "--min-layout-score", "0.30",
            "--proposal-top-k", "128",
            "--sample-top-k", "16",
            "--temperature", "0.03",
            "--w-grasp", "0.05",
            "--w-ik", "0.10",
            "--w-distance", "0.05",
            "--grasp-reward-weight", "0.05",
            "--ik-reward-weight", "0.10",
            "--resolution", "0.02",
            "--max-parts", "12",
            "--max-poses", "16",
            "--cdprim-type", "box",
            "--planner-obstacle-mode", "staging_aware",
            "--seed", "300"
        )
    }

    # Never train before validating the collected dataset.
    Invoke-PythonStep "Validate multi-region demonstration dataset" @(
        "-m", "sealp.examples.layout.assembly_rl.inspect_multiregion_arm_demo_dataset",
        "--dataset-dir", $MULTI_DATASET
    )

    # -------------------------------------------------------------------------
    # Step 5: train multi-region BC.
    # -------------------------------------------------------------------------
    if ($ForceRetrain) {
        Remove-OutputSafely $CHECKPOINT_DIR "old Stage-E checkpoint directory"
    }

    if (Test-Path $BEST_CHECKPOINT) {
        Write-Host "[SKIP] Best checkpoint already exists: $BEST_CHECKPOINT"
    }
    else {
        if ((Test-Path $CHECKPOINT_DIR) -and
            ((Get-ChildItem $CHECKPOINT_DIR -Force | Measure-Object).Count -gt 0)) {
            throw @"
The checkpoint directory exists but has no completed best checkpoint:
$CHECKPOINT_DIR

Run this script again with -ForceRetrain to delete it and train from scratch.
"@
        }

        Invoke-PythonStep "Train multi-region masked BC" @(
            "-m", "sealp.examples.layout.assembly_rl.train_arm_bc",
            "--dataset-dir", $MULTI_DATASET,
            "--output-dir", $CHECKPOINT_DIR,
            "--epochs", "200",
            "--batch-size", "4",
            "--learning-rate", "0.0003",
            "--weight-decay", "0.00001",
            "--overfit-all",
            "--seed", "0",
            "--device", "auto",
            "--num-workers", "0",
            "--log-every", "10",
            "--patience", "80"
        )
    }

    # Quick checkpoint reload test.
    Invoke-PythonStep "Reload and evaluate Stage-E checkpoint offline" @(
        "-m", "sealp.examples.layout.assembly_rl.evaluate_arm_bc_offline",
        "--dataset-dir", $MULTI_DATASET,
        "--checkpoint", $BEST_CHECKPOINT,
        "--batch-size", "4",
        "--device", "auto"
    )

    Write-Host ""
    Write-Host ("#" * 90)
    Write-Host "[ALL DONE] Stage-E pipeline completed successfully."
    Write-Host "Merged IK cache : $IK_MERGED"
    Write-Host "Dataset         : $MULTI_DATASET"
    Write-Host "Checkpoint      : $BEST_CHECKPOINT"
    Write-Host "Log             : $TRANSCRIPT"
    Write-Host ("#" * 90)
}
catch {
    Write-Host ""
    Write-Host ("!" * 90)
    Write-Host "[PIPELINE FAILED]"
    Write-Host $_
    Write-Host "Full log: $TRANSCRIPT"
    Write-Host ("!" * 90)
    exit 1
}
finally {
    Stop-Transcript | Out-Null
}

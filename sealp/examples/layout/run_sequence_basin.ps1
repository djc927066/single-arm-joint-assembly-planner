param(
    [string]$IkHint = "",
    [int]$MaxEvals = 120,
    [int]$BeamWidth = 24,
    [int]$PartCandidates = 14,
    [int]$ExactProposals = 30,
    [int]$RobustEvals = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PY = "D:\Soft\tools\anaconda\envs\spatialvla\python.exe"
$PROJECT = "D:\Project\wrs-sealp"
$LAYOUT_DIR = "$PROJECT\sealp\examples\layout"
$SCRIPT = "$LAYOUT_DIR\find_optimal_initial_layout_tower_sequence_basin.py"

$MERGED = "$PROJECT\hint_cache\tower_ik_hint_r1row.npz"
$SINGLE = "$PROJECT\hint_cache\tower_ik_hint_r1c1.npz"

if (-not $IkHint) {
    if (Test-Path $MERGED) {
        $IkHint = $MERGED
    }
    elseif (Test-Path $SINGLE) {
        $IkHint = $SINGLE
    }
    else {
        throw "No IK hint found: $MERGED or $SINGLE"
    }
}

if (-not (Test-Path $SCRIPT)) {
    throw "Missing search script: $SCRIPT"
}
if (-not (Test-Path $IkHint)) {
    throw "Missing IK hint: $IkHint"
}

$OutputName = "tower_sequence_basin"
$Report = "$LAYOUT_DIR\_output\${OutputName}_basin_report.json"

Write-Host "IK Hint        : $IkHint"
Write-Host "Max evaluations: $MaxEvals"
Write-Host "Beam width     : $BeamWidth"
Write-Host "Exact proposals: $ExactProposals"

& $PY `
  -m sealp.examples.layout.find_optimal_initial_layout_tower_sequence_basin `
  --asmdef "$PROJECT\sealp\assembly_sequence\_demo_output\topdown_tower.asmdef" `
  --config "$PROJECT\sealp\config\sample_config.yaml" `
  --grasp-dir "$PROJECT\sealp\examples\grasp\tower_grasp" `
  --output-name $OutputName `
  --output-dir "$LAYOUT_DIR\_output" `
  --n-samples 20 `
  --seed 0 `
  --cdprim-type box `
  --planner-obstacle-mode staging_aware `
  --basin-ik-hint $IkHint `
  --basin-beam-width $BeamWidth `
  --basin-part-candidates $PartCandidates `
  --basin-exact-proposals $ExactProposals `
  --basin-elite 3 `
  --basin-margin-cap 0.08 `
  --basin-margin-weight 0.45 `
  --basin-hint-weight 0.40 `
  --basin-distance-weight 0.15 `
  --basin-nms-cells 2 `
  --basin-fallback-explore 20 `
  --basin-robust-evals $RobustEvals `
  --basin-robust-sigma 0.01 `
  --basin-report-json $Report `
  --global-refine-steps 0.03,0.015,0.008 `
  --global-refine-rounds 1 `
  --global-max-evals $MaxEvals

if ($LASTEXITCODE -ne 0) {
    throw "Sequence-basin search failed with exit code $LASTEXITCODE"
}

Write-Host "[OK] report: $Report"

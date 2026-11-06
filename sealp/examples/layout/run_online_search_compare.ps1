# Batch tower online layout search compare (Global / MLP / DeepSets / GCN / GAT).
# Usage: .\sealp\examples\layout\run_online_search_compare.ps1
#        .\sealp\examples\layout\run_online_search_compare.ps1 -DryRun -Methods global,mlp,deepsets

[CmdletBinding()]
param(
    [string[]] $Methods = @("global", "mlp", "deepsets"),
    [int] $Seed = 0,
    [int] $MaxEvals = 200,
    [int] $TopKProposals = 32,
    [int] $ScorerPool = 400,
    [string] $StationMode = "grid3x3",
    [string] $CdprimType = "box",
    [int] $GlobalElite = 3,
    [string] $GlobalRefineSteps = "0.03,0.015,0.008",
    [switch] $NoRefine,
    [string] $ResultsDir = "",
    [switch] $DryRun,
    [switch] $SkipMissingCheckpoint
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
}

function Write-Info([string] $Msg) {
    Write-Host "[compare] $Msg" -ForegroundColor Cyan
}

function Write-Warn([string] $Msg) {
    Write-Host "[compare] WARN: $Msg" -ForegroundColor Yellow
}

function Get-MethodConfig {
    param(
        [string] $Method,
        [string] $RepoRoot
    )

    $ckptV3 = Join-Path $RepoRoot "checkpoints\layout_models_transfer_v3"
    $ckptV3Auto = Join-Path $RepoRoot "checkpoints\layout_models_transfer_v3_auto"
    $ckptV2 = Join-Path $RepoRoot "checkpoints\layout_models_transfer_v2"

    switch ($Method.ToLower()) {
        "global" {
            return @{
                Kind = "global"
                Module = "sealp.examples.layout.find_optimal_initial_layout_tower_global"
                Checkpoint = $null
            }
        }
        "mlp" {
            return @{
                Kind = "neural"
                Module = "sealp.examples.layout.find_optimal_initial_layout_tower_neural"
                Model = "mlp"
                Checkpoint = Join-Path $ckptV3 "mlp_stratified\mlp_best.pt"
            }
        }
        "deepsets" {
            return @{
                Kind = "neural"
                Module = "sealp.examples.layout.find_optimal_initial_layout_tower_neural"
                Model = "deepsets"
                Checkpoint = @(
                    Join-Path $ckptV3 "deepsets_stratified\deepsets_best.pt"
                    Join-Path $ckptV3Auto "deepsets_stratified\deepsets_best.pt"
                ) | Where-Object { Test-Path $_ } | Select-Object -First 1
            }
        }
        "gcn" {
            return @{
                Kind = "neural"
                Module = "sealp.examples.layout.find_optimal_initial_layout_tower_neural"
                Model = "gcn"
                Checkpoint = Join-Path $ckptV2 "gcn_stratified\gcn_best.pt"
            }
        }
        "gat" {
            return @{
                Kind = "neural"
                Module = "sealp.examples.layout.find_optimal_initial_layout_tower_neural"
                Model = "gat"
                Checkpoint = Join-Path $ckptV2 "gat_stratified\gat_best.pt"
            }
        }
        default {
            throw "Unknown method: $Method (supported: global,mlp,deepsets,gcn,gat)"
        }
    }
}

function Build-Command {
    param(
        [hashtable] $Cfg,
        [string] $OutputName,
        [int] $Seed,
        [int] $MaxEvals,
        [int] $TopKProposals,
        [int] $ScorerPool,
        [string] $StationMode,
        [string] $CdprimType,
        [int] $GlobalElite,
        [string] $GlobalRefineSteps,
        [switch] $NoRefine
    )

    $pyArgs = @(
        "-m", $Cfg.Module,
        "--output-name", $OutputName,
        "--seed", "$Seed",
        "--cdprim-type", $CdprimType
    )

    if ($Cfg.Kind -eq "global") {
        $pyArgs += @(
            "--global-max-evals", "$MaxEvals",
            "--global-elite", "$GlobalElite",
            "--global-refine-steps", $GlobalRefineSteps
        )
    }
    else {
        if (-not $Cfg.Checkpoint) {
            throw "Checkpoint not found for model $($Cfg.Model)"
        }
        $pyArgs += @(
            "--model", $Cfg.Model,
            "--checkpoint", $Cfg.Checkpoint,
            "--global-max-evals", "$MaxEvals",
            "--top-k-proposals", "$TopKProposals",
            "--scorer-pool", "$ScorerPool",
            "--global-elite", "$GlobalElite",
            "--global-refine-steps", $GlobalRefineSteps,
            "--station-mode", $StationMode
        )
    }

    if ($NoRefine) {
        $pyArgs += "--no-refine"
    }

    return @("python") + $pyArgs
}

function Parse-LogMetrics {
    param(
        [string] $Text,
        [string] $Kind
    )

    $metrics = [ordered]@{
        best_score_stdout = $null
        real_evals = $null
        cache_hits = $null
        feasible_found = $null
        first_l2_ok_eval = $null
        best_region = $null
        wall_time_s = $null
        l3_passed = $false
        failed = $false
        best_score_eval = $null
    }

    if ($Text -match '\[BEST-L2\] score=([\d.+-]+)') {
        $metrics.best_score_stdout = [double]$Matches[1]
    }
    if ($Text -match 'real evaluations\s*=\s*(\d+)') {
        $metrics.real_evals = [int]$Matches[1]
    }
    if ($Text -match 'eval cache hits\s*=\s*(\d+)') {
        $metrics.cache_hits = [int]$Matches[1]
    }
    if ($Text -match 'feasible found\s*=\s*(\d+)') {
        $metrics.feasible_found = [int]$Matches[1]
    }
    if ($Text -match '\[BEST-L2\] score=[\d.+-]+ region=(\S+)') {
        $metrics.best_region = $Matches[1]
    }
    if ($Text -match 'wall-clock total = ([\d.]+)s') {
        $metrics.wall_time_s = [double]$Matches[1]
    }
    if ($Text -match '\[OK\] L3 passed') {
        $metrics.l3_passed = $true
    }
    if ($Text -match '\[FAIL\]') {
        $metrics.failed = $true
    }

    if ($Kind -eq "neural") {
        $tag = "score"
    }
    else {
        $tag = "explore"
    }

    $firstOk = [regex]::Match($Text, "\[$tag\]\s+(\d+)/\d+\s+L2_OK")
    if ($firstOk.Success) {
        $metrics.first_l2_ok_eval = [int]$firstOk.Groups[1].Value
    }

    if ($Text -match 'first L2_OK at eval # = (\d+)') {
        $metrics.first_l2_ok_eval = [int]$Matches[1]
    }
    if ($Text -match 'best score first at\s+= #(\d+)') {
        $metrics.best_score_eval = [int]$Matches[1]
    }

    return $metrics
}

function Read-DebugMetrics {
    param([string] $DebugPath)

    $out = [ordered]@{
        best_score = $null
        assembly_region_id = $null
        assembly_region_rc = $null
        first_l2_ok_eval = $null
        best_score_eval = $null
    }

    if (-not (Test-Path $DebugPath)) {
        return $out
    }

    try {
        $dbg = Get-Content -Raw -Encoding UTF8 $DebugPath | ConvertFrom-Json
        if ($null -ne $dbg.score) {
            $out.best_score = [double]$dbg.score
        }
        if ($null -ne $dbg.assembly_region_id) {
            $out.assembly_region_id = [string]$dbg.assembly_region_id
        }
        if ($null -ne $dbg.assembly_region_rc) {
            $out.assembly_region_rc = ($dbg.assembly_region_rc -join ",")
        }
        if ($null -ne $dbg.search_eval_stats) {
            if ($null -ne $dbg.search_eval_stats.first_l2_ok_eval) {
                $out.first_l2_ok_eval = [int]$dbg.search_eval_stats.first_l2_ok_eval
            }
            if ($null -ne $dbg.search_eval_stats.best_score_eval) {
                $out.best_score_eval = [int]$dbg.search_eval_stats.best_score_eval
            }
        }
    }
    catch {
        Write-Warn "Failed to parse debug json: $DebugPath ($($_.Exception.Message))"
    }

    return $out
}

function Export-SummaryTable {
    param(
        [object[]] $Rows,
        [string] $CsvPath
    )

    $Rows | Export-Csv -Path $CsvPath -NoTypeInformation -Encoding UTF8

    Write-Host ""
    Write-Host "========== Online Search Compare Summary ==========" -ForegroundColor Green
    $fmt = "{0,-10} {1,8} {2,6} {3,6} {4,6} {5,8} {6,8} {7,8}"
    Write-Host ($fmt -f "method", "score", "evals", "1stOK", "bestAt", "feas", "time_s", "exit")
    Write-Host ("-" * 78)

    foreach ($r in $Rows) {
        $score = if ($null -ne $r.best_score) { "{0:F4}" -f $r.best_score } else { "n/a" }
        $firstOk = if ($null -ne $r.first_l2_ok_eval) { "$($r.first_l2_ok_eval)" } else { "-" }
        $bestAt = if ($null -ne $r.best_score_eval) { "$($r.best_score_eval)" } else { "-" }
        $feas = if ($null -ne $r.feasible_found) { "$($r.feasible_found)" } else { "-" }
        $time = if ($null -ne $r.wall_time_s) { "{0:F1}" -f $r.wall_time_s } else { "-" }
        Write-Host ($fmt -f $r.method, $score, $r.real_evals, $firstOk, $bestAt, $feas, $time, $r.exit_code)
    }

    Write-Host ""
    Write-Host "CSV -> $CsvPath"
}

# ---------- main ----------
$repoRoot = Get-RepoRoot
Set-Location $repoRoot

if ($Methods.Count -eq 1 -and $Methods[0] -match ',') {
    $Methods = $Methods[0].Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}

if ([string]::IsNullOrWhiteSpace($ResultsDir)) {
    $ResultsDir = Join-Path $repoRoot "sealp\examples\layout\_output\online_compare"
}
$logDir = Join-Path $ResultsDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$runTag = "b${MaxEvals}_s${Seed}"
Write-Info "repo=$repoRoot"
Write-Info "methods=$($Methods -join ',') seed=$Seed max_evals=$MaxEvals top_k=$TopKProposals pool=$ScorerPool station=$StationMode"
Write-Info "results -> $ResultsDir"

$rows = New-Object System.Collections.Generic.List[object]

foreach ($method in $Methods) {
    $method = $method.Trim().ToLower()
    $cfg = Get-MethodConfig -Method $method -RepoRoot $repoRoot

    if ($cfg.Kind -eq "neural") {
        if (-not $cfg.Checkpoint -or -not (Test-Path $cfg.Checkpoint)) {
            $msg = "Skip $method : checkpoint missing -> $($cfg.Checkpoint)"
            if ($SkipMissingCheckpoint) {
                Write-Warn $msg
                continue
            }
            throw $msg
        }
    }

    $outputName = "tower_online_${method}_${runTag}"
    $cmd = Build-Command -Cfg $cfg -OutputName $outputName -Seed $Seed `
        -MaxEvals $MaxEvals -TopKProposals $TopKProposals -ScorerPool $ScorerPool `
        -StationMode $StationMode -CdprimType $CdprimType `
        -GlobalElite $GlobalElite -GlobalRefineSteps $GlobalRefineSteps -NoRefine:$NoRefine

    $logPath = Join-Path $logDir "${outputName}.log"
    $cmdLine = ($cmd -join " ")

    Write-Host ""
    Write-Info ">>> $method"
    Write-Host "    $cmdLine"
    if ($cfg.Checkpoint) {
        Write-Host "    checkpoint: $($cfg.Checkpoint)"
    }

    if ($DryRun) {
        $rows.Add([pscustomobject]@{
            method = $method
            output_name = $outputName
            checkpoint = $cfg.Checkpoint
            cmd = $cmdLine
            log = $logPath
        }) | Out-Null
        continue
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $cmd[0] $cmd[1..($cmd.Length - 1)] 2>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
    $sw.Stop()

    $text = Get-Content -Raw -Encoding UTF8 $logPath
    $parsed = Parse-LogMetrics -Text $text -Kind $cfg.Kind
    $debugPath = Join-Path $repoRoot "sealp\examples\layout\_output\${outputName}_debug.json"
    $dbg = Read-DebugMetrics -DebugPath $debugPath

    $bestScore = $dbg.best_score
    if ($null -eq $bestScore) {
        $bestScore = $parsed.best_score_stdout
    }

    $firstL2 = $parsed.first_l2_ok_eval
    if ($null -eq $firstL2) { $firstL2 = $dbg.first_l2_ok_eval }
    $bestAt = $parsed.best_score_eval
    if ($null -eq $bestAt) { $bestAt = $dbg.best_score_eval }

    $row = [pscustomobject]@{
        method = $method
        seed = $Seed
        max_evals = $MaxEvals
        top_k_proposals = if ($cfg.Kind -eq "neural") { $TopKProposals } else { $null }
        scorer_pool = if ($cfg.Kind -eq "neural") { $ScorerPool } else { $null }
        station_mode = if ($cfg.Kind -eq "neural") { $StationMode } else { $null }
        checkpoint = $cfg.Checkpoint
        output_name = $outputName
        exit_code = $exitCode
        best_score = $bestScore
        best_region = if ($dbg.assembly_region_id) { $dbg.assembly_region_id } else { $parsed.best_region }
        assembly_region_rc = $dbg.assembly_region_rc
        real_evals = $parsed.real_evals
        cache_hits = $parsed.cache_hits
        feasible_found = $parsed.feasible_found
        first_l2_ok_eval = $firstL2
        best_score_eval = $bestAt
        l3_passed = [bool]$parsed.l3_passed
        failed_flag = [bool]$parsed.failed
        wall_time_s = if ($parsed.wall_time_s) { $parsed.wall_time_s } else { [math]::Round($sw.Elapsed.TotalSeconds, 1) }
        log = $logPath
        layout = Join-Path $repoRoot "sealp\examples\layout\_output\${outputName}.layout"
        debug_json = $debugPath
    }

    $rows.Add($row) | Out-Null

    $scoreTxt = if ($null -ne $row.best_score) { "{0:F4}" -f $row.best_score } else { "n/a" }
    Write-Info "[done] $method exit=$exitCode score=$scoreTxt evals=$($row.real_evals) bestAt=$($row.best_score_eval) firstOK=$($row.first_l2_ok_eval)"
}

if ($DryRun) {
    Write-Host ""
    Write-Info "DryRun finished; no search jobs executed."
    $rows | Format-Table -AutoSize
    exit 0
}

$csvPath = Join-Path $ResultsDir "online_compare_${runTag}.csv"
Export-SummaryTable -Rows $rows -CsvPath $csvPath

$meta = @{
    created_at = (Get-Date).ToString("o")
    repo_root = $repoRoot
    seed = $Seed
    max_evals = $MaxEvals
    top_k_proposals = $TopKProposals
    scorer_pool = $ScorerPool
    station_mode = $StationMode
    methods = $Methods
    rows = $rows
} | ConvertTo-Json -Depth 6

$jsonPath = Join-Path $ResultsDir "online_compare_${runTag}.json"
Set-Content -Path $jsonPath -Value $meta -Encoding UTF8
Write-Info "JSON -> $jsonPath"

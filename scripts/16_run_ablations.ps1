# run_ablations.ps1 - E5 ablations (research plan, Step 6)
# 3 seeds; each variant: 3 source models (zero-shot) + fine-tune at 1 % and 5 % of IEEE 30.
# Restartable: finished runs are skipped. Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File scripts\16_run_ablations.ps1

Set-Location (Split-Path -Parent $PSScriptRoot)   # project folder = parent of scripts\ (works after renaming)

$variants = @(
    @{ tag = "_nope";    src = @("--extra=--no-pe");        tgt = @("--extra=--no-pe") },
    @{ tag = "_noattn";  src = @("--extra=--no-attn");      tgt = @("--extra=--no-attn") },
    @{ tag = "_nophys";  src = @("--extra=--lambda-pb 0");  tgt = @("--extra=--lambda-pb 0") },
    @{ tag = "_augnone"; src = @("--aug", "none");          tgt = @() },
    @{ tag = "_augL";    src = @("--aug", "L");             tgt = @() },
    @{ tag = "_augLP";   src = @("--aug", "LP");            tgt = @() }
)

$start = Get-Date
foreach ($v in $variants) {
    $src = $v.src
    $tgt = $v.tgt
    Write-Host "`n=== Ablation $($v.tag): source models (seeds 0-2) ===  $(Get-Date -Format 'HH:mm')"
    & .\.venv\Scripts\python.exe scripts\14_run_main_experiments.py --stage source --seeds 0 1 2 --tag $v.tag @src
    Write-Host "`n=== Ablation $($v.tag): fine-tune at 1 % and 5 % ===  $(Get-Date -Format 'HH:mm')"
    & .\.venv\Scripts\python.exe scripts\14_run_main_experiments.py --stage target --methods ft --fractions 0.01 0.05 --seeds 0 1 2 --tag $v.tag @tgt
}
Write-Host "`nAll ablations finished in $([math]::Round(((Get-Date) - $start).TotalHours, 1)) h"
& .\.venv\Scripts\python.exe scripts\25_summarize_results.py
& .\.venv\Scripts\python.exe scripts\31_make_figures.py

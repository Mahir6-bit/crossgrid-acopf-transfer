# run_baselines.ps1 - neural baselines (research plan, Step 6), 3 seeds.
#   GCN, GAT (homogeneous bus graphs, parameter-matched to the proposed model, ~1.3M):
#       source on IEEE 14 (+LPT augmentation) -> zero-shot, fine-tune and from scratch at 5 % and 100 %
#   MLP (fixed-size, cannot transfer): from scratch only, 5 % and 100 %
# Same data, loss, lambda, decoder and step budget as the proposed model.
# Run:  powershell -ExecutionPolicy Bypass -File scripts\15_run_baselines.ps1   (restartable)

Set-Location (Split-Path -Parent $PSScriptRoot)   # project folder = parent of scripts\ (works after renaming)
$py = ".\.venv\Scripts\python.exe"
$start = Get-Date

$gnns = @(
    @{ tag = "_gcn"; extra = "--extra=--arch gcn --hidden 448" },
    @{ tag = "_gat"; extra = "--extra=--arch gat --hidden 320" }
)
foreach ($b in $gnns) {
    Write-Host "`n=== Baseline $($b.tag): source models on IEEE 14 ===  $(Get-Date -Format 'HH:mm')"
    & $py scripts\14_run_main_experiments.py --stage source --seeds 0 1 2 --tag $b.tag $b.extra
    Write-Host "`n=== Baseline $($b.tag): fine-tune and from scratch at 5 % and 100 % ===  $(Get-Date -Format 'HH:mm')"
    & $py scripts\14_run_main_experiments.py --stage target --methods ft scratch --fractions 0.05 1.0 --seeds 0 1 2 --tag $b.tag $b.extra
}

Write-Host "`n=== Baseline _mlp: from scratch at 5 % and 100 % ===  $(Get-Date -Format 'HH:mm')"
& $py scripts\14_run_main_experiments.py --stage target --methods scratch --fractions 0.05 1.0 --seeds 0 1 2 --tag _mlp "--extra=--arch mlp --hidden 768"

Write-Host "`nAll baselines finished in $([math]::Round(((Get-Date) - $start).TotalHours, 1)) h"
& $py scripts\25_summarize_results.py
& $py scripts\27_stats_tests.py

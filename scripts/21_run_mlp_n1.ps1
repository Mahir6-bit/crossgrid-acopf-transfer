# run_mlp_n1.ps1 - (a) full MLP data-efficiency curve, (b) IEEE 30 N-1 topology test.
# Run:  powershell -ExecutionPolicy Bypass -File scripts\21_run_mlp_n1.ps1    (restartable)

Set-Location (Split-Path -Parent $PSScriptRoot)   # project folder = parent of scripts\ (works after renaming)
$py = ".\.venv\Scripts\python.exe"
$start = Get-Date

Write-Host "`n=== (a) MLP from scratch at 1 %, 10 %, 25 % (3 seeds) ===  $(Get-Date -Format 'HH:mm')"
& $py scripts\14_run_main_experiments.py --stage target --methods scratch --fractions 0.01 0.1 0.25 --seeds 0 1 2 --tag _mlp "--extra=--arch mlp --hidden 768"

Write-Host "`n=== (b1) IEEE 30 N-1 test set: one branch out per test hour ===  $(Get-Date -Format 'HH:mm')"
if (-not (Test-Path data\processed\graph_tensors_case30_n1test.pt)) {
    & $py scripts\12_make_n1_sets.py
}

Write-Host "`n=== (b2) Evaluate all IEEE 30 models on the N-1 test set ===  $(Get-Date -Format 'HH:mm')"
& $py scripts\22_evaluate_n1.py

Write-Host "`n=== Summary, statistics, figures ===  $(Get-Date -Format 'HH:mm')"
& $py scripts\25_summarize_results.py | Out-Null
& $py scripts\27_stats_tests.py | Select-String -Pattern "FT vs MLP" -Context 0,40
& $py scripts\31_make_figures.py
Write-Host "`nFinished in $([math]::Round(((Get-Date) - $start).TotalHours, 1)) h"

# run_n1_adapt.ps1 - N-1 adaptation experiment (3 seeds, 5 % and 25 % of IEEE 30 training hours).
# Every selected training hour contributes its intact case AND one N-1 case (one random branch out);
# validation likewise. All models are tested on the IEEE 30 N-1 test set and the intact test set.
#   Proposed GNN, fine-tuned from the IEEE 14 source (topology-aware, transfer)
#   Proposed GNN, from scratch
#   MLP, topology-blind (cannot see which branch is out)
#   MLP + one-hot of the outaged branch (grid-specific topology input)
# Run:  powershell -ExecutionPolicy Bypass -File scripts\20_run_n1_adaptation.ps1    (restartable)

Set-Location (Split-Path -Parent $PSScriptRoot)   # project folder = parent of scripts\ (works after renaming)
$py = ".\.venv\Scripts\python.exe"
$start = Get-Date

foreach ($s in @("train", "val")) {
    if (-not (Test-Path "data\processed\graph_tensors_case30_n1$s.pt")) {
        Write-Host "`n=== IEEE 30 N-1 $s set ===  $(Get-Date -Format 'HH:mm')"
        & $py scripts\12_make_n1_sets.py --split $s
    }
}

Write-Host "`n=== Proposed GNN: fine-tune and from scratch, with N-1 data ===  $(Get-Date -Format 'HH:mm')"
& $py scripts\14_run_main_experiments.py --stage target --methods ft scratch --fractions 0.05 0.25 --seeds 0 1 2 --tag _n1 --source-tag= "--extra=--n1"

Write-Host "`n=== MLP (topology-blind), with N-1 data ===  $(Get-Date -Format 'HH:mm')"
& $py scripts\14_run_main_experiments.py --stage target --methods scratch --fractions 0.05 0.25 --seeds 0 1 2 --tag _n1mlp "--extra=--n1 --arch mlp --hidden 768"

Write-Host "`n=== MLP + outage one-hot, with N-1 data ===  $(Get-Date -Format 'HH:mm')"
& $py scripts\14_run_main_experiments.py --stage target --methods scratch --fractions 0.05 0.25 --seeds 0 1 2 --tag _n1mlptopo "--extra=--n1 --arch mlp_topo --hidden 768"

Write-Host "`nFinished in $([math]::Round(((Get-Date) - $start).TotalHours, 1)) h"
& $py scripts\26_summarize_n1.py

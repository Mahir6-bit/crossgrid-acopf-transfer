# run_unseen.ps1 - unseen-outage experiment (IEEE 30, 25 % of training hours, 5 seeds).
# The outages of 8 branches are held out of all N-1 training/validation data; the models are
# then tested separately on N-1 test hours with seen and with unseen (held-out) outages.
# Run:  powershell -ExecutionPolicy Bypass -File scripts\23_run_unseen_outages.ps1    (restartable)

Set-Location (Split-Path -Parent $PSScriptRoot)   # project folder = parent of scripts\ (works after renaming)
$py = ".\.venv\Scripts\python.exe"
$start = Get-Date

Write-Host "`n=== GNN fine-tune and from scratch, 8 outages held out ===  $(Get-Date -Format 'HH:mm')"
& $py scripts\14_run_main_experiments.py --stage target --methods ft scratch --fractions 0.25 --seeds 0 1 2 3 4 --tag _n1ho --source-tag= "--extra=--n1 --n1-holdout 8"

Write-Host "`n=== MLP + outage one-hot, 8 outages held out ===  $(Get-Date -Format 'HH:mm')"
& $py scripts\14_run_main_experiments.py --stage target --methods scratch --fractions 0.25 --seeds 0 1 2 3 4 --tag _n1homlptopo "--extra=--n1 --n1-holdout 8 --arch mlp_topo --hidden 768"

Write-Host "`n=== Evaluate on seen / unseen outages ===  $(Get-Date -Format 'HH:mm')"
& $py scripts\24_evaluate_unseen.py
Write-Host "`nFinished in $([math]::Round(((Get-Date) - $start).TotalHours, 1)) h"

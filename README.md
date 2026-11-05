# Cross-grid transfer learning for AC optimal power flow

Code for the article

> F. M. Mahir Daiyan, Md Asif Khan, Rajin Al Fatemi and Abu Shufian, "Quantifying target-data savings of cross-grid transfer learning for graph neural network surrogates of AC optimal power flow" (submitted).

A heterogeneous graph neural network with a physics-informed loss is pre-trained on the IEEE 14-bus system and fine-tuned on the IEEE 30-bus system. It is compared with training from scratch, with GCN, GAT and MLP baselines and with a DC-OPF dispatch. The experiments E1–E8 cover accuracy, data efficiency, ablation, feasibility after AC power-flow correction, computation time and N-1 robustness.

## Repository structure

```
scripts/
  01_ ... 34_            numbered steps, run in this order (see the table below)
  networks.py            test cases and dataset metadata          (shared modules: imported by the
  physics.py             AC power-flow physics in PyTorch          numbered steps, not run directly)
  opf_dataset.py         graph dataset, batching, effective-resistance encoding
  model.py               heterogeneous GNN, GCN/GAT and MLP baselines
  train.py               training of one model (called by steps 13 and 14)
  evaluate.py            test metrics of a trained model
  augment_case14.py      source-grid augmentation (run through step 09)
  pf_postprocess.py      AC power-flow correction and DC-OPF (run through step 17)
  figure_style.py        figure style of step 33
data/                    input and processed data (not in the repository, see below)
results/                 trained models, metrics, tables and figures (created by the scripts)
```

## Requirements

* Windows 10/11 or Linux, Python 3.13.5; an NVIDIA GPU is strongly recommended for training.
* Package versions used for the article: PyTorch 2.14.1 (CUDA 13.0), PyTorch Geometric 2.8.1, pandapower 3.5.6, NumPy 2.4.6, pandas 2.3.3, SciPy 1.18.1, plus matplotlib, networkx, pyarrow and python-docx. `requirements.txt` lists the exact versions.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

## Input data

1. Hourly bus-load time series of 2007 for the IEEE 14- and 30-bus systems: M. H. Shahriar, "Time series data of load, generation, and power flow in IEEE bus systems," IEEE DataPort, 2025, doi: [10.21227/bpp4-0r42](https://doi.org/10.21227/bpp4-0r42), introduced in Shahriar et al., iDDAF, SecureComm 2021, doi: [10.1007/978-3-030-90022-9_17](https://doi.org/10.1007/978-3-030-90022-9_17). Save the two files as
   * `data/raw/IEEE_14_Bus_Time_Series_Hourly.csv`
   * `data/raw/IEEE_30_Bus_Time_Series_Hourly.csv`
2. Test cases: PGLib-OPF v23.07 `pglib_opf_case14_ieee` (downloaded automatically by `networks.py`) and MATPOWER `case30` (included in pandapower).

Only the bus loads of the time series are used; its generator values and line flows are not.

## Run order

Run every command from the project folder with the environment active. Seeds are fixed, and finished runs are skipped, so any step can be stopped and restarted. PowerShell steps are started with `powershell -ExecutionPolicy Bypass -File scripts\<name>.ps1`.

### A. Optional checks

| No. | Command | Purpose |
|---|---|---|
| 01 | `python scripts\01_check_environment.py` | PyTorch, CUDA and PyTorch Geometric work |
| 02 | `python scripts\02_check_networks.py` | test cases match the dataset and solve at nominal load |
| 03 | `python scripts\03_load_profile_diagnostics.py` | statistics of the hourly load profiles |
| 04 | `python scripts\04_verify_dataset_structure.py` | why only the loads of the dataset are reused |
| 05 | `python scripts\05_load_band_feasibility.py` | AC-OPF feasibility versus load level (choice of the 0.6–1.0 band) |
| 06 | `python scripts\06_pilot_acopf.py` | AC-OPF on a sample of hours before the full label run |

### B. Data

| No. | Command | Output | Article |
|---|---|---|---|
| 07 | `python scripts\07_build_load_scenarios.py` | hourly load scenarios of both grids | Eqs. (11)–(12), Fig. 2 |
| 08 | `python scripts\08_generate_acopf_labels.py` | AC-OPF labels (8,760 h per grid), network tables | Table 2, Fig. 3 |
| 09 | `python scripts\09_augment_source_grid.py` | augmentation pools L, LP, LPT (IEEE 14-bus) | Table 3 |
| 10 | `python scripts\10_make_splits.py` | chronological split, nested subsets, seeds 0–4 | Table 6 |
| 11 | `python scripts\11_build_graph_dataset.py` | heterogeneous graph tensors | Table 4 |
| 12 | `python scripts\12_make_n1_sets.py` | IEEE 30-bus N-1 test set (seed 2027) | E8 |
| 13 | `python scripts\13_tune_lambda_pb.py` | choice of the power-balance weight λ_pb | Physics-informed learning objective |

### C. Experiments E1–E8

| No. | Command | Experiment | Article |
|---|---|---|---|
| 14 | `python scripts\14_run_main_experiments.py --stage source`, then `--stage target` | E1, E2 source models; E3, E4 fine-tuning and training from scratch (5 seeds) | Table 7, Figs. 6–9 |
| 15 | `scripts\15_run_baselines.ps1` | E4 GCN, GAT and MLP baselines (3 seeds) | Table 8 |
| 16 | `scripts\16_run_ablations.ps1` | E5 ablation of components and augmentation | Table 10, Fig. 10 |
| 17 | `python scripts\17_pf_postprocess.py` | E6 AC power-flow correction, feasibility, DC-OPF | Table 11, Fig. 11 |
| 18 | `python scripts\18_pf_breakdown_sensitivity.py` | E6 violated limits, zero-tolerance and slack variants | E6 text |
| 19 | `python scripts\19_benchmark_runtime.py` | E7 computation time (close other programs first) | Table 12, Fig. 12 |
| 20 | `scripts\20_run_n1_adaptation.ps1` | E8 training with outage data (N-1 training/validation sets, seeds 2028/2029) | Table 13, Fig. 13 |
| 21 | `scripts\21_run_mlp_n1.ps1` | grid-specific MLP at all data fractions; N-1 evaluation (runs step 22) | Tables 9 and 13 |
| 22 | `python scripts\22_evaluate_n1.py` | N-1 test metrics of all models (also run by step 21) | Table 13 |
| 23 | `scripts\23_run_unseen_outages.ps1` | E8 with eight outages withheld (seed 7); runs step 24 | Table 13, Fig. 13 |
| 24 | `python scripts\24_evaluate_unseen.py` | metrics on seen and unseen outages (also run by step 23) | Table 13 |

### D. Summaries, statistics, tables and figures

| No. | Command | Output | Article |
|---|---|---|---|
| 25 | `python scripts\25_summarize_results.py` | mean ± SD of all runs (`results\*.csv`) | Tables 7–10 |
| 26 | `python scripts\26_summarize_n1.py` | N-1 summary | Table 13 |
| 27 | `python scripts\27_stats_tests.py` | paired tests with Holm correction | Statistical reliability |
| 28 | `python scripts\28_stats_ci.py` | 95 % confidence intervals, effect sizes | Table 14 |
| 29 | `python scripts\29_predict_test.py` | saved test-set predictions (seed 0) | Figs. 6, 7, 9 |
| 30 | `python scripts\30_make_tables.py` | all tables as Word file and CSV (`results\tables\`) | Tables 2, 3, 5, 7–14 |
| 31 | `python scripts\31_make_figures.py` | data-efficiency and ablation figures | Figs. 8, 10 |
| 32 | `python scripts\32_make_paper_figures.py` | remaining result figures (`results\figures\paper\`) | Figs. 1–3, 9, 11–13 |
| 33 | `python scripts\33_make_e1_e2_figures.py` | source-grid parity and zero-shot dispatch | Figs. 6, 7 |
| 34 | `python scripts\34_paper_values.py` | values quoted in the text (withheld branches, N-1 convergence, graph-building time) | E2, E7, E8 text |

`30_make_tables.py` uses its own table numbers: its Tables 3, 4 and 5 are Tables 5, 2 and 3 of the article, and its Supplementary Table S2 contains the tests summarized in Table 14. Tables 1, 4, 6 and 15 and the schematic Figs. 4 and 5 are written or drawn in the article and are not produced by code. The result figures of the article were restyled for the journal; the scripts produce the same quantities.

## Seeds

| Purpose | Seeds |
|---|---|
| Model training and nested fine-tuning subsets | 0–4 (baselines and ablation 0–2) |
| Source augmentation pools L, LP, LPT | 2026, 2027, 2028 |
| N-1 test, training and validation sets | 2027, 2028, 2029 |
| Outages withheld in the unseen-outage experiment | 7 |

## Archived version

This repository (release v1.0.0) is archived on Zenodo: [10.5281/zenodo.23989653](https://doi.org/10.5281/zenodo.23989653).

## License

MIT (see `LICENSE`). Please cite the article and the dataset if you use this code.

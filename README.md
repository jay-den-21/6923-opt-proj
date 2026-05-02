# CS-GY 6923 Optional Project: SVG Scaling Laws

This repository implements a compact, reproducible version of the optional SVG language-model scaling project.

## Setup

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks virtualenv activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

All project paths are relative to the repository root. On Windows, first `cd` into the project folder, then run the same `python scripts/...` commands. Do not use the macOS absolute path from the original workspace.

## 1. Preprocess Data

For a smoke test:

```bash
python scripts/preprocess.py --limit 2000
python scripts/train_tokenizer.py --vocab_size 2048 --max_tokens 1024
```

For the real run:

```bash
python scripts/preprocess.py --datasets starvector/svg-icons-simple starvector/svg-emoji-simple starvector/svg-fonts-simple
python scripts/train_tokenizer.py --vocab_size 2048 --max_tokens 1024
python scripts/plot_dataset_stats.py --data_dir data/processed --out_dir outputs/analysis
```

Outputs:

- `data/processed/preprocess_stats.json`
- `data/processed/tokenizer_stats.json`
- `data/processed/{train,val,test}.bin`
- `outputs/analysis/sequence_length_histogram.png`
- `outputs/analysis/dataset_examples.html`

For a full-credit attempt, verify that `data/processed/tokenizer_stats.json` reports at least 100M training tokens after filtering. If it is below 100M, add another supplementary dataset or raise the supplementary-data subsample.

## 2. Tiny LR Sweep

Run a short sanity check first:

```bash
python scripts/train.py --config configs/tiny.yaml --out_dir outputs/runs/tiny_smoke --max_steps 100
```

Then run the learning-rate sweep:

macOS/Linux:

```bash
for lr in 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2; do
  python scripts/train.py --config configs/tiny.yaml --learning_rate $lr --out_dir outputs/runs/tiny_lr_$lr
done
```

Windows PowerShell:

```powershell
foreach ($lr in "1e-5","3e-5","1e-4","3e-4","1e-3","3e-3","1e-2") {
  python scripts/train.py --config configs/tiny.yaml --learning_rate $lr --out_dir outputs/runs/tiny_lr_$lr
}
```

Pick the LR with the lowest `best_val_loss` in `outputs/runs/tiny_lr_*/summary.json`.

Plot the sweep:

```bash
python scripts/plot_runs.py --mode lr_sweep --runs outputs/runs/tiny_lr_* --out_dir outputs/analysis
```

## 3. Scaling Runs

Pass the selected LR into every run:

macOS/Linux:

```bash
BEST_LR=3e-4
python scripts/train.py --config configs/tiny.yaml --learning_rate $BEST_LR --out_dir outputs/runs/tiny
python scripts/train.py --config configs/small.yaml --learning_rate $BEST_LR --out_dir outputs/runs/small
python scripts/train.py --config configs/medium.yaml --learning_rate $BEST_LR --out_dir outputs/runs/medium
python scripts/train.py --config configs/large_lite.yaml --learning_rate $BEST_LR --out_dir outputs/runs/large_lite
python scripts/train.py --config configs/xl_lite.yaml --learning_rate $BEST_LR --out_dir outputs/runs/xl_lite
```

Windows PowerShell:

```powershell
$BEST_LR = "3e-4"
python scripts/train.py --config configs/tiny.yaml --learning_rate $BEST_LR --out_dir outputs/runs/tiny
python scripts/train.py --config configs/small.yaml --learning_rate $BEST_LR --out_dir outputs/runs/small
python scripts/train.py --config configs/medium.yaml --learning_rate $BEST_LR --out_dir outputs/runs/medium
python scripts/train.py --config configs/large_lite.yaml --learning_rate $BEST_LR --out_dir outputs/runs/large_lite
python scripts/train.py --config configs/xl_lite.yaml --learning_rate $BEST_LR --out_dir outputs/runs/xl_lite
```

If time is tight, add `--max_steps 1000` to every command and report that all models used a fixed token budget instead of a full epoch.

## 4. Fit Scaling Curve

```bash
python scripts/fit_scaling.py \
  --runs outputs/runs/tiny outputs/runs/small outputs/runs/medium outputs/runs/large_lite outputs/runs/xl_lite \
  --out_dir outputs/analysis
python scripts/plot_runs.py \
  --mode curves \
  --runs outputs/runs/tiny outputs/runs/small outputs/runs/medium outputs/runs/large_lite outputs/runs/xl_lite \
  --out_dir outputs/analysis
```

Outputs:

- `outputs/analysis/scaling_fit.json`
- `outputs/analysis/scaling_plot.png`
- `outputs/analysis/validation_curves.png`

## 4b. muP Scaling Study

Install dependencies, including `mup`, then repeat the LR sweep and scaling runs with muP:

```bash
python scripts/train.py --config configs/tiny.yaml --parameterization mup --learning_rate 1e-4 --out_dir outputs/runs/mup_tiny_lr_1e-4
```

Windows PowerShell has staged helpers:

```powershell
.\scripts\run_experiments.ps1 -Stage mup-lr-sweep -Device cuda
.\scripts\run_experiments.ps1 -Stage mup-scaling -Device cuda
.\scripts\run_experiments.ps1 -Stage analysis-mup
.\scripts\run_experiments.ps1 -Stage compare-scaling
```

The muP path uses `mup.MuReadout`, `mup.MuAdamW`, base/delta shape annotation, and transformer attention scaling by `1 / d_head`.

## 5. Generate and Evaluate Samples

Use the best checkpoint:

```bash
python scripts/sample.py --ckpt outputs/runs/xl_lite/ckpt.pt --out_dir outputs/runs/xl_lite/samples
python scripts/evaluate.py \
  --ckpt outputs/runs/xl_lite/ckpt.pt \
  --samples outputs/runs/xl_lite/samples/samples.jsonl \
  --out outputs/runs/xl_lite/evaluation.json
```

The evaluator reports test perplexity, XML validity rate, and render validity rate.
It also reports a structural validity rate for valid `<svg>` roots and basic attribute sanity checks. Use `scripts/make_sample_sheet.py` to create an HTML grid with prefixes, generated code, and rendered SVGs.

## Report

Use `report/report_template.md` as the report skeleton. Every number in the report should come from JSON or JSONL logs in `data/processed` or `outputs`.

## Notes on Scope

The full assignment asks for at least 100M training tokens, one-epoch comparisons, and a complete fixed-LR vs. muP comparison. If you run with `--max_steps`, describe the result as fixed-budget rather than one epoch. If the full muP or 100M-token runs are incomplete, report that limitation directly.

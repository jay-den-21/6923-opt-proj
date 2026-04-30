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
python scripts/preprocess.py
python scripts/train_tokenizer.py --vocab_size 2048 --max_tokens 1024
```

Outputs:

- `data/processed/preprocess_stats.json`
- `data/processed/tokenizer_stats.json`
- `data/processed/{train,val,test}.bin`

## 2. Tiny LR Sweep

Run a short sanity check first:

```bash
python scripts/train.py --config configs/tiny.yaml --out_dir outputs/runs/tiny_smoke --max_steps 100
```

Then run the learning-rate sweep:

macOS/Linux:

```bash
for lr in 1e-4 3e-4 1e-3 3e-3 1e-2; do
  python scripts/train.py --config configs/tiny.yaml --learning_rate $lr --out_dir outputs/runs/tiny_lr_$lr
done
```

Windows PowerShell:

```powershell
foreach ($lr in "1e-4","3e-4","1e-3","3e-3","1e-2") {
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

## Report

Use `report/report_template.md` as the report skeleton. Every number in the report should come from JSON or JSONL logs in `data/processed` or `outputs`.

## Notes on Scope

The full assignment asks for at least 100M training tokens and a complete fixed-LR vs. muP comparison. This implementation is designed for a conservative, honest reduced-compute submission. If the reduced runs are the only completed experiments, describe them as reduced-compute experiments and do not claim that they satisfy the full ideal experimental scale.

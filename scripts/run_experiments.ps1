param(
  [ValidateSet(
    "setup",
    "preprocess-smoke",
    "preprocess-full",
    "tiny-smoke",
    "lr-sweep",
    "scaling",
    "analysis",
    "sample-eval",
    "all-smoke"
  )]
  [string]$Stage = "all-smoke",

  [string]$BestLr = "",
  [int]$MaxSteps = 1000,
  [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$LrValues = @("1e-4", "3e-4", "1e-3", "3e-3", "1e-2")
$ScaleConfigs = @(
  @{ Name = "tiny"; Config = "configs/tiny.yaml" },
  @{ Name = "small"; Config = "configs/small.yaml" },
  @{ Name = "medium"; Config = "configs/medium.yaml" },
  @{ Name = "large_lite"; Config = "configs/large_lite.yaml" },
  @{ Name = "xl_lite"; Config = "configs/xl_lite.yaml" }
)

function Invoke-Step {
  param([string]$Name, [scriptblock]$Command)
  Write-Host ""
  Write-Host "== $Name ==" -ForegroundColor Cyan
  & $Command
}

function Ensure-Venv {
  if (-not (Test-Path $Python)) {
    Invoke-Step "Create Python 3.12 virtualenv" {
      py -3.12 -m venv .venv
    }
  }
}

function Run-Python {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
  Ensure-Venv
  & $Python @Args
}

function Install-Dependencies {
  Ensure-Venv
  Invoke-Step "Install dependencies" {
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -r requirements.txt
  }
}

function Preprocess-Smoke {
  Invoke-Step "Preprocess smoke dataset" {
    Run-Python scripts/preprocess.py --limit 2000
  }
  Invoke-Step "Train tokenizer and encode smoke splits" {
    Run-Python scripts/train_tokenizer.py --vocab_size 2048 --max_tokens 1024
  }
}

function Preprocess-Full {
  Invoke-Step "Preprocess full dataset" {
    Run-Python scripts/preprocess.py
  }
  Invoke-Step "Train tokenizer and encode full splits" {
    Run-Python scripts/train_tokenizer.py --vocab_size 2048 --max_tokens 1024
  }
}

function Train-TinySmoke {
  Invoke-Step "Train tiny smoke run" {
    Run-Python scripts/train.py `
      --config configs/tiny.yaml `
      --out_dir outputs/runs/tiny_smoke `
      --max_steps 100 `
      --device $Device
  }
}

function Train-LrSweep {
  foreach ($lr in $LrValues) {
    Invoke-Step "Tiny LR sweep lr=$lr" {
      Run-Python scripts/train.py `
        --config configs/tiny.yaml `
        --learning_rate $lr `
        --out_dir "outputs/runs/tiny_lr_$lr" `
        --max_steps $MaxSteps `
        --device $Device
    }
  }
  Invoke-Step "Plot LR sweep" {
    Run-Python scripts/plot_runs.py `
      --mode lr_sweep `
      --runs outputs/runs/tiny_lr_1e-4 outputs/runs/tiny_lr_3e-4 outputs/runs/tiny_lr_1e-3 outputs/runs/tiny_lr_3e-3 outputs/runs/tiny_lr_1e-2 `
      --out_dir outputs/analysis
  }
}

function Get-BestLr {
  if ($BestLr) {
    return $BestLr
  }
  $summaries = Get-ChildItem outputs/runs/tiny_lr_*/summary.json -ErrorAction SilentlyContinue
  if (-not $summaries) {
    throw "No LR sweep summaries found. Run -Stage lr-sweep first or pass -BestLr 3e-4."
  }
  $best = $summaries |
    ForEach-Object { Get-Content $_.FullName | ConvertFrom-Json } |
    Sort-Object best_val_loss |
    Select-Object -First 1
  return [string]$best.config.learning_rate
}

function Train-Scaling {
  $lr = Get-BestLr
  Write-Host "Using BestLr=$lr" -ForegroundColor Green
  foreach ($item in $ScaleConfigs) {
    Invoke-Step "Scaling run $($item.Name)" {
      Run-Python scripts/train.py `
        --config $item.Config `
        --learning_rate $lr `
        --out_dir "outputs/runs/$($item.Name)" `
        --max_steps $MaxSteps `
        --device $Device
    }
  }
}

function Run-Analysis {
  $runs = @(
    "outputs/runs/tiny",
    "outputs/runs/small",
    "outputs/runs/medium",
    "outputs/runs/large_lite",
    "outputs/runs/xl_lite"
  )
  Invoke-Step "Fit scaling curve" {
    $args = @("scripts/fit_scaling.py", "--runs") + $runs + @("--out_dir", "outputs/analysis")
    Run-Python @args
  }
  Invoke-Step "Plot validation curves" {
    $args = @("scripts/plot_runs.py", "--mode", "curves", "--runs") + $runs + @("--out_dir", "outputs/analysis")
    Run-Python @args
  }
}

function Run-SampleEval {
  Invoke-Step "Sample from best checkpoint" {
    Run-Python scripts/sample.py --ckpt outputs/runs/xl_lite/ckpt.pt --out_dir outputs/runs/xl_lite/samples --device $Device
  }
  Invoke-Step "Evaluate samples" {
    Run-Python scripts/evaluate.py `
      --ckpt outputs/runs/xl_lite/ckpt.pt `
      --samples outputs/runs/xl_lite/samples/samples.jsonl `
      --out outputs/runs/xl_lite/evaluation.json `
      --device $Device
  }
}

switch ($Stage) {
  "setup" { Install-Dependencies }
  "preprocess-smoke" { Preprocess-Smoke }
  "preprocess-full" { Preprocess-Full }
  "tiny-smoke" { Train-TinySmoke }
  "lr-sweep" { Train-LrSweep }
  "scaling" { Train-Scaling }
  "analysis" { Run-Analysis }
  "sample-eval" { Run-SampleEval }
  "all-smoke" {
    Install-Dependencies
    Preprocess-Smoke
    Train-TinySmoke
  }
}

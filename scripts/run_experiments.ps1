param(
  [ValidateSet(
    "setup",
    "preprocess-smoke",
    "preprocess-full",
    "tiny-smoke",
    "lr-sweep",
    "scaling",
    "analysis",
    "dataset-stats",
    "mup-lr-sweep",
    "mup-scaling",
    "analysis-mup",
    "compare-scaling",
    "sample-eval",
    "sample-sheet",
    "all-smoke"
  )]
  [string]$Stage = "all-smoke",

  [string]$BestLr = "",
  [int]$MaxSteps = 0,
  [string]$Device = "auto",
  [string[]]$Datasets = @("starvector/svg-icons-simple", "starvector/svg-emoji-simple", "starvector/svg-fonts-simple"),
  [int]$MaxRecords = 0
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$LrValues = @("1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2")
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
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed with exit code $LASTEXITCODE`: $Python $($Args -join ' ')"
  }
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
    $cmdArgs = @("scripts/preprocess.py", "--datasets") + $Datasets
    if ($MaxRecords -gt 0) {
      $cmdArgs = $cmdArgs + @("--max_records", [string]$MaxRecords)
    }
    Run-Python @cmdArgs
  }
  Invoke-Step "Train tokenizer and encode full splits" {
    Run-Python scripts/train_tokenizer.py --vocab_size 2048 --max_tokens 1024
  }
}

function Plot-DatasetStats {
  Invoke-Step "Plot dataset stats and examples" {
    Run-Python scripts/plot_dataset_stats.py --data_dir data/processed --out_dir outputs/analysis
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
      $cmdArgs = @(
        "scripts/train.py",
        "--config", "configs/tiny.yaml",
        "--learning_rate", $lr,
        "--out_dir", "outputs/runs/tiny_lr_$lr",
        "--device", $Device
      )
      if ($MaxSteps -gt 0) {
        $cmdArgs += @("--max_steps", [string]$MaxSteps)
      }
      Run-Python @cmdArgs
    }
  }
  Invoke-Step "Plot LR sweep" {
    $runs = $LrValues | ForEach-Object { "outputs/runs/tiny_lr_$_" }
    $cmdArgs = @("scripts/plot_runs.py", "--mode", "lr_sweep", "--runs") + $runs + @("--out_dir", "outputs/analysis")
    Run-Python @cmdArgs
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

function Train-MupLrSweep {
  foreach ($lr in $LrValues) {
    Invoke-Step "muP tiny LR sweep lr=$lr" {
      $cmdArgs = @(
        "scripts/train.py",
        "--config", "configs/tiny.yaml",
        "--parameterization", "mup",
        "--learning_rate", $lr,
        "--out_dir", "outputs/runs/mup_tiny_lr_$lr",
        "--device", $Device
      )
      if ($MaxSteps -gt 0) {
        $cmdArgs += @("--max_steps", [string]$MaxSteps)
      }
      Run-Python @cmdArgs
    }
  }
  Invoke-Step "Plot muP LR sweep" {
    $runs = $LrValues | ForEach-Object { "outputs/runs/mup_tiny_lr_$_" }
    $cmdArgs = @("scripts/plot_runs.py", "--mode", "lr_sweep", "--runs") + $runs + @("--out_dir", "outputs/analysis_mup")
    Run-Python @cmdArgs
  }
}

function Get-MupBestLr {
  if ($BestLr) {
    return $BestLr
  }
  $summaries = Get-ChildItem outputs/runs/mup_tiny_lr_*/summary.json -ErrorAction SilentlyContinue
  if (-not $summaries) {
    throw "No muP LR sweep summaries found. Run -Stage mup-lr-sweep first or pass -BestLr 3e-4."
  }
  $best = $summaries |
    ForEach-Object { Get-Content $_.FullName | ConvertFrom-Json } |
    Sort-Object best_val_loss |
    Select-Object -First 1
  return [string]$best.config.learning_rate
}

function Train-MupScaling {
  $lr = Get-MupBestLr
  Write-Host "Using muP BestLr=$lr" -ForegroundColor Green
  foreach ($item in $ScaleConfigs) {
    Invoke-Step "muP scaling run $($item.Name)" {
      $cmdArgs = @(
        "scripts/train.py",
        "--config", $item.Config,
        "--parameterization", "mup",
        "--learning_rate", $lr,
        "--out_dir", "outputs/runs/mup_$($item.Name)",
        "--device", $Device
      )
      if ($MaxSteps -gt 0) {
        $cmdArgs += @("--max_steps", [string]$MaxSteps)
      }
      Run-Python @cmdArgs
    }
  }
}

function Train-Scaling {
  $lr = Get-BestLr
  Write-Host "Using BestLr=$lr" -ForegroundColor Green
  foreach ($item in $ScaleConfigs) {
    Invoke-Step "Scaling run $($item.Name)" {
      $cmdArgs = @(
        "scripts/train.py",
        "--config", $item.Config,
        "--learning_rate", $lr,
        "--out_dir", "outputs/runs/$($item.Name)",
        "--device", $Device
      )
      if ($MaxSteps -gt 0) {
        $cmdArgs += @("--max_steps", [string]$MaxSteps)
      }
      Run-Python @cmdArgs
    }
  }
}

function Run-AnalysisMup {
  $runs = @(
    "outputs/runs/mup_tiny",
    "outputs/runs/mup_small",
    "outputs/runs/mup_medium",
    "outputs/runs/mup_large_lite",
    "outputs/runs/mup_xl_lite"
  )
  Invoke-Step "Fit muP scaling curve" {
    $cmdArgs = @("scripts/fit_scaling.py", "--runs") + $runs + @("--out_dir", "outputs/analysis_mup")
    Run-Python @cmdArgs
  }
  Invoke-Step "Plot muP validation curves" {
    $cmdArgs = @("scripts/plot_runs.py", "--mode", "curves", "--runs") + $runs + @("--out_dir", "outputs/analysis_mup")
    Run-Python @cmdArgs
  }
}

function Compare-Scaling {
  if (-not (Test-Path "outputs/analysis/scaling_fit.json")) {
    throw "Missing outputs/analysis/scaling_fit.json. Run -Stage analysis first."
  }
  if (-not (Test-Path "outputs/analysis_mup/scaling_fit.json")) {
    throw "Missing outputs/analysis_mup/scaling_fit.json. Run -Stage analysis-mup first."
  }
  Invoke-Step "Compare SP vs muP scaling" {
    Run-Python scripts/compare_scaling.py `
      --sp_fit outputs/analysis/scaling_fit.json `
      --mup_fit outputs/analysis_mup/scaling_fit.json `
      --out outputs/analysis/sp_vs_mup_scaling.png
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
    $cmdArgs = @("scripts/fit_scaling.py", "--runs") + $runs + @("--out_dir", "outputs/analysis")
    Run-Python @cmdArgs
  }
  Invoke-Step "Plot validation curves" {
    $cmdArgs = @("scripts/plot_runs.py", "--mode", "curves", "--runs") + $runs + @("--out_dir", "outputs/analysis")
    Run-Python @cmdArgs
  }
}

function Make-SampleSheet {
  Invoke-Step "Build generated sample sheet" {
    Run-Python scripts/make_sample_sheet.py `
      --samples outputs/runs/xl_lite/samples/samples_evaluated.jsonl `
      --out outputs/runs/xl_lite/sample_contact_sheet.html
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
  "dataset-stats" { Plot-DatasetStats }
  "tiny-smoke" { Train-TinySmoke }
  "lr-sweep" { Train-LrSweep }
  "scaling" { Train-Scaling }
  "analysis" { Run-Analysis }
  "mup-lr-sweep" { Train-MupLrSweep }
  "mup-scaling" { Train-MupScaling }
  "analysis-mup" { Run-AnalysisMup }
  "compare-scaling" { Compare-Scaling }
  "sample-eval" { Run-SampleEval }
  "sample-sheet" { Make-SampleSheet }
  "all-smoke" {
    Install-Dependencies
    Preprocess-Smoke
    Train-TinySmoke
  }
}

# Windows 实验运行说明

这个项目的实验分成几步：先下载并清洗 SVG 数据，再训练 tokenizer，然后用 tiny 模型扫学习率，最后用选出来的学习率训练不同规模模型并画 scaling curve。

## 进入项目

```powershell
cd C:\6923-opt-proj
```

## 1. 安装环境

本机有 Python 3.12 和 3.14。建议用 Python 3.12，因为 PyTorch 对 3.12 更稳。

```powershell
.\scripts\run_experiments.ps1 -Stage setup
```

如果 PowerShell 不允许运行脚本，先执行一次：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 2. 快速验证流程

先用 2000 条数据跑 smoke test，确认环境、数据下载、tokenizer、训练都能工作：

```powershell
.\scripts\run_experiments.ps1 -Stage all-smoke
```

这会做三件事：

- 创建 `.venv`
- 安装 `requirements.txt`
- 跑 `preprocess --limit 2000`、tokenizer、`tiny_smoke`

## 3. 正式实验流程

完整数据预处理：

```powershell
.\scripts\run_experiments.ps1 -Stage preprocess-full
```

现在 `preprocess-full` 默认会合并：

- `starvector/svg-icons-simple`
- `starvector/svg-emoji-simple`
- `starvector/svg-fonts-simple`

目标是让 `data/processed/tokenizer_stats.json` 里的 train tokens 达到至少 100M。数据会比较大、下载时间会明显增加。如果只是先试流程，可以加 `-MaxRecords 20000`。

生成数据统计图和 SVG 难度示例：

```powershell
.\scripts\run_experiments.ps1 -Stage dataset-stats
```

扫 tiny 模型学习率：

```powershell
.\scripts\run_experiments.ps1 -Stage lr-sweep -MaxSteps 1000
```

输出在：

- `outputs/runs/tiny_lr_*/summary.json`
- `outputs/analysis/lr_sweep.png`

训练五组不同规模参数：

```powershell
.\scripts\run_experiments.ps1 -Stage scaling -MaxSteps 1000
```

如果你想手动指定学习率，例如 `3e-4`：

```powershell
.\scripts\run_experiments.ps1 -Stage scaling -BestLr 3e-4 -MaxSteps 1000
```

拟合 scaling curve 并画图：

```powershell
.\scripts\run_experiments.ps1 -Stage analysis
```

## 4. µP 实验

满分版本需要做 standard parameterization 和 µP 对比。先跑 µP 的 tiny 学习率 sweep：

```powershell
.\scripts\run_experiments.ps1 -Stage mup-lr-sweep -MaxSteps 1000 -Device cuda
```

然后用 µP tiny 选出来的学习率训练五个规模：

```powershell
.\scripts\run_experiments.ps1 -Stage mup-scaling -MaxSteps 1000 -Device cuda
```

拟合 µP scaling curve：

```powershell
.\scripts\run_experiments.ps1 -Stage analysis-mup
```

画 standard vs. µP 对比图：

```powershell
.\scripts\run_experiments.ps1 -Stage compare-scaling
```

采样和评估最大模型：

```powershell
.\scripts\run_experiments.ps1 -Stage sample-eval
```

生成带 prefix / SVG code / render 结果的 HTML 表：

```powershell
.\scripts\run_experiments.ps1 -Stage sample-sheet
```

## 参数组

模型规模配置在 `configs/`：

| 运行名 | 配置文件 | 主要规模 |
| --- | --- | --- |
| tiny | `configs/tiny.yaml` | 128 hidden, 4 layers |
| small | `configs/small.yaml` | 192 hidden, 4 layers |
| medium | `configs/medium.yaml` | 256 hidden, 6 layers |
| large_lite | `configs/large_lite.yaml` | 384 hidden, 6 layers |
| xl_lite | `configs/xl_lite.yaml` | 512 hidden, 8 layers |

学习率 sweep 默认跑：

```text
1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2
```

## 结果位置

- 预处理统计：`data/processed/preprocess_stats.json`
- tokenizer 统计：`data/processed/tokenizer_stats.json`
- 每次训练结果：`outputs/runs/<run_name>/summary.json`
- 每次训练曲线日志：`outputs/runs/<run_name>/metrics.jsonl`
- LR sweep 图：`outputs/analysis/lr_sweep.png`
- scaling fit：`outputs/analysis/scaling_fit.json`
- scaling 图：`outputs/analysis/scaling_plot.png`
- validation curves：`outputs/analysis/validation_curves.png`
- sequence length histogram：`outputs/analysis/sequence_length_histogram.png`
- dataset examples：`outputs/analysis/dataset_examples.html`
- µP scaling fit：`outputs/analysis_mup/scaling_fit.json`
- standard vs. µP 图：`outputs/analysis/sp_vs_mup_scaling.png`

## 备注

`-MaxSteps 1000` 是较保守的固定训练预算，适合先跑通全部模型。满分要求更接近“至少 100M train tokens + 每个模型完整 1 epoch + standard vs. µP 对比”。如果有 GPU 和时间，可以去掉 `-MaxSteps`，让脚本按数据量跑完整一轮。

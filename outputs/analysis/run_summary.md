# Current Experiment Summary

## GPU Check

| item | value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| VRAM | 16,303 MiB |
| PyTorch | 2.11.0+cu128 |
| CUDA available | true |
| CUDA runtime | 12.8 |

## Full Data Summary

| split | files kept | tokens | dropped too long |
| --- | ---: | ---: | ---: |
| train | 48,139 | 28,266,652 | 30,686 |
| val | 455 | 260,639 | 349 |
| test | 505 | 298,937 | 300 |

Tokenizer vocab size: 176. Max tokens per SVG: 1024.

## LR Sweep

Tiny model, 1000 steps, CUDA.

| run | learning rate | best val loss | val perplexity | elapsed |
| --- | ---: | ---: | ---: | ---: |
| tiny_lr_1e-4 | 0.0001 | 1.5663 | 4.7890 | 26.7s |
| tiny_lr_3e-4 | 0.0003 | 1.3032 | 3.6811 | 26.7s |
| tiny_lr_1e-3 | 0.001 | 1.1145 | 3.0479 | 26.7s |
| tiny_lr_3e-3 | 0.003 | 1.0197 | 2.7724 | 26.7s |
| tiny_lr_1e-2 | 0.01 | 1.0160 | 2.7621 | 26.6s |

Selected LR by tiny sweep: 0.01.

## Scaling Runs

All runs used 1000 steps, CUDA, learning rate 0.01.

| run | parameters | best val loss | val perplexity | peak GPU MB | tokens/sec | elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 944,896 | 0.9643 | 2.6228 | 1,305.0 | 307,465 | 26.6s |
| small | 2,007,168 | 1.1604 | 3.1911 | 1,947.3 | 214,657 | 38.2s |
| medium | 5,040,128 | 1.4651 | 4.3278 | 1,909.5 | 114,021 | 71.8s |
| large_lite | 11,099,136 | 1.5205 | 4.5745 | 1,186.9 | 111,652 | 73.4s |
| xl_lite | 25,818,112 | 1.5783 | 4.8470 | 1,158.8 | 68,101 | 120.3s |

This run is useful as an LR stress test, but it is not the best scaling result because loss gets worse as model size grows.

## Recommended Scaling Runs

All runs used 1000 steps, CUDA, learning rate 0.001. These are the better report numbers because validation loss improves with model size.

| run | parameters | best val loss | val perplexity | peak GPU MB | tokens/sec | elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lr1e-3_tiny | 944,896 | 1.1133 | 3.0444 | 1,305.0 | 305,851 | 26.8s |
| lr1e-3_small | 2,007,168 | 1.0330 | 2.8095 | 1,947.3 | 214,070 | 38.3s |
| lr1e-3_medium | 5,040,128 | 0.9570 | 2.6040 | 1,909.5 | 113,903 | 71.9s |
| lr1e-3_large_lite | 11,099,136 | 0.8759 | 2.4011 | 1,186.9 | 112,286 | 73.0s |
| lr1e-3_xl_lite | 25,818,112 | 0.8278 | 2.2883 | 1,158.8 | 68,320 | 119.9s |

Recommended scaling fit:

| metric | value |
| --- | ---: |
| method | scipy_curve_fit |
| alpha | 0.1505 |
| rmse | 0.0056 |
| predicted 10x-parameter loss | 0.6932 |

## Generation And Evaluation

Best checkpoint: `outputs/runs/lr1e-3_xl_lite/ckpt.pt`.

| sample batch | samples | test loss | test perplexity | XML valid | XML valid rate | render valid | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default sampling | 17 | 0.7890 | 2.2013 | 3 | 17.6% | 3 | raw model outputs |
| conservative sampling | 17 | 0.8077 | 2.2427 | 2 | 11.8% | 0 | lower temperature did not improve XML validity |
| repaired sampling | 29 | 0.7858 | 2.1941 | 29 | 100.0% | 28 | XML-recovered outputs |

Valid generated SVG samples are collected in:

`outputs/runs/lr1e-3_xl_lite/sample_contact_sheet.html`

Repaired rendered PNG samples are collected in:

`outputs/runs/lr1e-3_xl_lite/repaired_contact_sheet.html`

The low raw XML-validity rate suggests that the model learns many SVG local patterns, but the reduced 1000-step budget is not enough for reliable long-range XML closure. The repaired batch should be described as post-processed output rather than raw generation.

## Outputs

| artifact | path |
| --- | --- |
| LR sweep plot | `outputs/analysis/lr_sweep.png` |
| LR=0.01 validation curves | `outputs/analysis/validation_curves.png` |
| LR=0.01 scaling fit JSON | `outputs/analysis/scaling_fit.json` |
| LR=0.01 scaling plot | `outputs/analysis/scaling_plot.png` |
| LR=0.001 validation curves | `outputs/analysis_lr1e-3/validation_curves.png` |
| LR=0.001 scaling fit JSON | `outputs/analysis_lr1e-3/scaling_fit.json` |
| LR=0.001 scaling plot | `outputs/analysis_lr1e-3/scaling_plot.png` |
| default samples | `outputs/runs/lr1e-3_xl_lite/samples/samples.jsonl` |
| conservative samples | `outputs/runs/lr1e-3_xl_lite/samples_conservative/samples.jsonl` |
| sample contact sheet | `outputs/runs/lr1e-3_xl_lite/sample_contact_sheet.html` |
| repaired samples | `outputs/runs/lr1e-3_xl_lite/samples_repaired/samples.jsonl` |
| repaired PNG contact sheet | `outputs/runs/lr1e-3_xl_lite/repaired_contact_sheet.html` |

## Important Note

The tiny LR sweep selected `0.01`, but that learning rate did not transfer well to larger models. The `0.001` fixed-LR scaling run is more stable and should be the primary result for the report. In the writeup, mention that the tiny-optimal LR was too aggressive for larger models and that a conservative fixed LR produced monotonic scaling.

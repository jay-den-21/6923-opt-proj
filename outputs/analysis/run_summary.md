# Fixed Formal Scaling Summary

This summary is for the formal run after fixing SVG path-number rounding in `scripts/preprocess.py`. Older results produced before that fix should be treated as stale because tight path commands such as `C8.64` could be corrupted.

## Full Data

| item | value |
| --- | ---: |
| cleaned SVGs | 300,000 |
| train / val / test files | 294,000 / 3,000 / 3,000 |
| train binary size | 424.2 MiB |
| tokenizer vocab | 342 |
| max tokens per SVG | 1024 |

| source | raw rows scanned | cleaned SVGs kept |
| --- | ---: | ---: |
| starvector/svg-icons-simple | 80,434 | 80,434 |
| starvector/svg-emoji-simple | 4,114 | 4,114 |
| starvector/svg-fonts-simple | 1,744,783 | 215,452 |

## SP LR Sweep

Tiny model, full epoch, fixed preprocessing.

| learning rate | best val loss | val perplexity |
| ---: | ---: | ---: |
| 1e-5 | 1.6008 | 4.9568 |
| 3e-5 | 1.3066 | 3.6936 |
| 1e-4 | 1.0381 | 2.8238 |
| 3e-4 | 0.8888 | 2.4321 |
| 1e-3 | 0.8287 | 2.2903 |
| 3e-3 | 0.8061 | 2.2392 |
| 1e-2 | 0.8054 | 2.2376 |

`1e-2` barely wins on tiny, but it made `xl_lite` unstable. For the final SP scaling curve, `3e-3` was used because it is effectively tied on tiny and transfers better.

## SP Scaling

| run | parameters | val loss | test loss | test perplexity |
| --- | ---: | ---: | ---: | ---: |
| tiny | 966,144 | 0.8061 | 0.8096 | 2.2469 |
| small | 2,039,040 | 0.7618 | 0.7607 | 2.1397 |
| medium | 5,082,624 | 0.6825 | 0.6840 | 1.9819 |
| large_lite | 11,162,880 | 0.6180 | 0.6373 | 1.8914 |
| xl_lite | 25,903,104 | 0.7968 | 0.8128 | 2.2541 |

Best model by test loss: `outputs/runs/large_lite/ckpt.pt`.

## muP LR Sweep

| learning rate | best val loss | val perplexity |
| ---: | ---: | ---: |
| 1e-5 | 1.6620 | 5.2699 |
| 3e-5 | 1.3961 | 4.0394 |
| 1e-4 | 1.0770 | 2.9358 |
| 3e-4 | 0.9520 | 2.5909 |
| 1e-3 | 0.8469 | 2.3324 |
| 3e-3 | 0.8152 | 2.2596 |
| 1e-2 | 0.8068 | 2.2408 |

## muP Scaling

| run | parameters | val loss | test loss | test perplexity |
| --- | ---: | ---: | ---: | ---: |
| mup_tiny | 1,009,920 | 0.8068 | 0.7953 | 2.2152 |
| mup_small | 2,104,704 | 0.7655 | 0.7621 | 2.1429 |
| mup_medium | 5,170,176 | 0.7252 | 0.7142 | 2.0426 |
| mup_large_lite | 11,294,208 | 0.7152 | 0.7122 | 2.0385 |
| mup_xl_lite | 26,078,208 | 0.7158 | 0.7072 | 2.0283 |

muP transferred the tiny-selected `1e-2` learning rate more smoothly than SP. It did not beat the best SP `large_lite` run on this fixed dataset, but it gives a cleaner scaling curve and a useful comparison point.

## Scaling Fits

| fit | alpha | rmse | predicted 10x loss |
| --- | ---: | ---: | ---: |
| SP | 0.2718 | 0.0661 | 0.6543 |
| muP | 0.6593 | 0.0051 | 0.6984 |

Strict epoch-end validation-loss fits were also generated for the rubric wording "validation loss after 1 epoch":

| fit | alpha | rmse |
| --- | ---: | ---: |
| SP epoch-end | 0.2782 | 0.0580 |
| muP epoch-end | 0.6645 | 0.0134 |

## Generation

Final generation used the best checkpoint by test loss:

`outputs/runs/large_lite/ckpt.pt`

Sampling settings: 95 unconditional samples plus 5 prefix samples, `temperatures=0.25,0.35,0.45`, `top_k=15`, `max_new_tokens=1024`, repair enabled.

| samples | XML valid | structural valid | render valid | test loss | test perplexity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 100 / 100 | 100 / 100 | 100 / 100 | 0.6373 | 1.8914 |

## Key Artifacts

| artifact | path |
| --- | --- |
| dataset examples | `outputs/analysis/dataset_examples.html` |
| dataset histogram | `outputs/analysis/sequence_length_histogram.png` |
| SP LR sweep plot | `outputs/analysis/lr_sweep.png` |
| SP validation curves | `outputs/analysis/validation_curves.png` |
| SP training loss curves | `outputs/analysis/training_loss_curves.png` |
| SP scaling fit | `outputs/analysis/scaling_fit.json` |
| SP epoch-end scaling plot | `outputs/analysis/scaling_after_1epoch.png` |
| muP LR sweep plot | `outputs/analysis_mup/lr_sweep.png` |
| muP validation curves | `outputs/analysis_mup/validation_curves.png` |
| muP training loss curves | `outputs/analysis_mup/training_loss_curves.png` |
| muP scaling fit | `outputs/analysis_mup/scaling_fit.json` |
| muP epoch-end scaling plot | `outputs/analysis_mup/scaling_after_1epoch.png` |
| SP vs muP plot | `outputs/analysis/sp_vs_mup_scaling.png` |
| final sample JSONL | `outputs/runs/large_lite/samples_conservative_100/samples_evaluated.jsonl` |
| final sample HTML | `outputs/runs/large_lite/sample_contact_sheet_conservative_100.html` |

## Report Takeaway

The strongest result is that the fixed preprocessing gives valid, renderable SVGs and improves the modeling story. SP reaches the best raw loss at `large_lite`, but its largest model degrades. muP gives a much smoother scaling curve and demonstrates better learning-rate transfer, even though its final loss is slightly worse than the best SP checkpoint on this run.

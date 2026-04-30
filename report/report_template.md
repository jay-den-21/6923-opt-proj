# Scaling Laws for SVG Language Models

## 1. Introduction

This project studies decoder-only Transformer language models trained on SVG source code. SVG is text, but it has stricter syntax and a direct visual interpretation, which makes it useful for combining quantitative language-model evaluation with rendered qualitative inspection.

Research questions:

- How does validation loss scale with parameter count for SVG code?
- Does a learning rate selected on the smallest model transfer to larger models?
- What syntactic and visual structure appears in generated SVG samples?

## 2. Data

Dataset: `starvector/svg-icons-simple`.

Preprocessing:

- Removed XML comments and non-essential `metadata`, `title`, and `desc` elements.
- Rounded floating-point numeric values to one decimal place.
- Filtered SVGs shorter than 50 characters and examples above the configured maximum length.
- Parsed cleaned SVGs with `lxml` and kept only valid `<svg>` roots.
- Split by file into train/validation/test before token concatenation.

Fill in from `data/processed/preprocess_stats.json` and `data/processed/tokenizer_stats.json`:

- Files before/after filtering:
- Train/val/test token counts:
- Tokenizer vocabulary size:
- Sequence length p50/p95/max:

Include rendered examples from the filtered dataset at low, medium, and high token lengths.

## 3. Methods

Tokenizer: Byte-level BPE with vocabulary size 2048. This keeps the vocabulary small enough for fast training while still allowing common SVG tags, attributes, numeric fragments, and path commands to become reusable tokens.

Model: decoder-only Transformer with causal self-attention, learned token and position embeddings, GELU MLP blocks, AdamW, cosine learning-rate decay, and warmup.

Model table:

| Name | d_model | Layers | Heads | d_ff | Params | Batch tokens | Best val loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny | 128 | 4 | 4 | 512 |  |  |  |
| small | 192 | 4 | 6 | 768 |  |  |  |
| medium | 256 | 6 | 8 | 1024 |  |  |  |
| large_lite | 384 | 6 | 8 | 1536 |  |  |  |
| xl_lite | 512 | 8 | 8 | 2048 |  |  |  |

Learning-rate sweep: run Tiny at `1e-4`, `3e-4`, `1e-3`, `3e-3`, and `1e-2`. Select the validation-loss minimum and reuse that LR for all model sizes.

Scaling fit: fit `L = aN^-alpha + c`, where `N` is parameter count and `L` is validation loss after the fixed training budget.

## 4. Results

Add these figures from `outputs/`:

- Tiny learning-rate sweep table or plot.
- Scaling curve from `outputs/analysis/scaling_plot.png`.
- Training and validation curves from each run's `metrics.jsonl`.
- Generated SVG grid from `outputs/runs/<best>/samples/png`.

Quantitative generation metrics:

| Metric | Value |
|---|---:|
| Test perplexity |  |
| XML validity rate |  |
| SVG render rate |  |

Scaling fit:

- Alpha:
- Offset `c`:
- RMSE:
- 10x extrapolated parameter count:
- Predicted validation loss:

## 5. Discussion

Discuss whether validation loss consistently decreased with model size. If the largest model underperforms, report it honestly and connect it to limited training budget, fixed LR transfer, or optimization instability.

Discuss generated samples:

- Whether outputs are valid XML.
- Whether the model learns common SVG conventions such as `viewBox`, paths, circles, rectangles, fills, and strokes.
- Whether lower temperatures improve validity and higher temperatures improve diversity.
- Whether prefix completions preserve context.

Limitations:

- This is a reduced-compute version of the full optional project.
- The token budget and model sizes are smaller than the ideal assignment.
- muP was treated as a stretch goal and should only be reported if actually run.

## 6. Conclusion

Summarize the strongest empirical finding and the main resource limitation. Do not claim a full-scale law if the results are based on reduced runs.

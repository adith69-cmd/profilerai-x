# S8b — Skill-aligned data fix, retrain, re-eval

## Diagnosis (from S8)
S6 training data sampled references and targets independently from the
broad domain pool. Profile skills never aligned with reference content,
so the model learned "output 10 questions" but not "condition on profile."

## Fix
Regenerated S6 data with skill-aligned sampling: for each profile, only
sample references/targets whose text shares a substring with at least one
profile skill. Fallback to full pool if matched pool < 13 rows.

## Training (v2)
- Same config as v1 (r=16, alpha=32, 3 epochs, lr=2e-4)
- Train loss: 0.309 (v1: 0.466)
- Val loss: 0.206 (v1: 0.387)
- Trainable params: 1.18%

## Results (N=50 held-out)

| Metric             | Base  | FT v1 | FT v2 |
|--------------------|-------|-------|-------|
| Format adherence   | 0.160 | 1.000 | 1.000 |
| Avg # questions    | 6.1   | 10.0  | 10.0  |
| Skill coverage     | 0.653 | 0.093 | 0.333 |

## Interpretation
- Skill coverage improved 3.6× over v1 (0.093 → 0.333).
- Format preserved at 1.00.
- Residual gap to base (0.333 vs 0.653) attributed to:
  1. Noisy substring-based skill matching in data construction.
  2. 1.5B model capacity — learns one strong pattern per fine-tune;
     format dominated, skill conditioning partially overwritten.

## Decision
Not iterating a third time. Documented as a residual limitation.
The honest story (diagnose → fix → partial recovery) is more valuable
than over-fitting the data pipeline to chase a metric.

## Artifacts
- Adapter v1: huggingface.co/Adith69/profilerai-x-qwen2.5-1.5b-qlora
- Adapter v2: huggingface.co/Adith69/profilerai-x-qwen2.5-1.5b-qlora-v2
- Eval: data/eval_results_v2.json
# SLM comparison

Two evaluations are recorded here. The full-split comparison of 4 November 2026 is the
result; the 64-question comparison of 18 October 2026 is kept below it as the pilot that
preceded it, because the contrast between them is informative.

## Full GSM8K test split, 4 November 2026

All 1,319 official GSM8K test questions. Each arm ran in its own Kaggle T4 session, which
is only legitimate because the harness records a SHA-256 for the evaluation sample, the
rendered prompt ids, the evaluation source files and the resolved library versions, and
refuses to pair runs whose records disagree. All four agree across the three sessions.
Greedy decoding, 512-token answer limit, eight-shot prompt, NF4 4-bit loading, seed 42.

| Model | Correct | Accuracy | Wilson 95% interval | Token-cap rate | Answer-marker rate |
|---|---:|---:|---:|---:|---:|
| BASE | 1072/1319 | 81.27% | 79.08%-83.29% | 1.06% | 97.80% |
| SFT | 1009/1319 | 76.50% | 74.13%-78.71% | 0.45% | 99.55% |
| GRPO | 1016/1319 | 77.03% | 74.68%-79.22% | 0.38% | 99.62% |

Adapter provenance, verified before each arm ran rather than after:
SFT `3cf54dcf0345452ff690bd71d90112979c67432f9f6275c1e77e6b722c5f6f73`,
GRPO `9cdc1fa17b4263e8de9899d36471f3fe005ed7f93cc4f3b86dabb17cc1554633`.
Evaluation sample `d3d4400878780edf494b36e2ed38fcba258243706d5533ceb2a3ae0f0240f08e`.

### Paired comparisons

| Comparison | Only first correct | Only second correct | exact McNemar p | Significant |
|---|---:|---:|---:|:---:|
| BASE vs SFT | 163 | 100 | 0.000123 | yes |
| BASE vs GRPO | 158 | 102 | 0.000620 | yes |
| SFT vs GRPO | 19 | 26 | 0.371 | no |

### What the diagnostics rule out

Truncation is not the explanation: the trained arms hit the 512-token cap *less* often than
the base model. Parse failure is not either, and runs the other way: the answer-marker rate
rises from 97.80% to 99.55%, and 304 of SFT's 310 errors are well-formed answers that are
simply wrong.

What did change is style. GSM8K reference solutions carry inline `<<a op b = c>>` calculator
annotations. The base model emits them in 0 of 1319 completions; after SFT they appear in
98.0%, and median completion length falls from 150 tokens to 116, toward the 63-token median
of the reference solutions. The warm-up learned the surface form of its targets. Note that
within each arm the *longer* completions are the less accurate ones, so this is not evidence
that shorter reasoning causes errors; it is evidence that the model moved toward its training
distribution and away from the verbose chain-of-thought that was serving it.

GRPO is barely a different policy from the SFT checkpoint it started from: 45 of 1319 items
disagree, against 263 for base versus SFT, with identical median completion length.

### Reproducing

```bash
python experiments/run_matched_eval_all.py --arms base --samples 1319
python experiments/run_matched_eval_all.py --arms sft  --samples 1319     --sft-adapter <dir> --sft-sha256 3cf54dcf0345452ff690bd71d90112979c67432f9f6275c1e77e6b722c5f6f73
python experiments/run_matched_eval_all.py --arms grpo --samples 1319     --grpo-adapter <dir> --grpo-sha256 9cdc1fa17b4263e8de9899d36471f3fe005ed7f93cc4f3b86dabb17cc1554633
python experiments/analyze_arms.py --directory artifacts/slm/evaluation-full --baseline base --candidate sft
```

## Pilot: 64 questions, 18 October 2026

64 fixed questions sampled with seed 42 from all 1,319 GSM8K test questions. All arms use the pinned Phi-3-mini base, the same base tokenizer and two-shot prompt, greedy decoding, a 512-token answer limit, and NF4 4-bit loading on the RTX 4060 laptop.

| Model | Correct | Accuracy | Wilson 95% interval | Token-cap rate | Answer-marker coverage |
|---|---:|---:|---:|---:|---:|
| BASE | 51/64 | 79.69% | 68.29%–87.73% | 0.00% | 100.00% |
| SFT | 49/64 | 76.56% | 64.87%–85.25% | 0.00% | 100.00% |
| GRPO | 50/64 | 78.12% | 66.57%–86.50% | 0.00% | 100.00% |

### Paired comparisons

- SFT minus BASE: -3.12 percentage points; 5 candidate-only correct, 7 baseline-only correct; exact McNemar p = 0.774414.
- GRPO minus BASE: -1.56 percentage points; 5 candidate-only correct, 6 baseline-only correct; exact McNemar p = 1.
- GRPO minus SFT: +1.56 percentage points; 2 candidate-only correct, 1 baseline-only correct; exact McNemar p = 1.

### Performance on this laptop

- BASE: mean 4.77 seconds per question, 169.2 generated tokens; peak PyTorch allocated memory 2.45 GiB.
- SFT: mean 6.09 seconds per question, 121.9 generated tokens; peak PyTorch allocated memory 2.83 GiB.
- GRPO: mean 8.89 seconds per question, 125.9 generated tokens; peak PyTorch allocated memory 2.83 GiB.

### Original GRPO training evidence

Recovered private Kaggle version 353908945 completed 93/93 optimizer steps. All recorded file hashes were independently checked. The adapter has 100,663,296 LoRA parameters.

Logged outcome reward fell from 1.400 to 0.042; mean completion length ended at 96, the training cap. Truncation is a plausible contributor to falling reward, not an established causal diagnosis. The local evaluation above uses a shared 512-token cap, so its scores must not be confused with rewards measured during 96-token stochastic training.


### Interpretation and limitations

This is a small local evaluation, not the full GSM8K benchmark. It establishes neither general superiority nor improvement on other tasks. The test sample is used for this comparison and should not subsequently be used to tune training settings.

Final-answer parsing prefers the #### marker and otherwise takes the last number. Answer-marker coverage is a formatting diagnostic, not a proof of valid reasoning. Token-cap hits exclude answers ending in EOS at the limit. Latency excludes loading; GPU memory is PyTorch allocated memory, not total device usage.

The trained demo uses GRPO, the higher-scoring trained arm on this sample (SFT wins a tie). This is a practical demo selection, not evidence of general superiority. The base model remains available as an explicit launcher option.

Neither trained arm outscored the base on this sample. The project demonstrates a working fine-tuning and evaluation pipeline; improved accuracy is not established.

### What the pilot was worth, in hindsight

The pilot ordered the three arms exactly as the full split later did, so its direction was
right. It could not support a claim: p = 0.774 on twelve discordant pairs and p = 1.000 on
eleven. A result at p = 1.000 is not weak evidence of an effect, it is no evidence at all.
It also mis-sized what it saw, putting GRPO 1.6 points behind the base model where the full
split puts it 4.2 behind and significant. The lesson is to read the discordance counts, not
the two accuracy percentages: eleven discordant items out of sixty-four makes the poverty of
the evidence legible in a way that "78.12% against 79.69%" does not.

## Local application verification

The real CUDA adapter served two arithmetic checks through the batch and streaming HTTP APIs. Health reported the adapter active; all three measured runs and the GUI assets were available. Responses and latency are saved in `docs/evidence/live_slm_app.json`. These are integration checks, not benchmark samples. Browser layout was not visually inspected because the computer-control tool could not initialize.

Raw completions, sample identities, hashes, exact settings and package versions are preserved under `artifacts/slm/evaluation`. Concise evidence is committed in `docs/evidence/slm_comparison.json`. See [the runbook](SLM_RUNBOOK.md) for reproduction.

# Original SLM comparison

Measured report: 2026-10-18.

64 fixed questions sampled with seed 42 from all 1,319 GSM8K test questions. All arms use the pinned Phi-3-mini base, the same base tokenizer and two-shot prompt, greedy decoding, a 512-token answer limit, and NF4 4-bit loading on the RTX 4060 laptop.

| Model | Correct | Accuracy | Wilson 95% interval | Token-cap rate | Answer-marker coverage |
|---|---:|---:|---:|---:|---:|
| BASE | 51/64 | 79.69% | 68.29%–87.73% | 0.00% | 100.00% |
| SFT | 49/64 | 76.56% | 64.87%–85.25% | 0.00% | 100.00% |
| GRPO | 50/64 | 78.12% | 66.57%–86.50% | 0.00% | 100.00% |

## Paired comparisons

- SFT minus BASE: -3.12 percentage points; 5 candidate-only correct, 7 baseline-only correct; exact McNemar p = 0.774414.
- GRPO minus BASE: -1.56 percentage points; 5 candidate-only correct, 6 baseline-only correct; exact McNemar p = 1.
- GRPO minus SFT: +1.56 percentage points; 2 candidate-only correct, 1 baseline-only correct; exact McNemar p = 1.

## Performance on this laptop

- BASE: mean 4.77 seconds per question, 169.2 generated tokens; peak PyTorch allocated memory 2.45 GiB.
- SFT: mean 6.09 seconds per question, 121.9 generated tokens; peak PyTorch allocated memory 2.83 GiB.
- GRPO: mean 8.89 seconds per question, 125.9 generated tokens; peak PyTorch allocated memory 2.83 GiB.

## Original GRPO training evidence

Recovered private Kaggle version 353908945 completed 93/93 optimizer steps. All recorded file hashes were independently checked. The adapter has 100,663,296 LoRA parameters.

Logged outcome reward fell from 1.400 to 0.042; mean completion length ended at 96, the training cap. Truncation is a plausible contributor to falling reward, not an established causal diagnosis. The local evaluation above uses a shared 512-token cap, so its scores must not be confused with rewards measured during 96-token stochastic training.


## Interpretation and limitations

This is a small local evaluation, not the full GSM8K benchmark. It establishes neither general superiority nor improvement on other tasks. The test sample is used for this comparison and should not subsequently be used to tune training settings.

Final-answer parsing prefers the #### marker and otherwise takes the last number. Answer-marker coverage is a formatting diagnostic, not a proof of valid reasoning. Token-cap hits exclude answers ending in EOS at the limit. Latency excludes loading; GPU memory is PyTorch allocated memory, not total device usage.

The trained demo uses GRPO, the higher-scoring trained arm on this sample (SFT wins a tie). This is a practical demo selection, not evidence of general superiority. The base model remains available as an explicit launcher option.

Neither trained arm outscored the base on this sample. The project demonstrates a working fine-tuning and evaluation pipeline; improved accuracy is not established.

## Local application verification

The real CUDA adapter served two arithmetic checks through the batch and streaming HTTP APIs. Health reported the adapter active; all three measured runs and the GUI assets were available. Responses and latency are saved in `docs/evidence/live_slm_app.json`. These are integration checks, not benchmark samples. Browser layout was not visually inspected because the computer-control tool could not initialize.

Raw completions, sample identities, hashes, exact settings and package versions are preserved under `artifacts/slm/evaluation`. Concise evidence is committed in `docs/evidence/slm_comparison.json`. See [the runbook](SLM_RUNBOOK.md) for reproduction.

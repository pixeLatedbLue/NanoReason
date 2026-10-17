# NanoReason: SLM fine-tuning working context

Updated: 18 October 2026. Active branch: `codex/slm-finetuning`, based on
original `main` commit `75f0fa7`.

## User correction and project scope

The user clarified that two projects had been confused. This project is the
original small-language-model fine-tuning application. Patent work, contextual
misuse prevention, hardware simulation and proof-graph research are not the
current objective. Do not resume those workstreams from earlier chat history.

Project statement: fine-tune Phi-3-mini (3.8B) for mathematical reasoning using
QLoRA supervised fine-tuning followed by GRPO, evaluate its actual performance
and provide a working local inference interface.

The later work remains preserved on `codex/proof-graph-verifier` and in ignored
local artifacts. It was not deleted, merged into this branch or published.
The 451-test and hardware-recovery results belong to that separate code line;
they do not describe this original branch or measure the model's accuracy.

## Original implementation

- Version 0.2.0 source, training/configuration pipeline and Solve/Verify/Runs UI.
- Base model: `microsoft/Phi-3-mini-4k-instruct`.
- QLoRA SFT plus GRPO with outcome, answer-format, local arithmetic-process and
  repetition/diversity reward components.
- Resumable Kaggle notebook: `notebooks/nanoreason_kaggle.ipynb`.
- Profiles: `configs/kaggle_t4.toml`, `configs/rtx4060_8gb.toml`,
  `configs/default.toml`, `configs/smoke.toml`.
- Real serving implementation exists; default auto mode can select a labelled
  scripted demo. An end-to-end real-model setup must be checked on this branch.
- The later Windows launchers are not in this branch. Do not claim that
  `Start-NanoReason.ps1` exists here or copy it without checking dependencies.

## Preserved training artifacts

On 18 October, all three local adapter weights were read and their SHA-256
values matched the preserved version-8 verification record on the other branch.
Each is 402,688,040 bytes, with Phi-3-mini as the configured base and LoRA rank 64.

| Local path under `artifacts/research/pilot` | Identity | SHA-256 |
|---|---|---|
| `sft-initial/adapter_model.safetensors` | Preserved SFT starting point | `3cf54dcf0345452ff690bd71d90112979c67432f9f6275c1e77e6b722c5f6f73` |
| `local-final/adapter_model.safetensors` | Later 16-step local-reward pilot arm | `492fd1c371ed8e1cfc6a4a66a773632edc2f15823f8d80eb7e7034fd43f88fff` |
| `graph-final/adapter_model.safetensors` | Later 16-step graph-reward pilot arm | `f855fc8f7eb6c8667d7f5ffcf86bc60f90ea43cc2df756ab6bfb36d71611b5ea` |

These files retain their historical directories to preserve provenance. The
`local-final` adapter is not the original longer GRPO checkpoint. Historical
records identify the original SFT/GRPO outputs in private Kaggle version 7,
script version `353908945`; retrieve/verify that original GRPO output before
using it as the original trained model. Remote availability has not been
rechecked during this scope reset.

The later pilot evaluated SFT and both 16-step arms on the same 64 test questions.
All scored 49/64 (76.5625%), with no accuracy improvement measured. There is no
base-model comparison in that pilot. These are existing limited observations,
not the original complete base/SFT/GRPO benchmark or evidence of general gains.

## Practical continuation

1. Verify/recover the original GRPO checkpoint and its training configuration;
   use the preserved SFT checkpoint as the known starting point meanwhile.
2. Restore a reproducible real-model launcher for this original code branch,
   with explicit adapter selection and a clearly labelled demo fallback.
3. Evaluate base, SFT and the selected original GRPO adapter on the same fixed
   test questions and decoding settings. Report accuracy, output truncation,
   response time and memory, including a result of no improvement if observed.
4. Fix concrete training or serving problems indicated by those checks, then
   package the usable app, model instructions and measured comparison.

Do not launch training simply because the old README says it is unfinished.
Use the saved artifacts first. Model merging/export can be chosen when packaging
requires it; the base-plus-adapter path is sufficient for inference.

## Checks and operating constraints

The original suite passed all **138 tests** on 18 October using the local Python
3.11 inference runtime (7.315 seconds of test execution). The first quiet run
produced no completion and was stopped; a diagnostic rerun with a 45-second
timeout completed successfully without a source change. Its initial delay was
not diagnosed. Do not substitute the other branch's 451-test count.

Keep changes and checkpoints local unless publication is explicitly requested.
Preserve all prior artifacts and the two untracked Kaggle screenshots. No new
cloud job, hardware purchase, patent filing or model training has started in
this scope-reset turn. Inspect owned processes before starting a GPU model.

The ordinary Windows sandbox launcher has previously failed; scoped escalated
workspace commands have worked. Existing runtimes are `.venv-gateway` for light
web checks and `C:/Users/naval/AppData/Local/NanoReason/inference/.venv` for the
model stack. Environment compatibility must be verified rather than assumed.

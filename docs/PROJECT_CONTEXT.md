# NanoReason: SLM fine-tuning working context

> **Superseded in part, 4 November 2026.** The 64-question figures below are the pilot.
> The full 1,319-question GSM8K test split has since been evaluated: base 81.27%,
> SFT 76.50%, GRPO 77.03%, with both trained arms significantly worse than base
> (p = 0.000123 and p = 0.000620) and GRPO indistinguishable from SFT (p = 0.371).
> See [SLM_RESULTS.md](SLM_RESULTS.md). The pilot detail is kept because it records
> how the checkpoints were recovered and verified.

Updated: 19 October 2026 (Asia/Calcutta). Active branch: `codex/slm-finetuning`, based on
original `main` commit `75f0fa7`.

## Dashboard redesign completed on 18 October

The user requested a full dashboard redesign using design-taste-frontend,
gpt-taste and minimalist-ui. Solve, Verify and Runs now share a warm neutral
workspace with self-hosted Geist, sidebar navigation, responsive layout,
dark/light themes and reduced-motion support. Real inference and measured
results remain connected; backend inference and evaluation code were not changed.

Solve places the prompt and live response side by side, supports copying and
Ctrl/Cmd+Enter, and shows actual model/adapter status. Verify provides a clearer
arithmetic inspector and describes its limitations. Runs visualizes the saved
base/SFT/GRPO accuracies and paired comparison without claiming improvement.
Source paths remain available in expandable details and JSON downloads.

Validation: 143 Python tests passed, 13 browser checks passed including real
GRPO inference (20 + 15 -> 35 in 3.5 seconds), all screens checked at five widths
from 320 to 1440px, and zero automated axe-core A/AA violations across three
screens in both themes. Details: docs/DASHBOARD_DESIGN.md.
Screenshots and browser/accessibility reports: artifacts/slm/dashboard/.
Frontend assets and license notices are vendored; no CDN is required.

The dashboard redesign is saved locally in commit `fd06d79`
("Redesign SLM dashboard with responsive workspace and measured results"),
after `ad5d6ef` (original SLM evaluation and live GRPO demo).
No training or public push occurred during the redesign.

The app was verified running at http://127.0.0.1:8000/ on 18 October. Runtime
availability has not been rechecked on 19 October; check health before claiming
it is still running or starting another GPU process. Refresh the existing
browser tab after frontend changes.

## Latest handoff: 19 October 2026

The user's latest request was to update this context file. The full dashboard
redesign and its validation are complete; no additional feature or training
work was requested in this turn. The original SLM fine-tuning scope remains
active. Preserve the existing measured results and do not resume the unrelated
patent, proof-graph, safety-controller or hardware workstreams.

At this handoff, Git shows only the two pre-existing untracked screenshots,
`kaggle-grpo-queued.png` and `kaggle-grpo-running.png`, outside this context
update. Leave them untouched. The validation numbers above are the completed
18 October checks, not new test runs performed on 19 October.

## User correction and project scope

The user clarified that two projects had been confused. This project is the
original small-language-model fine-tuning application. Patent work, contextual
misuse prevention, hardware simulation and proof-graph research are not the
current objective. Do not resume those workstreams from earlier chat history.

Project statement: fine-tune Phi-3-mini (3.8B) for mathematical reasoning using
QLoRA supervised fine-tuning followed by GRPO, evaluate its actual performance
and provide a working local inference interface.

## Active work on 18 October

The user asked Codex to continue the main work while handing small, independent
tasks to Claude. A copyable task was supplied for Claude to recover original
GRPO from private Kaggle version 7. Claude completed recovery to
`artifacts/slm/original-grpo` and `artifacts/slm/original-grpo-evidence`, and Codex
independently verified all 11 recorded hashes, 256 adapter tensors and 93/93 steps.
Codex owns evaluation, application changes, documentation and the local commit.

Browser and native computer-control tools failed before opening a session with
a runtime/volume error (Windows error 1005); no Kaggle credentials file was
available. This is an access-tool failure, not evidence of a Kaggle training
failure. The existing version 8 ZIP was inspected and contains no `grpo-final`.

The official full GSM8K test split (1,319 rows) is now preserved under
`artifacts/slm/data/gsm8k-test.jsonl`, SHA256
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
The new `nanoreason.matched_eval` command fixes 64 questions with seed 42 before
inference, uses the same pinned base tokenizer/prompts and 512-token cap for all
arms, saves every completion atomically, and rejects incompatible resumed runs.
The base run completed with **51/64 correct (79.6875%)**, no token-cap hits and
100% answer-marker coverage. SFT completed with **49/64 correct (76.5625%)**,
also zero token-cap hits and 100% marker coverage. Base versus SFT has 7 base-only
correct and 5 SFT-only correct items (exact McNemar p=0.7744140625).
Original GRPO completed with **50/64 correct (78.125%)**, zero token-cap hits and
100% marker coverage. All 192 evaluation responses are preserved. The exact
sample and prompt-token hashes agree across all three models. GRPO versus base
has 5 GRPO-only correct and 6 base-only correct items (p=1); GRPO versus SFT has
2 GRPO-only correct and 1 SFT-only correct item (p=1). No accuracy improvement
over base was established. Results are in `artifacts/slm/evaluation/runs`, with
full provenance and atomic per-item progress in `artifacts/slm/evaluation/progress`.

`Start-NanoReason.ps1` selects SFT, GRPO, Base or explicit Demo. The default
trained checkpoint is original GRPO, the higher-scoring trained arm on this
small sample; this is a practical demonstration choice, not a superiority claim.
Base remains available and had the highest observed score.
Pure inference reliability fixes were selectively reused from the preserved
branch: preload, serialized generation, context-length validation and propagation
of generation-thread errors. No graph or safety functionality was imported.
The GUI reports active adapter status and retains a failed label for partial
failed output. The current suite passes **143 tests**, and Ruff and JavaScript
syntax checks pass. See `docs/SLM_RUNBOOK.md` for exact commands and output paths.

The user authenticated in Claude's Chrome session, enabling version 7 recovery.
The recovered GRPO weight SHA256 is
`9cdc1fa17b4263e8de9899d36471f3fe005ed7f93cc4f3b86dabb17cc1554633`.
Verified evidence is in `docs/evidence/original_grpo_verification.json`. Training
reward declined substantially while mean completion length reached the 96-token
training cap. Truncation is a plausible contributor; the logs do not establish
causation, and completion length was not strictly monotonic (an initial dip is
present). The new matched evaluation uses 512 tokens for all three policies.

An attempted handoff write and local commit were blocked when automatic approval
review ran out of usage. No commit occurred, and `SLM_HANDOFF.md` was not created.
Earlier implementation edits and evaluation artifacts survived. Work resumed,
completed all three evaluations and passed real-model API checks. Both batch
and streaming calls returned the expected answers (35 and 55), with health
confirming `transformers`, `cuda:0` and `adapter_active=true`. The Runs API returned
all three completed evaluations, and HTML/JS/CSS returned HTTP 200. Evidence is
`docs/evidence/live_slm_app.json`. Visual browser inspection was not possible
because the computer-control runtime failed; opening the app in Codex was requested.

The app was left running at **http://127.0.0.1:8000/**. At verification, the
hidden PowerShell launcher PID was **17248** and Uvicorn's Python PID was **5196**.
Logs are under `artifacts/slm/live/`. Verify current command lines before stopping
processes; IDs can be reused. Do not start another GPU evaluation while the app
is loaded. For a foreground launch after stopping it, run `Start-NanoReason.ps1`.

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
script version `353908945`. Claude recovered original GRPO on 18 October and
Codex verified it, as recorded above. The recovered directory is
`artifacts/slm/original-grpo`; the later pilot adapters remain separate.

The later pilot evaluated SFT and both 16-step arms on the same 64 test questions.
All scored 49/64 (76.5625%), with no accuracy improvement measured. There is no
base-model comparison in that pilot. These are existing limited observations,
not the original complete base/SFT/GRPO benchmark or evidence of general gains.

## Completion and future scope

The authorized bounded original-project comparison and real-model application
are complete. `docs/SLM_RESULTS.md` contains measured results, paired tests,
confidence intervals, latency, memory, training evidence and limitations.
`docs/SLM_RUNBOOK.md` contains exact reproducible commands. Source and concise
evidence are being committed locally on `codex/slm-finetuning`; use `git log -1`
and `git status --short` to verify that final save.

The larger held-out evaluation has since been run: the full 1,319-question test
split, three arms, one arm per session, recorded in
[SLM_RESULTS.md](SLM_RESULTS.md). What remains optional future work is revised
training, specifically a GRPO retrain with a completion length above the
policy's output distribution, and that requires a new agreed scope. Do not
change the 64-question test set into training
data or tune to these results. Do not claim the model improved over its base.

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
web checks and `%LOCALAPPDATA%/NanoReason/inference/.venv` for the
model stack. Environment compatibility must be verified rather than assumed.

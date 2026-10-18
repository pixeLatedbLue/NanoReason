# Run the original NanoReason SLM project

This branch serves Phi-3-mini with the original QLoRA SFT or GRPO adapters.
The separately preserved proof-graph and safety projects are outside this scope.

## Local application

On the configured Windows laptop, from the repository root:

```powershell
.\Start-NanoReason.ps1
```

Open http://127.0.0.1:8000. The default is the recovered original **GRPO** model,
using `artifacts/slm/original-grpo`. It scored 50/64 versus SFT's 49/64 in the
matched sample; this pragmatic trained-demo selection is not a superiority claim.
The base scored 51/64 and remains available with `-Model Base`.
The launcher loads weights before serving, uses
the pinned cached base offline, and fails explicitly if an adapter is missing.

Other explicit choices:

```powershell
.\Start-NanoReason.ps1 -Model Base
.\Start-NanoReason.ps1 -Model SFT
.\Start-NanoReason.ps1 -Model GRPO
.\Start-NanoReason.ps1 -Model SFT -Adapter 'C:\path\to\sft-adapter' -Port 8001
.\Start-NanoReason.ps1 -Model Demo
```

GRPO expects `artifacts/slm/original-grpo`, recovered from private Kaggle version
7 and independently hash-verified. SFT uses `artifacts/research/pilot/sft-initial`;
that historical directory contains the preserved SFT, not a graph-trained policy.
Demo is scripted and visibly labelled.
Stop one server with Ctrl+C before starting another model or GPU evaluation.
The health endpoint reports model, adapter, device and whether the adapter is
actually active. Concurrent answer requests are serialized on the GPU; errors
are surfaced and partial failed streams are not labelled successful.

The existing Python runtime is
`C:\Users\naval\AppData\Local\NanoReason\inference\.venv\Scripts\python.exe`.
Pass `-Python` to use a different configured environment. The launcher does not
install packages or download model weights. A fresh machine needs the GPU and
serving dependencies from `pyproject.toml`, the pinned base snapshot from
`configs/local_inference.toml`, and the selected adapter. The model cache alone
is approximately 7.6 GB; inference is 4-bit NF4 with float16 computation.

## Reproduce the bounded comparison

The sample is fixed before generation: 64 questions selected with seed 42 from
all 1,319 official GSM8K test questions. This is a new original-project comparison,
not the later graph pilot's sample drawn from a 200-question pool. It does not
use training questions or select examples by verifier score or model correctness.

Run the following sequentially, with the app stopped:

```powershell
$taskPython = 'C:\Users\naval\AppData\Local\NanoReason\inference\.venv\Scripts\python.exe'
$env:PYTHONPATH = 'src'
& $taskPython experiments/download_gsm8k_test.py
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& $taskPython -u -m nanoreason.matched_eval --arm base
& $taskPython -u -m nanoreason.matched_eval --arm sft --adapter artifacts/research/pilot/sft-initial
# Original GRPO is now recovered and verified locally:
& $taskPython -u -m nanoreason.matched_eval --arm grpo --adapter artifacts/slm/original-grpo
```

Stop if any command fails. Do not start duplicate evaluation processes. Each arm
loads only one model. All arms use the same pinned base tokenizer, two-shot math
prompt, greedy decoding, 512-token answer limit, 4-bit quantization and sample.
Reference answers are used for scoring only, never included in the test prompt.

Outputs live in `artifacts/slm/evaluation`:

- `sample.json`: source URL, SHA256, population, seed and ordered questions.
- `progress/<arm>.json`: atomically saved per-question completions and provenance.
- `runs/<arm>.json`: completed runs, read directly by the app's Runs tab.
- `base-vs-<arm>.json`: exact paired McNemar comparison against base.

Restart the exact command after interruption. Resume rejects changed source data,
sample order, checkpoint files, protocol, code fingerprints or environment. If a
deliberate protocol change is needed, use a fresh `--output` directory and rerun
all arms. Partial runs are not published in the Runs directory.

Interpret final-answer accuracy alongside answer-marker coverage and token-cap
rates. Answer parsing follows the project's existing policy: prefer `####`,
otherwise use the last numeric token. A parsed answer alone does not establish
correct reasoning. Latencies include each question's generation and decoding,
exclude model loading, and depend on this laptop's load and power settings.
Wilson intervals and paired tests describe this small sample; they do not
establish broad model superiority or an improvement on other tasks.

## Validation

```powershell
$env:PYTHONPATH = 'src'
& $taskPython -m unittest discover -s tests -q
```

Keep adapters and raw results locally. Commit source, instructions and concise
verified evidence; do not publish private checkpoints or unrelated research work.

With the real GRPO app running, check its HTTP integration and regenerate the report:

```powershell
& $taskPython experiments/check_live_app.py --adapter artifacts/slm/original-grpo
& $taskPython experiments/report_slm.py --date 2026-10-18
```

Use the actual report date for subsequent runs. The integration check saves real
batch/stream responses, health information, run listings and asset availability.
It does not inspect browser layout or substitute for the 64-question comparison.

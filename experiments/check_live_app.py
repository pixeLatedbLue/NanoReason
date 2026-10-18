"""Exercise real inference through the local app; not a model-quality benchmark."""

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--adapter", required=True, help="Expected adapter directory")
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    with urllib.request.urlopen(base + "/api/health", timeout=15) as response:
        health = json.load(response)
    engine = health["engine"]
    assert engine["kind"] == "transformers" and not engine["is_demo"], engine
    assert engine["device"].startswith("cuda") and engine["adapter_active"], engine
    assert Path(engine["adapter"]).resolve() == Path(args.adapter).resolve(), engine

    question = ("A library had 29 books. It received 18 more books and then lent out 12. "
                "How many books remain?")
    request = urllib.request.Request(
        base + "/api/solve", headers={"Content-Type": "application/json"},
        data=json.dumps({"question": question, "gold": "35", "max_new_tokens": 256}).encode(),
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        solved = json.load(response)

    question = "A shop has 9 boxes with 7 pencils each. It gives away 8 pencils. How many pencils remain?"
    query = urllib.parse.urlencode({"question": question, "gold": "55", "max_new_tokens": 256})
    with urllib.request.urlopen(base + "/api/solve/stream?" + query, timeout=180) as response:
        frames = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                  if line.startswith("data: ")]
    with urllib.request.urlopen(base + "/api/runs", timeout=15) as response:
        runs = json.load(response)
    assets = {}
    for route in ("/", "/app.js", "/styles.css"):
        with urllib.request.urlopen(base + route, timeout=15) as response:
            assets[route] = {"status": response.status, "bytes": len(response.read())}
    evidence = {"health": health, "batch": solved, "stream_question": question,
                "stream_frames": frames, "runs": runs, "assets": assets,
                "visual_browser_inspection": False,
                "note": "Two arithmetic integration checks; not a held-out benchmark."}
    path = Path("docs/evidence/live_slm_app.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    assert solved["analysis"]["correct"] is True, solved
    assert frames and frames[-1]["type"] == "done" and not frames[-1].get("failed"), frames
    assert not any(frame["type"] == "error" for frame in frames), frames
    assert frames[-1]["analysis"]["correct"] is True, frames[-1]
    assert {row["name"] for row in runs["runs"]} == {"base", "sft", "grpo"}, runs
    print(json.dumps({"engine": engine, "batch_answer": solved["analysis"]["answer"],
                      "batch_latency_ms": solved["latency_ms"],
                      "stream_answer": frames[-1]["analysis"]["answer"],
                      "stream_latency_ms": frames[-1]["latency_ms"],
                      "run_count": len(runs["runs"]), "evidence": str(path)}))


if __name__ == "__main__":
    main()

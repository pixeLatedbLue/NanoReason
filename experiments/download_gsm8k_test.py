import hashlib
import json
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
EXPECTED_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
TARGET = Path(__file__).resolve().parents[1] / "artifacts/slm/data/gsm8k-test.jsonl"


def main() -> None:
    if TARGET.exists():
        data = TARGET.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=45) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Test data differs from the recorded source; refusing to overwrite or evaluate it")
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if len(rows) != 1319 or not all("question" in row and "answer" in row for row in rows):
        raise ValueError("Unexpected GSM8K test data schema or row count")
    if not TARGET.exists():
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        temporary = TARGET.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(TARGET)
    print(f"Verified {len(rows)} test items: {TARGET}\nSHA256: {EXPECTED_SHA256}")

if __name__ == "__main__":
    main()

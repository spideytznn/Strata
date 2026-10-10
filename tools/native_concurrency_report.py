"""Summarize two saved HTTP screens without treating them as peak benchmarks."""
import argparse
import json
from pathlib import Path
import re
from statistics import median


def summarize(folder):
    result = json.loads((folder / "results.json").read_text(encoding="utf8"))
    assert result.get("passed"), f"incomplete screen: {folder}"
    log = (folder / "engine.log").read_text(encoding="utf8")
    assert "post-residency expert source bytes=0" in log, folder
    assert "post-residency MTP source bytes=0" in log, folder
    assert not re.search(r"expert_file_bytes=[1-9]", log), folder
    groups = {}
    for clients in sorted({r["clients"] for r in result["rounds"]}):
        rows = [r for r in result["rounds"] if r["clients"] == clients]
        groups[clients] = {
            "rounds": len(rows),
            "aggregate_tok_s": median(r["aggregate_tok_s"] for r in rows),
            "last_first_content_s": median(max(r["ttft_s"]) for r in rows),
            "wall_s": median(r["wall_s"] for r in rows),
        }
    copies = [float(s) for s in re.findall(r"slot \d+ takes \d+ tokens \(copied in ([\d.]+) ms\)", log)]
    restores = [float(s) for s in re.findall(r"slot \d+ gave back .*? in ([\d.]+) ms", log)]
    return {"directory": str(folder), "exe_sha256": result.get("exe_sha256"),
            "startup_ms": result["info"].get("startup_ms"),
            "fixture_body_tokens": result.get("fixture_body_tokens"),
            "clients": groups,
            "slot_copy_median_ms": median(copies) if copies else None,
            "slot_restore_median_ms": median(restores) if restores else None,
            "expert_source_bytes": 0, "mtp_source_bytes": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    baseline, candidate = summarize(a.baseline), summarize(a.candidate)
    old = json.loads((a.baseline / "results.json").read_text(encoding="utf8"))
    new = json.loads((a.candidate / "results.json").read_text(encoding="utf8"))
    # Names identify the same prompt; exclude the deliberately truncated response.
    refs = {r["name"]: r["text"] for r in old["requests"] if not r["disconnected"]}
    checked = 0
    for row in new["requests"]:
        if row["name"] in refs and not row["disconnected"]:
            assert refs[row["name"]] == row["text"], f"cross-build text differs: {row['name']}"
            checked += 1
    changes = {}
    for clients, before in baseline["clients"].items():
        after = candidate["clients"][clients]
        changes[clients] = {
            "throughput_percent": 100 * (after["aggregate_tok_s"] / before["aggregate_tok_s"] - 1),
            "last_first_content_percent": 100 * (after["last_first_content_s"] / before["last_first_content_s"] - 1),
        }
    report = {"baseline": baseline, "candidate": candidate, "changes": changes,
              "matching_responses": checked,
              "limits": "Sequential warm-cache screens; admissions and HTTP included. GPU clocks not controlled. Repeated-text long prompts do not test retrieval quality."}
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

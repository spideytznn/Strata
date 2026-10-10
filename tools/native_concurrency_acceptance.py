"""Real native GEN/BGEN parity, cancellation, slot reuse and elastic KV checks.

Loads one private engine, never changes the desktop config, always terminates its own children.
This is a finite regression suite, not a general quality benchmark or long-context retrieval test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from serve.server import child_env, Vision, vision_env
from serve.frontend import ChatTemplate
from safetensors_tokenizer import SafetensorsTokenizer


class Protocol:
    def __init__(self, cfg, folder, vision=False):
        self.log_path = folder / "engine.log"
        self.log = self.log_path.open("w", encoding="utf8")
        self.q = queue.Queue()
        self.vision = None
        self.vision_log = (folder / "vision.log").open("w", encoding="utf8")
        self.proc = None
        try:
            env = child_env(cfg)
            if vision and cfg.get("vision"):
                self.vision = Vision(cfg["vision"], log=self.vision_log, env=vision_env(cfg, env))
            self.proc = subprocess.Popen([cfg["exe"], "--serve", *cfg["args"]], cwd=cfg["cwd"], env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                text=True, encoding="utf8", bufsize=1)
            def pump():
                for line in self.proc.stdout:
                    self.q.put(line.rstrip())
                self.q.put(None)
            threading.Thread(target=pump, daemon=True).start()
            self.info = []
            while True:
                line = self.read()
                if line.startswith("INFO "):
                    self.info.append(line)
                if line.startswith("READY "):
                    break
        except BaseException:
            self.close()
            raise

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read(self):
        line = self.q.get(timeout=300)
        if line is None:
            raise RuntimeError(f"engine exited ({self.proc.poll()}): {self.log_path}")
        if line.startswith(("ERR", "FATAL")):
            raise RuntimeError(line)
        return line

    def solo(self, ids, limit, keys):
        self.send(f"GEN {limit} {keys} {','.join(map(str, ids))}")
        got = []
        start = time.perf_counter()
        while True:
            line = self.read()
            if line.startswith("T "):
                got.append(int(line.split()[1]))
            elif line.startswith("DONE "):
                fields = line.split()
                assert int(fields[1]) == len(got) <= limit
                assert int(fields[12]) == 0 and float(fields[13]) == 0, line
                return {"ids": got, "done": line, "wall_s": time.perf_counter() - start}

    def batch(self, jobs, cancel_after=None):
        start = time.perf_counter()
        for slot, (ids, limit, keys) in enumerate(jobs):
            self.send(f"BGEN {slot} {limit} {keys} {','.join(map(str, ids))}")
        got = [{"ids": [], "times": []} for _ in jobs]
        admitted, done, current, cancelled = set(), set(), 0, False
        while len(done) < len(jobs):
            line = self.read()
            fields = line.split()
            if line.startswith("T "):
                got[current]["ids"].append(int(fields[1]))
                got[current]["times"].append(time.perf_counter() - start)
            elif line.startswith("DONE "):
                got[current]["admission"] = line
                assert int(fields[12]) == 0 and float(fields[13]) == 0, line
            elif line.startswith("BADM "):
                slot = int(fields[1])
                admitted.add(slot)
                got[slot]["continued"] = fields[2] == "1"
                if fields[2] == "0":
                    done.add(slot)
                current += 1
            elif line.startswith("BT "):
                slot = int(fields[1])
                got[slot]["ids"].append(int(fields[2]))
                got[slot]["times"].append(time.perf_counter() - start)
                if slot == 0 and cancel_after and len(got[slot]["ids"]) >= cancel_after and not cancelled:
                    self.send("BSTOP 0")
                    cancelled = True
            elif line.startswith("BDONE "):
                slot = int(fields[1])
                got[slot]["done"] = line
                assert int(fields[2]) == len(got[slot]["ids"]) <= jobs[slot][1], line
                done.add(slot)
        assert len(admitted) == len(jobs)
        return {"jobs": got, "wall_s": time.perf_counter() - start}

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.send("QUIT")
                self.proc.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait(timeout=15)
        if self.vision:
            self.vision.shutdown()
        self.log.close()
        self.vision_log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--adaptive", action="store_true", help="retain adaptive settings and require real two-row batch promotions")
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--growth-tokens", type=int, default=0, help="also test long-slot KV growth beside a short admission")
    ap.add_argument("--capacity-tokens", type=int, default=0, help="exact token count for a near-capacity two-long-slot check")
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_text(encoding="utf8"))
    cfg["exe"] = str(a.exe.resolve())
    args = cfg["args"]
    def setting(flag, value):
        if flag in args:
            args[args.index(flag) + 1] = str(value)
        else:
            args.extend([flag, str(value)])
    setting("--batch", 2)
    setting("--max-context", a.context)
    setting("--vram-reserve-mib", 1536)
    if not a.adaptive:
        setting("--adapt-every", 1000000)
        setting("--adapt-swaps", 0)
    cfg["env"]["STRATA_NATIVE_PREFILL_FIRST"] = "1"
    tok = SafetensorsTokenizer.from_directory(cfg["tokenizer"])
    template = ChatTemplate(Path(cfg["chat_template"]))
    def prompt(text):
        return tok.encode(template.render([{"role": "user", "content": text}], enable_thinking=False), parse_special=True)
    prompts = [prompt("Explain TCP and QUIC in detail, including congestion control and multiplexing."),
               prompt("写一个 Python 二分查找函数，解释边界条件和时间复杂度。")]
    result = {"config": cfg, "exe_sha256": hashlib.sha256(a.exe.read_bytes()).hexdigest(), "cases": []}
    def save():
        (a.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    save()
    eng = Protocol(cfg, a.output, a.vision)
    try:
        result["info"] = eng.info
        assert any("batch_slots=2" in line for line in eng.info), eng.info
        pairs = [
            ("greedy", ["temperature=0", "temperature=0"]),
            ("seeded", ["temperature=0.7 top_p=0.95 top_k=20 seed=6101", "temperature=1 top_p=0.8 top_k=12 seed=2718"]),
            ("penalties", ["temperature=0 penalty_last_n=128 penalty_repeat=1.12 penalty_present=0.3",
                           "temperature=0.7 top_p=0.95 top_k=20 seed=6101 penalty_last_n=256 penalty_freq=0.2"]),
        ]
        for name, keys in pairs:
            refs = [eng.solo(ids, a.limit, key) for ids, key in zip(prompts, keys)]
            batch = eng.batch([(ids, a.limit, key) for ids, key in zip(prompts, keys)])
            case = {"name": name, "solo": refs, "batch": batch}
            result["cases"].append(case)
            save()
            for i in range(2):
                assert refs[i]["ids"] == batch["jobs"][i]["ids"], f"{name}: slot {i} differs"
            times0, times1 = [job["times"] for job in batch["jobs"]]
            assert times0[1] < times1[-1] and times1[1] < times0[-1], "requests did not overlap"
            print(f"{name}: two slots identical, overlapping, {sum(len(x['ids']) for x in batch['jobs'])} tokens / {batch['wall_s']:.3f} s", flush=True)
        # A stopped slot must not consume or corrupt B, and its solo continuation must reproduce A.
        keys = "temperature=0"
        refs = [eng.solo(ids, a.limit, keys) for ids in prompts]
        batch = eng.batch([(ids, a.limit, keys) for ids in prompts], cancel_after=12)
        prefix = batch["jobs"][0]["ids"]
        assert prefix == refs[0]["ids"][:len(prefix)]
        assert batch["jobs"][1]["ids"] == refs[1]["ids"]
        resumed = eng.solo(prompts[0] + prefix, a.limit - len(prefix), keys)
        assert prefix + resumed["ids"] == refs[0]["ids"], "cancel/resume differs"
        result["cases"].append({"name": "cancel_resume", "batch": batch, "resumed": resumed})
        # Extend two finished conversations; compare live-slot reuse against serial reference.
        suffix = tok.encode("\nContinue with one concrete example.")
        followups = [ids + ref["ids"] + suffix for ids, ref in zip(prompts, refs)]
        refs2 = [eng.solo(ids, 32, keys) for ids in followups]
        batch2 = eng.batch([(ids, 32, keys) for ids in followups])
        assert all(x["ids"] == y["ids"] for x, y in zip(refs2, batch2["jobs"]))
        result["cases"].append({"name": "followup", "solo": refs2, "batch": batch2})
        # Admission itself can finish: no waiting forever for a BDONE that must not exist.
        short = eng.batch([(prompts[0], 1, keys), (prompts[1], 2, keys)])
        assert [len(x["ids"]) for x in short["jobs"]] == [1, 2]
        result["cases"].append({"name": "caps", "batch": short})
        eos_prompts = [prompt("Reply with exactly the single word Blue and stop."),
                       prompt("Reply with exactly the single word Green and stop.")]
        eos_refs = [eng.solo(ids, 48, keys) for ids in eos_prompts]
        eos = eng.batch([(ids, 48, keys) for ids in eos_prompts])
        assert all(x["ids"] == y["ids"] and len(y["ids"]) < 48 for x, y in zip(eos_refs, eos["jobs"]))
        result["cases"].append({"name": "eos", "solo": eos_refs, "batch": eos})
        if a.capacity_tokens:
            header = tok.encode("<|im_start|>user\n", parse_special=True)
            tail = tok.encode("\nWrite a detailed, 2000-word explanation of why schedules are useful, with many examples.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)
            piece = tok.encode("A report covers the schedule and budget. ")
            count = a.capacity_tokens - len(header) - len(tail)
            assert count > 0 and a.capacity_tokens + 256 < a.context
            long0 = header + (piece * ((count + len(piece) - 1) // len(piece)))[:count] + tail
            long1 = long0[:-len(tail)] + tok.encode("The final report is ready. ") + tail
            # Keep the first slot alive through the second long admission, so this checks
            # full-length state coexistence and actual overlapping decode, not two turns.
            cap_refs = [eng.solo(ids, 256, keys) for ids in (long0, long1)]
            cap = eng.batch([(ids, 256, keys) for ids in (long0, long1)])
            result["cases"].append({"name": "two_long_capacity", "prompt_tokens": [len(long0), len(long1)],
                                    "solo": cap_refs, "batch": cap})
            save()
            assert all(x["ids"] == y["ids"] for x, y in zip(cap_refs, cap["jobs"]))
            times0, times1 = [job["times"] for job in cap["jobs"]]
            assert times0[1] < times1[-1] and times1[1] < times0[-1], "long slots did not overlap"
            print(f"near-capacity two long slots: {len(long0)}, {len(long1)} tokens, identical", flush=True)
        if a.growth_tokens:
            paragraph = "The committee reviewed the budget, the schedule and open questions. A report follows next quarter. "
            body = paragraph * (a.growth_tokens // len(tok.encode(paragraph)) + 1)
            long_ids = prompt(body + "\nExplain why a written schedule is useful, in detail.")
            assert len(long_ids) + 256 < a.context
            long_ref = eng.solo(long_ids, 32, keys)
            grown = eng.batch([(long_ids, 32, keys), (prompts[1], a.limit, keys)])
            assert grown["jobs"][0]["ids"] == long_ref["ids"]
            assert grown["jobs"][1]["ids"] == refs[1]["ids"]
            follow = long_ids + long_ref["ids"] + suffix
            next_ref = eng.solo(follow, 24, keys)
            next_batch = eng.batch([(follow, 24, keys), (prompts[1], a.limit, keys)])
            assert next_batch["jobs"][0]["ids"] == next_ref["ids"]
            result["cases"].append({"name": "kv_growth_long_slot", "prompt_tokens": len(long_ids),
                                    "solo": long_ref, "batch": grown, "followup": next_batch})
            print(f"long-slot grow/reuse beside short admission: {len(long_ids)} tokens, identical", flush=True)
        if a.adaptive:
            log = eng.log_path.read_text(encoding="utf8")
            promotions = re.findall(r"safetensors batch adaptive: round=\d+ promoted=(\d+) completed_total=\d+ active_rows=2", log)
            result["batch_adaptive"] = {"two_row_rounds": len(promotions), "promoted": sum(map(int, promotions))}
            save()
            assert promotions and sum(map(int, promotions)) > 0, "no actual two-row batch promotions"
        result["passed"] = True
        save()
        print("cancel/resume, followup, caps: passed", flush=True)
    finally:
        eng.close()


if __name__ == "__main__":
    main()

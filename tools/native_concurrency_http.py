"""Private HTTP acceptance for native parallel requests. Does not start the desktop service."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from serve.server import Service, StrataEngine, Vision, child_env, engine_args, serve, vision_env
from serve.frontend import ChatTemplate
from safetensors_tokenizer import SafetensorsTokenizer
from native_vision_smoke import solid_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_text(encoding="utf8"))
    cfg["exe"] = str(a.exe.resolve())
    cfg["parallel"] = a.parallel
    args = cfg["args"]
    args[args.index("--vram-reserve-mib") + 1] = "1536"
    result = {"config": cfg, "rounds": [], "requests": []}
    def save():
        (a.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    save()
    env = child_env(cfg)
    vision_log = (a.output / "vision.log").open("w", encoding="utf8")
    vision, engine, httpd = None, None, None
    try:
        if cfg.get("vision"):
            vision = Vision(cfg["vision"], log=vision_log, env=vision_env(cfg, env))
        engine = StrataEngine(cfg["exe"], engine_args(cfg), cwd=cfg["cwd"],
                              log=str((a.output / "engine.log").resolve()), env=env)
        tok = SafetensorsTokenizer.from_directory(cfg["tokenizer"])
        svc = Service(engine, tok, ChatTemplate(Path(cfg["chat_template"])), vision=vision,
                      sampling_defaults=cfg.get("sampling"), fit_max_tokens=True)
        svc.allowed_hosts = cfg.get("allowed_hosts", [])
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        result["info"] = dict(engine.info)
        assert engine.batch == (a.parallel if a.parallel > 1 else 0)

        def chat(name, content, limit=96, barrier=None, started=None, disconnect=False, anthropic=False):
            if barrier:
                barrier.wait(timeout=30)
            body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": content}],
                    "max_tokens": limit, "temperature": 0, "stream": True,
                    "chat_template_kwargs": {"enable_thinking": False}}
            if anthropic:
                body.pop("chat_template_kwargs")
                body["thinking"] = {"type": "disabled"}
            else:
                body["stream_options"] = {"include_usage": True}
            req = urllib.request.Request(base + ("/v1/messages" if anthropic else "/v1/chat/completions"),
                data=json.dumps(body).encode("utf8"), headers={"Content-Type": "application/json"})
            t0 = time.perf_counter()
            times, text, usage, finish = [], "", None, None
            with urllib.request.urlopen(req, timeout=300) as response:
                for line in response:
                    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                        continue
                    ev = json.loads(line[6:])
                    if "error" in ev:
                        raise RuntimeError(ev)
                    if anthropic:
                        delta = ev.get("delta", {})
                        part = delta.get("text", "")
                        usage = ev.get("usage", usage)
                        finish = delta.get("stop_reason", finish)
                    else:
                        choice = (ev.get("choices") or [{}])[0]
                        part = choice.get("delta", {}).get("content", "")
                        usage = ev.get("usage") or usage
                        finish = choice.get("finish_reason") or finish
                    if part:
                        text += part
                        times.append(time.perf_counter() - t0)
                        if started:
                            started.set()
                        if disconnect and len(times) >= 12:
                            break
            row = {"name": name, "text": text, "times": times, "wall_s": time.perf_counter() - t0,
                   "usage": usage, "finish": finish, "disconnected": disconnect}
            result["requests"].append(row)
            return row

        questions = ["Explain TCP and QUIC in detail, including congestion control and multiplexing.",
                     "写一个 Python 二分查找函数，解释边界条件和时间复杂度。",
                     "Explain how a B-tree splits a node; give a worked example."]
        refs = [chat(f"reference-{i}", text) for i, text in enumerate(questions)]
        save()
        with ThreadPoolExecutor(max_workers=4) as pool:
            for count in (1, 2, 3):
                for round_id in range(a.rounds):
                    gate = threading.Barrier(count)
                    t0 = time.perf_counter()
                    futures = [pool.submit(chat, f"c{count}-r{round_id}-{i}", questions[i], barrier=gate)
                               for i in range(count)]
                    rows = [f.result(timeout=300) for f in futures]
                    for i, row in enumerate(rows):
                        assert row["text"] == refs[i]["text"], row["name"] + " differs from serial reference"
                        assert row["usage"] and row["usage"]["completion_tokens"] <= 96
                    elapsed = time.perf_counter() - t0
                    result["rounds"].append({"clients": count, "round": round_id, "wall_s": elapsed,
                        "aggregate_tok_s": sum(r["usage"]["completion_tokens"] for r in rows) / elapsed,
                        "ttft_s": [r["times"][0] for r in rows]})
                    save()
                    print(f"HTTP clients={count} round={round_id}: identical; {elapsed:.3f} s; "
                          f"TTFT {[round(r['times'][0],3) for r in rows]}", flush=True)
            # A client disappearing must not kill another slot or leave the engine's protocol out of step.
            barrier = threading.Barrier(2)
            cancelled = pool.submit(chat, "disconnect", questions[0], barrier=barrier, disconnect=True)
            survivor = pool.submit(chat, "survivor", questions[1], barrier=barrier)
            cancelled.result(timeout=300)
            assert survivor.result(timeout=300)["text"] == refs[1]["text"]
            assert chat("after-disconnect", questions[0])["text"] == refs[0]["text"]
            if vision and a.parallel > 1:
                image = a.output / "red.png"
                solid_png(image, (255, 0, 0))
                content = [{"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode("ascii")}},
                    {"type": "text", "text": "What is the solid background color? Answer with one English color word only."}]
                started = threading.Event()
                active = pool.submit(chat, "text-before-image", questions[0], started=started)
                assert started.wait(timeout=120)
                picture = pool.submit(chat, "image-waiting", content, limit=16)
                # This allows the image to encode and enter the engine's writer queue.
                deadline = time.monotonic() + 20
                while not engine.vision_waiting and not picture.done() and time.monotonic() < deadline:
                    time.sleep(0.02)
                late = pool.submit(chat, "text-after-image", questions[1])
                assert active.result(timeout=300)["text"] == refs[0]["text"]
                assert "red" in picture.result(timeout=300)["text"].lower()
                assert late.result(timeout=300)["text"] == refs[1]["text"]
            anth = chat("anthropic", questions[0], anthropic=True)
            assert anth["text"] == refs[0]["text"]
        assert not any(engine.slot_busy)
        assert engine.vision_readers == 0 and not engine.vision_writer
        result["passed"] = True
        save()
        print("HTTP parity, excess-client queue, disconnect, vision exclusion, Anthropic: passed", flush=True)
    finally:
        if httpd:
            httpd.shutdown()
            httpd.server_close()
        if engine:
            engine.unload()
        if vision:
            vision.shutdown()
        vision_log.close()


if __name__ == "__main__":
    main()

"""Compare the adapter with the checkpoint tokenizer, including cache resumes."""
import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer as HFTokenizer
from safetensors_tokenizer import SafetensorsTokenizer
from strata_tokenizer import PromptEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    args = ap.parse_args()
    ours = SafetensorsTokenizer.from_directory(args.model)
    ref = HFTokenizer.from_file(str(args.model / "tokenizer.json"))
    corpus = ["", "hello world", "Cafe\u0301", "Café", "中文和 emoji 🧠🚀", "\x00\t\r\n",
              "def f(x):\n    return x ** 2\n", "1.234e-10", " " * 200, "你好" * 4096]
    corpus += ["before" + t + "e\u0301after" for t in ours.special_tokens]
    for text in corpus:
        assert ours.encode(text, parse_special=True) == ref.encode(text, add_special_tokens=False).ids, repr(text[:80])
    encoder = PromptEncoder(ours)
    prompt = ""
    for i in range(30):
        prompt += f"<|im_start|>user\n{i}: Cafe\u0301 中文\n<|im_end|>\n"
        assert encoder.encode(prompt) == ours.encode(prompt, parse_special=True)
    text = "<|im_start|>user\nquoted </think> token<|im_end|>\n"
    start = text.index("</think>")
    plain = [(start, start + len("</think>"))]
    got = ours.encode(text, parse_special=True, plain=plain)
    assert ours.ids["</think>"] not in got
    assert ours.decode(got) == text
    print(json.dumps({"status": "pass", "hf_cases": len(corpus), "incremental_cases": 30,
                      "vocabulary": len(ours.tokens), "normalization": "original tokenizer.json"}))


if __name__ == "__main__":
    main()

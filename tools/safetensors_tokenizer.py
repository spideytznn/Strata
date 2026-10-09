"""Original Hugging Face tokenizer with Strata's prompt/cache byte interface.

The native path delegates normalizing and BPE to tokenizer.json. Special literals
retain Strata's explicit plain-span handling for quoted control tokens.
"""
import json
from pathlib import Path

from strata_tokenizer import Tokenizer


class SafetensorsTokenizer(Tokenizer):
    @classmethod
    def from_directory(cls, directory):
        from tokenizers import Tokenizer as HFTokenizer

        root = Path(directory)
        raw = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
        if raw["model"]["type"] != "BPE" or raw["decoder"]["type"] != "ByteLevel":
            raise ValueError("native tokenizer requires byte-level BPE")
        vocab = dict(raw["model"]["vocab"])
        types_by_id = {}
        for token in raw["added_tokens"]:
            if any(token.get(k, False) for k in ("lstrip", "rstrip", "single_word", "normalized")):
                raise ValueError("unsupported added-token matching flags")
            if token["content"] in vocab and vocab[token["content"]] != token["id"]:
                raise ValueError("conflicting added-token ID")
            vocab[token["content"]] = token["id"]
            types_by_id[token["id"]] = 3 if token["special"] else 4
        if sorted(vocab.values()) != list(range(len(vocab))):
            raise ValueError("tokenizer vocabulary IDs must be contiguous and unique")
        tokens = [None] * len(vocab)
        types = [1] * len(vocab)
        for token, index in vocab.items():
            tokens[index] = token
            types[index] = types_by_id.get(index, 1)
        merges = [" ".join(m) if isinstance(m, list) else m for m in raw["model"]["merges"]]
        obj = cls(tokens, merges, types)
        # Added tokens are matched before normalization by the inherited prompt
        # splitter. Removing them here also makes explicitly quoted spans plain.
        raw["added_tokens"] = []
        obj._hf_plain = HFTokenizer.from_str(json.dumps(raw, ensure_ascii=False))
        return obj

    def _encode_plain(self, text):
        return self._hf_plain.encode(text, add_special_tokens=False).ids

#!/usr/bin/env python3
"""gguf_types.py QUANT [QUANT ...]: the tensor types of Unsloth's Laguna S 2.1 GGUFs, read from the shards' headers
over HTTP range requests (a few MB a shard, no download).  Prints, per quant, the tensors grouped by role (routed
experts, shared expert, dense FFN, attention, embeddings/output, norms) with the ggml types and their counts, and
writes the per-tensor list to gguf-types-<QUANT>.json beside this script.

    python laguna/gguf_types.py UD-Q3_K_XL UD-Q4_K_S UD-Q5_K_S
"""
import collections
import json
import re
import struct
import sys
import urllib.request
from pathlib import Path

REPO = "unsloth/Laguna-S-2.1-GGUF"
REV = "750f92f90cf54159c4d7a610cb7b3e74498e75c6"
TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
         12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
         20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
         29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}


def fetch(url, n):
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{n - 1}", "User-Agent": "gguf-types"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


class Reader:
    def __init__(self, b):
        self.b, self.o = b, 0

    def take(self, fmt):
        v = struct.unpack_from("<" + fmt, self.b, self.o)
        self.o += struct.calcsize("<" + fmt)
        return v[0] if len(v) == 1 else v

    def string(self):
        n = self.take("Q")
        s = self.b[self.o:self.o + n].decode("utf-8", "replace")
        self.o += n
        return s

    def value(self, t):
        scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if t in scalar:
            return self.take(scalar[t])
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.take("I"), self.take("Q")
            if et in scalar:                      # skip the array's body fast
                size = struct.calcsize("<" + scalar[et])
                self.o += size * n
                return f"<array of {n}>"
            return [self.value(et) for _ in range(n)]
        raise ValueError(f"gguf value type {t}")


def header(url):
    """-> (kv dict without big arrays, [(name, dims, type)])"""
    n = 4 << 20
    while True:
        b = fetch(url, n)
        try:
            r = Reader(b)
            assert r.take("4s") == b"GGUF", "not a GGUF"
            r.take("I")
            n_tensors, n_kv = r.take("Q"), r.take("Q")
            kv = {}
            for _ in range(n_kv):
                k = r.string()
                kv[k] = r.value(r.take("I"))
            tensors = []
            for _ in range(n_tensors):
                name = r.string()
                dims = [r.take("Q") for _ in range(r.take("I"))]
                tensors.append((name, dims, TYPES.get(r.take("I"), "?")))
                r.take("Q")
            return kv, tensors
        except struct.error:
            if len(b) < n:
                raise
            n *= 4                                # the header is longer than what was read: read more


def role(name):
    if re.search(r"\.ffn_(gate|up|down|gate_up)_exps\.", name):
        return "routed experts " + re.search(r"ffn_(\w+?)_exps", name).group(1)
    if re.search(r"\.ffn_(gate|up|down)_shexp\.", name):
        return "shared expert"
    if re.search(r"\.ffn_(gate|up|down)\.", name):
        return "dense FFN (layer 0)"
    if ".attn_" in name and "norm" not in name:
        return "attention " + re.search(r"attn_(\w+?)\.", name).group(1)
    if "norm" in name:
        return "norms"
    if name.startswith(("token_embd", "output")):
        return name.split(".")[0]
    return "other: " + name.split(".")[-2]


def main():
    for quant in sys.argv[1:] or ["UD-Q3_K_XL", "UD-Q4_K_S", "UD-Q5_K_S"]:
        tensors = []
        for shard in (1, 2, 3):
            url = (f"https://huggingface.co/{REPO}/resolve/{REV}/{quant}/"
                   f"Laguna-S-2.1-{quant}-{shard:05d}-of-00003.gguf")
            kv, t = header(url)
            tensors += t
            if shard == 1:
                ftype = kv.get("general.file_type")
        groups = collections.defaultdict(collections.Counter)
        for name, dims, typ in tensors:
            groups[role(name)][typ] += 1
        print(f"\n{quant}: {len(tensors)} tensors (general.file_type {ftype})")
        for g in sorted(groups):
            print(f"  {g:34s} " + ", ".join(f"{t} x{n}" for t, n in sorted(groups[g].items())))
        out = Path(__file__).with_name(f"gguf-types-{quant}.json")
        out.write_text(json.dumps([{"name": n, "dims": d, "type": t} for n, d, t in tensors], indent=0))


if __name__ == "__main__":
    main()

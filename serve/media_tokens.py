"""serve/media_tokens.py - how many prompt tokens llama-server's mtmd makes of a picture, before it is sent.

Puddle plans a request by its prompt's length: the room left for the answer, and the 400 when it does not fit.  With
llama-server reading the pictures itself (a config with "engine": "llama-server" and --mmproj), a picture's length
comes from the size mtmd resizes it to, which this file works out as mtmd does at llama.cpp 08c252e22
(tools/mtmd/mtmd-image.cpp's calc_size_preserved_ratio, in float32 as there, and mtmd-model.cpp's n_output_tokens).
llama-server's own count, in its reply's last event, stays the authority: the engine says when the two differ.

Only projector types whose rule is written here are counted; any other has no estimate (None).
"""
from __future__ import annotations

import math
import struct

# projector types counted here: one picture resized within min..max pixels to a multiple of patch x merge, its
# (W/patch) x (H/patch) patches merged merge x merge into one token each, between two tokens of text around it
DYN_SIZE = {"nemotron_omni": ("<img>", "</img>")}


def _f32(x: float) -> float:
    """x rounded to a float32, as C++'s float arithmetic leaves it (each +, *, / and sqrt of float32 values computed
    in double and rounded once gives the float32 result)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def image_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) of a PNG, JPEG, BMP or GIF from its header (what stb_image will decode), or None."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        return struct.unpack("<HH", data[6:10])
    if data[:2] == b"BM" and len(data) >= 26:
        w, h = struct.unpack("<ii", data[18:26])
        return abs(w), abs(h)
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                return None
            marker = data[i + 1]
            if marker == 0xFF:                           # fill bytes
                i += 1
                continue
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:   # no length
                i += 2
                continue
            length = struct.unpack(">H", data[i + 2:i + 4])[0]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):   # a start of frame
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h
            i += 2 + length
    return None


def resized(width: int, height: int, align: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """mtmd's calc_size_preserved_ratio (longest_edge 0, round_half_even false): the size it resizes a picture to."""
    if width <= 0 or height <= 0:
        return 0, 0
    f = float(align)

    def round_by(x):                                     # std::round: halves away from zero
        q = _f32(x / f)
        return int(math.floor(q + 0.5)) * align

    def ceil_by(x):
        return int(math.ceil(_f32(x / f))) * align

    def floor_by(x):
        return int(math.floor(_f32(x / f))) * align

    w_bar, h_bar = max(align, round_by(float(width))), max(align, round_by(float(height)))
    if max_pixels > 0 and h_bar * w_bar > max_pixels:
        beta = _f32(math.sqrt(_f32(_f32(float(height) * width) / max_pixels)))
        h_bar = max(align, floor_by(_f32(height / beta)))
        w_bar = max(align, floor_by(_f32(width / beta)))
    elif min_pixels > 0 and h_bar * w_bar < min_pixels:
        beta = _f32(math.sqrt(_f32(float(min_pixels) / _f32(float(height) * width))))
        h_bar = ceil_by(_f32(height * beta))
        w_bar = ceil_by(_f32(width * beta))
    return w_bar, h_bar


def mmproj_params(path) -> dict | None:
    """What the count needs from an mmproj GGUF's metadata, or None when its projector type is not one counted here
    (or the file cannot be read)."""
    try:
        from gguf_reader import GGUFFile
        m = GGUFFile(path).metadata
    except Exception:  # noqa: BLE001 - no estimate, llama-server still counts
        return None
    kind = str(m.get("clip.projector_type") or "")
    if kind not in DYN_SIZE:
        return None
    try:
        return {"projector": kind, "patch": int(m["clip.vision.patch_size"]),
                "merge": int(m.get("clip.vision.projector.scale_factor") or 1),
                "min_pixels": int(m["clip.vision.image_min_pixels"]), "max_pixels": int(m["clip.vision.image_max_pixels"])}
    except (KeyError, TypeError, ValueError):
        return None


def picture_tokens(data: bytes, params: dict, tokenizer=None) -> int | None:
    """The prompt tokens mtmd makes of one picture (its rows, and the text tokens around them: <img> and </img> are
    one token each in Nemotron's vocabulary), or None when the picture's size cannot be read."""
    size = image_size(data)
    if size is None or not params:
        return None
    align = params["patch"] * params["merge"]
    w, h = resized(size[0], size[1], align, params["min_pixels"], params["max_pixels"])
    rows = (w // params["patch"]) * (h // params["patch"]) // (params["merge"] * params["merge"])
    around = 0
    for text in DYN_SIZE[params["projector"]]:
        around += len(tokenizer.encode(text, parse_special=True)) if tokenizer is not None else 1
    return rows + around

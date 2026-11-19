"""The elastic image encoder (on the CPU, the default there): the first picture starts it, later pictures use it as it
is, an idle encoder stops, the expert cache is never asked for VRAM, and the engine's load never starts it on its own.
An encoder on the GPU is never elastic (strata-poly's expert cache does not make room for one; phase 1d)."""
import tempfile
import time
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, Vision
from serve.test_vram import VramEngine

ROOT = Path(__file__).resolve().parents[1]
IMAGE = {"type": "image", "source": "x.png"}


class ElasticVision:
    """An encoder process stand-in: starts, stops, and turns every picture into three image tokens."""
    def __init__(self, d, elastic=True):
        self.elastic, self.elastic_idle_s, self.on_cpu = elastic, 60.0, True
        self.up, self.starts, self.unloads = False, 0, 0
        self.dir = Path(d)
        self.rows = self.dir / "img.sve"
        self.rows.write_bytes(b"rows")

    def alive(self):
        return self.up

    def restart(self):
        self.up, self.starts = True, self.starts + 1

    def unload(self):
        self.up, self.unloads = False, self.unloads + 1

    def encode(self, source):
        assert self.up, "a picture reached an encoder that was not started"
        return self.rows, 3


class Elastic(unittest.TestCase):
    def setUp(self):
        d = tempfile.mkdtemp()
        self.tok = ByteTokenizer()
        self.engine = VramEngine(self.tok)
        self.engine.loaded = True
        self.vision = ElasticVision(d)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=self.vision)
        self.addCleanup(self.drop_embeddings)

    def drop_embeddings(self):
        path = getattr(getattr(self.svc, "embeddings", None), "path", None)
        if path is not None:
            path.unlink(missing_ok=True)

    def picture(self):
        self.svc.prepare([{"role": "user", "content": [{"type": "text", "text": "what is it"}, IMAGE]}], None, {})

    def test_the_first_picture_starts_the_encoder_and_an_idle_one_stops(self):
        self.picture()
        self.assertEqual(self.vision.starts, 1)
        self.picture()
        self.assertEqual(self.vision.starts, 1)              # a running encoder is used as it is
        self.assertEqual(self.engine.calls, [])             # it takes no VRAM, so the cache is left as it is
        self.assertEqual(self.svc._vision_idle_down(), "busy")   # a picture came less than elastic_idle_s ago
        self.svc.last_image_at = 0
        self.assertEqual(self.svc._vision_idle_down(), "down")
        self.assertFalse(self.vision.up)
        self.assertEqual(self.svc._vision_idle_down(), "nothing")
        self.assertEqual(self.engine.calls, [])

    def test_the_load_never_starts_an_elastic_encoder(self):
        self.assertFalse(self.svc._vision_down())
        self.svc.vision = ElasticVision(tempfile.mkdtemp(), elastic=False)
        self.assertTrue(self.svc._vision_down())            # a non-elastic encoder that is down is started again

    def test_an_unload_stops_it(self):
        self.picture()
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertFalse(self.vision.up)

    def test_the_config_keys(self):
        v = Vision({"exe": "e", "mmproj": "m", "model": "t", "gpu": False, "elastic_idle_s": 30}, lazy=False)
        self.assertTrue(v.elastic and v.on_cpu and v.elastic_idle_s == 30 and not v.alive())   # on the CPU by default
        v = Vision({"exe": "e", "mmproj": "m", "model": "t", "elastic": False}, lazy=True)
        self.assertFalse(v.elastic)                         # and kept running when the config says so
        v = Vision({"exe": "e", "mmproj": "m", "model": "t", "gpu": True, "elastic": True}, lazy=True)
        self.assertFalse(v.elastic)                         # never on the GPU
        self.assertEqual(v.elastic_idle_s, 120.0)


if __name__ == "__main__":
    unittest.main()

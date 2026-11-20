"""Puddle's own families (phase 1c): Laguna-S 2.1 and Laguna-XS 2.1, beside Gemma 4.  Each size's files
are pinned (revision, size, SHA-256), the run config a --yes install writes has the names, the KV format and the image
encoder the live configs on echo have, and Laguna-S is the menu's default.  Mocked (tools/poly_setup_harness.py): no
GPU, no downloads.

    python -m unittest tools.test_setup_puddle_families
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from tools.poly_setup_harness import PolyMain  # noqa: E402

SHA = re.compile(r"/resolve/[0-9a-f]{40}/$")
# echo's run configs (local/liveConfigs, 2026-11-19): the names clients ask the server for
LIVE_NAMES = {("laguna-s", "UD-Q3_K_XL"): ("laguna-s-2.1-ud-q3_k_xl", "Laguna S 2.1 UD-Q3_K_XL"),
              ("laguna-xs", "Q4_K_M"): ("laguna-xs-2.1-q4_k_m", "Laguna XS 2.1 Q4_K_M")}


def arg(cfg, flag):
    a = cfg["args"]
    return a[a.index(flag) + 1] if flag in a else None


class Catalog(unittest.TestCase):
    def test_the_families_and_their_order(self):
        self.assertEqual(list(setup.FAMILIES), ["laguna-s", "laguna-xs", "gemma4"])
        self.assertEqual(list(setup.family_models("laguna-s")), ["UD-Q3_K_XL", "UD-Q4_K_XL", "Q4_K_M"])
        self.assertEqual(list(setup.family_models("laguna-xs")), ["Q4_K_M"])

    def test_every_file_is_pinned_with_its_size_and_sha256(self):
        for family, fam in setup.FAMILIES.items():
            for model, d in setup.family_models(family).items():
                sha = {**fam["sha256"], **d.get("sha256", {})}
                with self.subTest(family=family, model=model):
                    self.assertRegex(d.get("hf", fam["hf"]), SHA)
                    files = [setup.model_file(fam, model, i) for i in range(1, setup.model_shards(fam, model) + 1)]
                    if d.get("draft_model"):                      # a size's download has its draft model too
                        files.append(d["draft_model"])
                    total = 0
                    for f in files:
                        size, digest = sha[Path(f).name]
                        self.assertRegex(digest, r"^[0-9a-f]{64}$")
                        total += size
                    self.assertAlmostEqual(total / 1e9, d["download_gb"], delta=0.06)
                    self.assertLess(d["arena_gb"], d["download_gb"])
                    if fam.get("vision"):
                        self.assertRegex(d.get("mmproj_hf", fam["mmproj_hf"]), SHA)
                        self.assertIn(fam["mmproj"], sha)

    def test_the_architectures(self):
        self.assertEqual(setup.POLY_GGUF_ARCHS["laguna-s"], ("laguna",))
        self.assertEqual(setup.POLY_GGUF_ARCHS["laguna-xs"], ("laguna",))
        self.assertEqual(setup.POLY_GGUF_ARCHS["gemma4"], ("gemma4",))

    def test_laguna_s_shards_and_poolside_s_own_file(self):
        fam = setup.FAMILIES["laguna-s"]
        self.assertEqual([setup.model_file(fam, "UD-Q3_K_XL", i) for i in (1, 2, 3)],
                         [f"UD-Q3_K_XL/Laguna-S-2.1-UD-Q3_K_XL-0000{i}-of-00003.gguf" for i in (1, 2, 3)])
        d = setup.family_models("laguna-s")["Q4_K_M"]
        self.assertEqual(setup.model_shards(fam, "Q4_K_M"), 1)
        self.assertEqual(setup.model_file(fam, "Q4_K_M", 1), "laguna-s-2.1-Q4_K_M.gguf")
        self.assertIn("/poolside/Laguna-S-2.1-GGUF/", d["hf"])
        self.assertIn("BF16", d["about"])                           # what it costs is said in the menu


class Install(unittest.TestCase):
    setUp, tearDown, main = PolyMain.setUp, PolyMain.tearDown, PolyMain.main

    def test_each_family_writes_its_sampling(self):
        # Laguna: echo's testing (min_p 0.1); Gemma 4: Google's model card
        want = {"laguna-s": {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.1},
                "laguna-xs": {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.1},
                "gemma4": {"temperature": 1.0, "top_p": 0.95, "top_k": 64}}
        self.assertEqual({f: fam.get("sampling") for f, fam in setup.FAMILIES.items()}, want)
        for family, sampling in want.items():
            with self.subTest(family=family):
                self.tearDown()
                self.setUp()
                code, out, cfg = self.main(family=family, ram=128.0, vram=15.9, n_gpus=2)
                self.assertEqual(code, 0, out[-3000:])
                self.assertEqual(cfg["sampling"], sampling)
                self.assertIn("sampling for requests that set none", out)
                # the server takes the block as it is (a bad value would stop it at its start)
                sys.path.insert(0, str(ROOT / "serve"))
                from serve import server
                self.assertEqual(server.sampling_defaults_from_config(cfg), sampling)

    def test_a_sampling_block_of_the_user_s_is_kept(self):
        mine = {"temperature": 0.6, "top_p": 0.9}
        code, out, cfg = self.main(family="laguna-s", model="UD-Q3_K_XL", ram=128.0, vram=15.9, n_gpus=2,
                                   configs=[("strata-laguna-s21-ud-q3_k_xl.json",
                                             {"exe": "x", "args": [], "sampling": mine})])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["sampling"], mine)
        self.assertNotIn("sampling for requests that set none", out)

    def test_the_menu_s_default_is_laguna_s(self):
        code, out, cfg = self.main(family=None, ram=128.0, vram=15.9, n_gpus=2)
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["model_name"], "laguna-s-2.1-ud-q3_k_xl")

    def test_the_names_are_the_live_configs(self):
        for (family, model), (name, display) in LIVE_NAMES.items():
            with self.subTest(family=family):
                self.tearDown()
                self.setUp()
                code, out, cfg = self.main(family=family, model=model, ram=128.0, vram=15.9, n_gpus=2)
                self.assertEqual(code, 0, out[-3000:])
                self.assertEqual((cfg["model_name"], cfg["display_name"]), (name, display))

    def test_laguna_s_q3(self):
        code, out, cfg = self.main(family="laguna-s", model="UD-Q3_K_XL", ram=128.0, vram=15.9, n_gpus=2)
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(arg(cfg, "--kv"), "q8_0")                  # as it runs on echo
        model = Path(arg(cfg, "--model"))
        self.assertEqual(model.parts[-3:], ("laguna-s21-UD-Q3_K_XL", "UD-Q3_K_XL",
                                            "Laguna-S-2.1-UD-Q3_K_XL-00001-of-00003.gguf"))
        self.assertEqual(len(self.downloads), 3)
        self.assertTrue(all("/unsloth/Laguna-S-2.1-GGUF/resolve/750f92f9" in u and "/UD-Q3_K_XL/" in u
                            for u in self.downloads), self.downloads)
        self.assertNotIn("vision", cfg)
        self.assertIn("OpenMDW-1.1", out)

    def test_poolside_s_q4_k_m_comes_from_poolside(self):
        code, out, cfg = self.main(family="laguna-s", model="Q4_K_M", ram=128.0, vram=15.9, n_gpus=2)
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(self.downloads, ["https://huggingface.co/poolside/Laguna-S-2.1-GGUF/resolve/"
                                          "19036076775e9ba6758595f78af99cb976ef8ff0/laguna-s-2.1-Q4_K_M.gguf"])
        self.assertEqual(arg(cfg, "--kv"), "q8_0")

    def test_laguna_xs(self):
        code, out, cfg = self.main(family="laguna-xs", ram=64.0, vram=15.9)
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(arg(cfg, "--kv"), "q8_0")
        self.assertEqual(Path(arg(cfg, "--model")).parts[-2:], ("laguna-xs21-Q4_K_M", "Laguna-XS-2.1-Q4_K_M.gguf"))

    def test_a_small_pc_gets_the_smaller_size(self):
        code, out, cfg = self.main(family="gemma4", ram=14.0, vram=7.9)   # UD-Q3_K_M wants more
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["model_name"], "gemma-4-26b-a4b-ud-q2_k_xl")


if __name__ == "__main__":
    unittest.main()

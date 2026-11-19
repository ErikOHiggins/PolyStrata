"""Tests for image generation's setup side (tools/image/sdimage.py, setup.py --images flux2-klein): the pins, the quant
and backend choices, the CMake definitions, the source unpacking with its submodules, pnpm for the Web UI, and the run
config's "images" section.  No GPU, no network, nothing is compiled.

    python -m unittest tools.test_sdimage
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "image"))
import sdimage as SI  # noqa: E402
import setup  # noqa: E402


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


class Pins(unittest.TestCase):
    def test_shapes(self):
        commits = (SI.SD_CPP_COMMIT, SI.SD_GGML_COMMIT, SI.SD_WEBUI_COMMIT, SI.CUTLASS_COMMIT, *SI.REVISIONS.values())
        for c in commits:
            self.assertRegex(c, r"^[0-9a-f]{40}$")
        self.assertTrue(SI.SD_CPP_RELEASE.endswith(SI.SD_CPP_COMMIT[:7]))
        for name, (repo, path, size, sha, category) in SI.FILES.items():
            self.assertIn(repo, SI.REVISIONS, name)
            self.assertRegex(sha, r"^[0-9a-f]{64}$")
            self.assertGreater(size, 100 << 20)
            self.assertIn(category, SI.CATEGORIES)
        for q in SI.QUANTS:
            self.assertEqual([SI.FILES[n][4] for n in SI.files_for(q)], ["diffusion_models", "text_encoders", "vae"])

    def test_urls_and_places(self):
        self.assertEqual(SI.file_url("Q8_0"), "https://huggingface.co/leejet/FLUX.2-klein-4B-GGUF/resolve/"
                         "3b1f5a9dc3abb32238b053aeb3d823c30afdacbd/flux-2-klein-4b-Q8_0.gguf")
        self.assertEqual(SI.file_url("vae", "https://hf-mirror.com/"),
                         "https://hf-mirror.com/Comfy-Org/Flux-2-Klein-4B/resolve/"
                         "5b3abca5c0dc1ecd4eaf3340868d09063fc1c965/split_files/vae/flux2-vae.safetensors")
        self.assertEqual(SI.local_path(Path("m"), "vae"), Path("m") / "images" / "vae" / "flux2-vae.safetensors")
        self.assertEqual(SI.file_id("llm-Q8_0"), "text_encoders/Qwen3-4B-Q8_0.gguf")
        with tempfile.TemporaryDirectory() as d:
            root = SI.make_model_root(Path(d))
            self.assertEqual(sorted(p.name for p in root.iterdir()), sorted(SI.CATEGORIES))


class Choices(unittest.TestCase):
    def test_quant_by_memory(self):
        self.assertEqual([SI.pick_quant(g) for g in (4, 8, 8.5, 12, 24)], ["Q4_0", "Q4_0", "Q8_0", "Q8_0", "Q8_0"])
        self.assertEqual(SI.files_for("Q4_0"), ["Q4_0", "llm-Q4_K_M", "vae"])
        self.assertEqual(SI.files_for("Q8_0"), ["Q8_0", "llm-Q8_0", "vae"])

    def test_backend(self):
        self.assertEqual(SI.pick_backend(None, False), "cpu")
        self.assertEqual(SI.pick_backend({"vendor": "nvidia"}, False), "cuda")
        self.assertEqual(SI.pick_backend({"vendor": "amd"}, False), "hip")
        self.assertEqual(SI.pick_backend({"vendor": "amd", "uma": True}, False), "vulkan")
        self.assertEqual(SI.pick_backend({"vendor": "amd"}, True), "vulkan")                  # AMD on Windows
        self.assertEqual(SI.pick_backend({"vendor": "intel"}, False), "vulkan")
        self.assertEqual(SI.pick_backend({"vendor": "amd", "arch": "gfx1103"}, False), "vulkan")

    def test_default_args(self):
        self.assertEqual(SI.default_args("cuda"), ["--offload-to-cpu"])
        self.assertEqual(SI.default_args("vulkan"), ["--offload-to-cpu", "--vae-tiling", "--vae-tile-overlap", "0.25"])

    def test_setup_recommends_never_forces(self):
        ns = argparse.Namespace(image_backend=None, image_quant=None)
        self.assertEqual(quiet(setup.image_backend_choice, ns, {"vram_gb": 24.0, "arch": 86}, False)[:2],
                         ("cuda", "Q8_0"))
        self.assertEqual(quiet(setup.image_backend_choice, ns, {"vram_gb": 8.0, "arch": 86}, False)[:2],
                         ("cuda", "Q4_0"))
        ns.image_quant = "Q8_0"                      # more than recommended: kept, with a warning
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            r = setup.image_backend_choice(ns, {"vram_gb": 6.0, "arch": 86}, False)
        self.assertEqual(r[:2], ("cuda", "Q8_0"))
        self.assertIn("recommends Q4_0", out.getvalue())
        ns.image_quant, ns.image_backend = None, "cpu"
        self.assertEqual(quiet(setup.image_backend_choice, ns, {"vram_gb": 24.0}, False)[:2], ("cpu", "Q4_0"))


class Build(unittest.TestCase):
    def test_cmake_defs(self):
        d = SI.cmake_defs("cuda", arch=86, cuda_root=Path("/usr/local/cuda"), pnpm="/x/pnpm")
        for want in ("-DSD_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=86", "-DSD_WEBP=OFF", "-DSD_BUILD_EXAMPLES=ON",
                     "-DSD_SERVER_BUILD_FRONTEND=ON", "-DPNPM_EXECUTABLE=/x/pnpm"):
            self.assertIn(want, d)
        self.assertIn("-DSD_VULKAN=ON", SI.cmake_defs("vulkan"))
        self.assertFalse([x for x in SI.cmake_defs("cpu") if x.startswith("-DPNPM")])
        with tempfile.TemporaryDirectory() as r:
            rocm = Path(r)
            (rocm / "lib" / "llvm" / "bin").mkdir(parents=True)
            d = SI.cmake_defs("hip", arch="gfx1103:sramecc-:xnack-", rocm=rocm)
            self.assertIn("-DGPU_TARGETS=gfx1103", d)
        with self.assertRaises(ValueError):
            SI.cmake_defs("hip", arch="gfx1103")
        with self.assertRaises(ValueError):
            SI.cmake_defs("metal")

    def test_submodules(self):
        self.assertEqual(SI.submodules_for("cuda"), ["ggml", "examples/server/frontend", "thirdparty/cutlass"])
        self.assertEqual(SI.submodules_for("vulkan"), ["ggml", "examples/server/frontend"])
        self.assertIn("cutlass", SI.build_stamp("cuda", 86))
        self.assertNotIn("cutlass", SI.build_stamp("hip", "gfx1100"))
        self.assertEqual(SI.build_stamp("cpu", None)["target"], "sd-server")

    def test_unpack_puts_the_submodules_in_place(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            sd = d / "sd.zip"
            with zipfile.ZipFile(sd, "w") as z:
                z.writestr(f"stable-diffusion.cpp-{SI.SD_CPP_COMMIT}/CMakeLists.txt", "project(sd)")
                z.writestr(f"stable-diffusion.cpp-{SI.SD_CPP_COMMIT}/ggml/", "")
                z.writestr(f"stable-diffusion.cpp-{SI.SD_CPP_COMMIT}/examples/server/frontend/", "")
            subs = {}
            for sub in SI.submodules_for("vulkan"):
                subs[sub] = d / f"{Path(sub).name}.zip"
                with zipfile.ZipFile(subs[sub], "w") as z:
                    z.writestr(f"top-{Path(sub).name}/{SI.SUBMODULE_MARKS[sub]}", sub)
            src = SI.unpack_source(d, sd, subs)
            self.assertEqual((src / "ggml" / "CMakeLists.txt").read_text(), "ggml")
            self.assertEqual((src / "examples/server/frontend/package.json").read_text(), "examples/server/frontend")
            self.assertTrue(SI.source_ok(src, "vulkan"))
            self.assertFalse(SI.source_ok(src, "cuda"))  # CUDA needs CUTLASS too: unpacked again with it
            pin = (src / ".strata-pin").read_text().replace(SI.SD_GGML_COMMIT, "0" * 40)
            (src / ".strata-pin").write_text(pin)
            self.assertFalse(SI.source_ok(src, "vulkan"))  # another pin: unpacked again

    def test_build_is_skipped_when_the_stamp_matches(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "IMAGE_ENGINE", Path(d)):
            (Path(d) / SI.SD_EXE).write_bytes(b"x")
            (Path(d) / SI.WEBUI_FILE).write_bytes(b"<html>")
            (Path(d) / "BUILD.json").write_text(json.dumps({**SI.build_stamp("cuda", 86), "lib_dirs": [d]}))
            with mock.patch.object(setup, "cmake_build") as cb, mock.patch.object(setup, "find_pnpm") as fp:
                exe, webui, libs = quiet(setup.build_image_engine, "cuda", {"arch": 86})
            cb.assert_not_called()
            fp.assert_not_called()
            self.assertEqual((exe, webui, libs), (Path(d) / SI.SD_EXE, Path(d) / SI.WEBUI_FILE, [d]))
            (Path(d) / SI.WEBUI_FILE).unlink()           # no Web UI: built again (here: no nvcc, so it stops)
            with mock.patch.object(setup, "cmake_build") as cb, mock.patch.object(setup, "get_sd_cpp"), \
                    mock.patch.object(setup, "find_pnpm", return_value="pnpm"), \
                    mock.patch.object(setup, "find_nvcc", return_value=(None, None)):
                with self.assertRaises(SystemExit):
                    quiet(setup.build_image_engine, "cuda", {"arch": 86})

    def test_the_build_copies_sd_server_and_its_page(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src, eng = d / "src", d / "engine-image"
            (src / SI.WEBUI_DIST).parent.mkdir(parents=True)
            (src / SI.WEBUI_DIST).write_text("<html>webui</html>")
            (eng / ("sd-cli.exe" if setup.WIN else "sd-cli")).parent.mkdir()
            (eng / ("sd-cli.exe" if setup.WIN else "sd-cli")).write_bytes(b"old")

            def fake_build(s, bdir, target, defs, vcvars, bat):
                self.assertEqual(target, "sd-server")
                self.assertIn("-DPNPM_EXECUTABLE=/p/pnpm", defs)
                (bdir / "bin").mkdir(parents=True)
                (bdir / "bin" / SI.SD_EXE).write_bytes(b"sd-server")
            with mock.patch.object(setup, "IMAGE_ENGINE", eng), mock.patch.object(setup, "ROOT", d), \
                    mock.patch.object(setup, "get_sd_cpp", return_value=src), \
                    mock.patch.object(setup, "find_pnpm", return_value="/p/pnpm"), \
                    mock.patch.object(setup, "find_vcvars", return_value="vcvars.bat"), \
                    mock.patch.object(setup, "cmake_build", side_effect=fake_build):
                exe, webui, libs = quiet(setup.build_image_engine, "cpu", {})
            self.assertEqual(exe.read_bytes(), b"sd-server")
            self.assertEqual(webui.read_text(), "<html>webui</html>")
            self.assertFalse((eng / ("sd-cli.exe" if setup.WIN else "sd-cli")).exists())
            self.assertEqual(json.loads((eng / "BUILD.json").read_text())["target"], "sd-server")


class Pnpm(unittest.TestCase):
    def test_node_is_needed(self):
        with mock.patch.object(setup.shutil, "which", return_value=None), self.assertRaises(SystemExit):
            quiet(setup.find_pnpm)

    def test_old_node_is_refused(self):
        done = mock.Mock(stdout="v18.19.0\n")
        with mock.patch.object(setup.shutil, "which", side_effect=lambda n: "/bin/node" if n == "node" else None), \
                mock.patch.object(setup.subprocess, "run", return_value=done), self.assertRaises(SystemExit):
            quiet(setup.find_pnpm)

    def test_pnpm_on_path_or_installed_locally(self):
        done = mock.Mock(stdout="v22.1.0\n")
        with mock.patch.object(setup.shutil, "which", side_effect=lambda n: f"/bin/{n}"), \
                mock.patch.object(setup.subprocess, "run", return_value=done):
            self.assertTrue(quiet(setup.find_pnpm).startswith("/bin/pnpm"))
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "third_party" / "pnpm" / "node_modules" / ".bin" / ("pnpm.cmd" if setup.WIN else "pnpm")
            calls = []

            def npm_install(cmd, **kw):
                calls.append(cmd)
                exe.parent.mkdir(parents=True)
                exe.write_text("pnpm")
            which = {"node": "/bin/node", "npm": "/bin/npm", "npm.cmd": "/bin/npm.cmd"}
            with mock.patch.object(setup.shutil, "which", side_effect=which.get), \
                    mock.patch.object(setup.subprocess, "run", return_value=done), \
                    mock.patch.object(setup, "ROOT", Path(d)), mock.patch.object(setup, "run", side_effect=npm_install):
                self.assertEqual(quiet(setup.find_pnpm), str(exe))
            self.assertIn(f"pnpm@{SI.PNPM_VERSION}", calls[0])
            self.assertIn("--prefix", calls[0])


class Config(unittest.TestCase):
    def test_section(self):
        c = SI.image_config("vulkan", "Q8_0", Path("/e/sd-server"), Path("/m"), webui=Path("/e/sd-webui.html"),
                            uma=True, lib_dirs=["/l"])
        self.assertEqual(c["model_dir"], str(Path("/m/images")))
        self.assertEqual(c["recipe"], "flux2-klein-4b")
        self.assertEqual(c["models"], {"diffusion_model": "diffusion_models/flux-2-klein-4b-Q8_0.gguf",
                                       "llm": "text_encoders/Qwen3-4B-Q8_0.gguf", "vae": "vae/flux2-vae.safetensors"})
        self.assertEqual(c["args"], ["--offload-to-cpu", "--vae-tiling", "--vae-tile-overlap", "0.25"])
        self.assertTrue(c["uma"])
        self.assertEqual(c["gpu_policy"], "auto")
        from serve.images import ImageServer, old_section
        self.assertFalse(old_section(c))
        with tempfile.TemporaryDirectory() as d:
            ImageServer({**c, "out_dir": d})           # serve accepts what setup writes

    def test_setup_images_writes_the_section(self):
        ns = argparse.Namespace(images="flux2-klein", image_backend=None, image_quant=None, image_gpu_policy=None,
                                yes=True)
        with tempfile.TemporaryDirectory() as d:
            got = []
            with mock.patch.object(setup, "build_image_engine",
                                   return_value=(Path(d) / "sd-server", Path(d) / "sd-webui.html", [])), \
                    mock.patch.object(setup, "download", side_effect=lambda url, dst, what=None: got.append(url)), \
                    mock.patch.object(setup, "verify_sha256"), mock.patch.object(setup, "free_gb", return_value=999):
                cfg = quiet(setup.setup_images, ns, {"vram_gb": 24.0, "arch": 86}, False, Path(d))
            self.assertEqual(len(got), 3)
            self.assertEqual(cfg["quant"], "Q8_0")
            self.assertEqual(cfg["webui"], str(Path(d) / "sd-webui.html"))
            self.assertTrue((Path(d) / "images" / "loras").is_dir())
        ns.images = "no"
        self.assertIsNone(setup.setup_images(ns, {}, False, Path("/nowhere")))

    def test_images_no_drops_the_section(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-q2_0.json"
            p.write_text(json.dumps({"exe": "x", "images": {"exe": "sd-server"}, "mine": 1}))
            quiet(setup.write_setup_config, p, {"exe": "y", "images": None})
            cfg = json.loads(p.read_text())
            self.assertNotIn("images", cfg)
            self.assertEqual(cfg["mine"], 1)

    def test_flags(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        for flag in ("--images", "--image-quant", "--image-backend", "--image-gpu-policy"):
            self.assertRegex(src, re.escape(f'ap.add_argument("{flag}"'))
        self.assertIn('choices=["flux2-klein", "no"]', src)
        self.assertNotIn("--image-native", src)


if __name__ == "__main__":
    unittest.main()

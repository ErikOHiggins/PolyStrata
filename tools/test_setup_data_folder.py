"""setup.py's data folder and settings (phase 1.1a): Puddle keeps its own per-user settings file, puts the model files
in Puddle-data next to the Puddle folder, moves files only from earlier Puddle folders, never from a PolyStrata
install (nor rewrites its configs), and takes a symlink to the data folder (PolyStrata-data -> Puddle-data) for the
data folder itself.  No GPU, no network; every folder is a temporary one.

    python -m unittest tools.test_setup_data_folder
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


def model_file(folder: Path, name="m.gguf", data=b"gguf") -> Path:
    p = folder / "models" / "fam" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def config(folder: Path, model: Path) -> Path:
    c = folder / "strata-fam.json"
    c.write_text(json.dumps({"exe": "x", "args": ["--model", str(model)]}), encoding="utf-8")
    return c


class SettingsFile(unittest.TestCase):
    def test_linux_and_windows_paths_are_puddles(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(setup, "WIN", False), mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": d}):
                self.assertEqual(setup.settings_path(), Path(d) / "puddle" / "settings.json")
            with mock.patch.object(setup, "WIN", True), mock.patch.dict(os.environ, {"APPDATA": d}):
                self.assertEqual(setup.settings_path(), Path(d) / "Puddle" / "settings.json")

    def test_polystratas_settings_file_is_not_read(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": d, "APPDATA": d}):
            for name in ("polystrata", "PolyStrata"):
                (Path(d) / name).mkdir(exist_ok=True)
                (Path(d) / name / "settings.json").write_text(json.dumps({"data_dir": "/elsewhere"}))
            self.assertEqual(setup.load_settings(), {})


class Folders(unittest.TestCase):
    """A home folder with ~/Puddle (this checkout) beside other folders, as on echo."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.root = self.home / "Puddle"
        self.root.mkdir()
        (self.root / "setup.py").write_text("")
        env = {"XDG_CONFIG_HOME": str(self.home / ".config"), "APPDATA": str(self.home / "AppData")}
        self.patches = [mock.patch.object(setup, "ROOT", self.root), mock.patch.dict(os.environ, env)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def install(self, name: str) -> Path:
        f = self.home / name
        f.mkdir()
        (f / "setup.py").write_text("")
        return f

    def test_the_default_data_folder_is_puddle_data_and_remembered(self):
        dest, elsewhere = quiet(setup.data_folder, None)
        self.assertEqual(dest, self.home / "Puddle-data")
        self.assertTrue(dest.is_dir())
        self.assertEqual(elsewhere, [])
        saved = setup.load_settings()
        self.assertEqual(Path(saved["data_dir"]), dest)
        self.assertEqual(saved["installs"], [str(self.root)])

    def test_other_installs_are_puddle_folders_only(self):
        poly = self.install("PolyStrata")
        poly_nested = poly / "PolyStrata-main"
        poly_nested.mkdir()
        (poly_nested / "setup.py").write_text("")
        old = self.install("Puddle-main (1)")
        found = setup.other_installs({})
        self.assertEqual(found, [old.resolve()])
        self.assertNotIn(poly.resolve(), found)
        self.assertNotIn(poly_nested.resolve(), found)

    def test_a_polystrata_install_keeps_its_files_and_configs(self):
        poly = self.install("PolyStrata")
        m = model_file(poly)
        c = config(poly, m)
        before = c.read_text(encoding="utf-8")
        dest, _ = quiet(setup.data_folder, None)
        self.assertTrue(m.is_file(), "a PolyStrata install's model file was moved")
        self.assertEqual(c.read_text(encoding="utf-8"), before, "a PolyStrata install's config was rewritten")
        self.assertFalse((dest / "models" / "fam" / "m.gguf").exists())

    def test_an_earlier_puddle_folder_moves_into_the_data_folder(self):
        old = self.install("Puddle-old")
        m = model_file(old)
        c = config(old, m)
        dest, elsewhere = quiet(setup.data_folder, None)
        moved = dest / "models" / "fam" / "m.gguf"
        self.assertTrue(moved.is_file())
        self.assertFalse(m.exists())
        self.assertEqual(json.loads(c.read_text(encoding="utf-8"))["args"], ["--model", str(moved)])
        self.assertEqual(elsewhere, [])

    def test_a_symlink_to_the_data_folder_is_the_data_folder(self):
        """echo: ~/PolyStrata-data -> ~/Puddle-data, and a settings file that names the link.  Nothing moves onto
        itself, and every file stays."""
        data = self.home / "Puddle-data"
        m = model_file(data, data=b"x" * 4096)
        link = self.home / "PolyStrata-data"
        try:
            os.symlink(data, link, target_is_directory=True)
        except (OSError, NotImplementedError) as e:     # Windows without the privilege: a junction needs none
            try:
                import _winapi
                _winapi.CreateJunction(str(data), str(link))
            except (ImportError, OSError, AttributeError):
                self.skipTest(f"no symlink or junction here ({e})")
        self.assertTrue(setup.same_folder(link, data))
        setup.save_settings({"data_dir": str(link)})
        dest, elsewhere = quiet(setup.data_folder, None)
        self.assertTrue(setup.same_folder(dest, data))
        self.assertEqual(m.read_bytes(), b"x" * 4096)
        self.assertTrue(link.exists() and setup.same_folder(link, data))   # the link is still a link
        self.assertEqual(elsewhere, [])
        dest, elsewhere = quiet(setup.data_folder, str(data))   # --data-dir the real folder, the link remembered
        self.assertEqual(dest, data.resolve())
        self.assertEqual(m.read_bytes(), b"x" * 4096)
        self.assertEqual(elsewhere, [])

    def test_same_folder(self):
        a = self.home / "a"
        a.mkdir()
        self.assertTrue(setup.same_folder(a, self.home / "a"))
        self.assertTrue(setup.same_folder(a, self.home / "b" / ".." / "a"))
        self.assertFalse(setup.same_folder(a, self.home / "b"))
        self.assertFalse(setup.same_folder(self.home / "x", self.home / "y"))   # neither exists


class Texts(unittest.TestCase):
    def test_puddles_names_in_the_launchers(self):
        for name in ("update.sh", "UPDATE.bat"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("ErikOHiggins/Puddle/archive/refs/heads/main.zip", text, name)
            self.assertNotIn("PolyStrata-data", text, name)
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('DATA_FOLDER_NAME = "Puddle-data"', src)
        self.assertNotIn('ROOT.parent / "PolyStrata-data"', src)


if __name__ == "__main__":
    unittest.main()

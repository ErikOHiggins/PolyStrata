"""tools/image/sdimage.py - FLUX.2 [klein] 4B for PolyStrata's image page: stable-diffusion.cpp's own server (sd-server)
with its Web UI (sdcpp-webui) at pinned commits, built for this PC's GPU, and the model's three files at pinned Hugging
Face revisions.

setup.py (--images flux2-klein) and tools/image/install.py use these pieces; serve/images.py runs the result.

  source    stable-diffusion.cpp release master-1106-532d2c1 with the commits of its submodules: ggml, the Web UI
            (examples/server/frontend) and, for CUDA builds, CUTLASS 3.9.2 (thirdparty/cutlass: the CuTe attention and
            INT8 GEMM kernels).  GitHub's zips leave submodules out, so each comes as its own zip (no git).
  web UI    built by sd.cpp's CMake with pnpm (-DSD_SERVER_BUILD_FRONTEND=ON) into one HTML file; setup copies it beside
            sd-server (WEBUI_FILE) and PolyStrata's server serves it at /sd/.  Needs Node.js 20 or newer.
  backend   CUDA on an NVIDIA card, HIP on an AMD card with its own memory (Linux), Vulkan on an APU/iGPU (AMD's shared
            memory: the HIP path's queue evictions reset a 780M on current firmware), on Intel and on AMD under
            Windows; CPU without a GPU.  --image-backend picks another.
  files     in the category folders sd-server's model catalog reads (models/images/diffusion_models, text_encoders,
            vae; loras/ is made empty for the user's own): the denoiser by the GPU's memory, Q4_0 up to 8 GB, Q8_0
            above (leejet's GGUF); Qwen3-4B as the text encoder at the matching size, Q4_K_M or Q8_0 (Unsloth's GGUF);
            the FLUX.2 VAE (Comfy-Org's split file: BFL's own tensor names, F32).  Each file is checked against the
            SHA-256 the Hugging Face API lists for it at the pinned revision.  All three are Apache 2.0.
  defaults  --offload-to-cpu: the weights stay in RAM and each part (text encoder, denoiser, VAE) is copied to the GPU
            while it runs and released after it, so sd-server keeps little VRAM between images.  On Vulkan also
            --vae-tiling at a quarter's overlap (an untiled 1024x1024 VAE decode is one dispatch long enough to trip
            the amdgpu job timeout on a 780M).  The generation settings are sd.cpp's flux2-klein-4b recipe (4 steps,
            CFG 1, euler, 1024x1024), which the Web UI shows and lets the user change.
"""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

SD_CPP_RELEASE = "master-1106-532d2c1"
SD_CPP_COMMIT = "532d2c15c430a785a712b2d94281f04e733008a4"     # "refactor(api): adopt opaque options ..." (#2365)
SD_GGML_COMMIT = "eeda9ec48fc84a3df0bb9dcc19ae9053075a2539"    # its ggml submodule (leejet/ggml)
SD_WEBUI_COMMIT = "b07665f893047e7b633bbdb69d97c3a9c594076e"   # its examples/server/frontend submodule (leejet/sdcpp-webui)
CUTLASS_COMMIT = "ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e"    # its thirdparty/cutlass submodule (NVIDIA/cutlass 3.9.2)
SD_CPP_ZIP = f"https://github.com/leejet/stable-diffusion.cpp/archive/{SD_CPP_COMMIT}.zip"
SD_GGML_ZIP = f"https://github.com/leejet/ggml/archive/{SD_GGML_COMMIT}.zip"
SD_WEBUI_ZIP = f"https://github.com/leejet/sdcpp-webui/archive/{SD_WEBUI_COMMIT}.zip"
CUTLASS_ZIP = f"https://github.com/NVIDIA/cutlass/archive/{CUTLASS_COMMIT}.zip"
# where each submodule's zip goes in the unpacked source ("" = the top)
SUBMODULES = {"ggml": (SD_GGML_ZIP, SD_GGML_COMMIT),
              "examples/server/frontend": (SD_WEBUI_ZIP, SD_WEBUI_COMMIT),
              "thirdparty/cutlass": (CUTLASS_ZIP, CUTLASS_COMMIT)}
CUDA_ONLY = ("thirdparty/cutlass",)

# the Web UI's build: the pnpm its package.json names ("packageManager"), on Node.js 20 or newer ("engines")
PNPM_VERSION = "10.15.1"
NODE_MIN = (20, 0)

# the LunarG Vulkan SDK for a Linux PC without Vulkan development files (headers, glslc): unpacked into third_party,
# no sudo; the version the 780M builds were made with
VULKAN_SDK_VERSION = "1.4.363.0"
VULKAN_SDK_URL = (f"https://sdk.lunarg.com/sdk/download/{VULKAN_SDK_VERSION}/linux/"
                  f"vulkansdk-linux-x86_64-{VULKAN_SDK_VERSION}.tar.xz")
VULKAN_SDK_SIZE = 366_564_872
VULKAN_SDK_SHA256 = "197962f5cbf80baf2775a03336a01cee7c8745686c65aaa70d3f751ade4d7e43"

REVISIONS = {                                       # the `sha` of https://huggingface.co/api/models/<repo>, 2026-11-16
    "leejet/FLUX.2-klein-4B-GGUF": "3b1f5a9dc3abb32238b053aeb3d823c30afdacbd",
    "unsloth/Qwen3-4B-GGUF": "22c9fc8a8c7700b76a1789366280a6a5a1ad1120",
    "Comfy-Org/Flux-2-Klein-4B": "5b3abca5c0dc1ecd4eaf3340868d09063fc1c965",
}
_DIT = "leejet/FLUX.2-klein-4B-GGUF"
_TE = "unsloth/Qwen3-4B-GGUF"
# name -> (repo, path in the repo, bytes, sha256, category folder): the Hub's LFS pointers at the pinned revisions
FILES = {
    "Q4_0": (_DIT, "flux-2-klein-4b-Q4_0.gguf", 2_460_378_560,
             "d1023499ef3f2f82ff7c50e6778495195c1b6cc34835741778868428111f9ff4", "diffusion_models"),
    "Q8_0": (_DIT, "flux-2-klein-4b-Q8_0.gguf", 4_300_629_440,
             "0bba6951258ec8f92d51114a8fa13e66828297bfff58a738f52729b3ef66fa28", "diffusion_models"),
    "llm-Q4_K_M": (_TE, "Qwen3-4B-Q4_K_M.gguf", 2_497_281_312,
                   "f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a", "text_encoders"),
    "llm-Q8_0": (_TE, "Qwen3-4B-Q8_0.gguf", 4_280_405_792,
                 "eed555233267a33c7e8ee31682762cc7751b3f6d224039086e0e846f05fffa5d", "text_encoders"),
    "vae": ("Comfy-Org/Flux-2-Klein-4B", "split_files/vae/flux2-vae.safetensors", 336_211_292,
            "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3", "vae"),
}
QUANTS = ("Q4_0", "Q8_0")                           # the denoiser's; the text encoder follows (LLM_QUANT)
LLM_QUANT = {"Q4_0": "llm-Q4_K_M", "Q8_0": "llm-Q8_0"}
BACKENDS = ("cuda", "hip", "vulkan", "cpu")
LICENSE_NOTE = ("FLUX.2 [klein] 4B (https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), its VAE and the Qwen3-4B "
                "text encoder are under the Apache 2.0 license")
MODEL_NAME = "flux2-klein-4b"
RECIPE = "flux2-klein-4b"                           # sd-server's built-in recipe (examples/server/recipes)
MODEL_DIR = "images"                                # the model root sd-server gets (--model-dir), in the models folder
CATEGORIES = ("diffusion_models", "text_encoders", "vae", "loras")
SD_EXE = "sd-server.exe" if os.name == "nt" else "sd-server"
WEBUI_FILE = "sd-webui.html"                        # the Web UI's single HTML file, beside sd-server
WEBUI_DIST = Path("examples") / "server" / "frontend" / "dist" / "index.html"


def pick_quant(gpu_gb: float) -> str:
    """The denoiser for a GPU with `gpu_gb` of memory: Q4_0 up to 8 GB, Q8_0 above (the text encoder follows)."""
    return "Q4_0" if gpu_gb <= 8 else "Q8_0"


def files_for(quant: str) -> list[str]:
    """The FILES names one install downloads: denoiser, text encoder, VAE."""
    return [quant, LLM_QUANT[quant], "vae"]


# AMD APUs (integrated GPUs on shared memory) by gfx target: Rembrandt 680M/660M, Mendocino 610M, Phoenix 780M/760M/
# 740M, Strix Point 890M/880M, Strix Halo 8060S, Krackan, Raven/Renoir/Cezanne
APU_ARCHS = ("gfx1035", "gfx1036", "gfx1037", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1153",
             "gfx902", "gfx909", "gfx90c")


def is_apu(gpu: dict | None) -> bool:
    """An integrated GPU on shared memory: setup's "uma" mark, or an APU's gfx target (setup marks only the APUs the
    text engine supports)."""
    if not gpu:
        return False
    return bool(gpu.get("uma")) or str(gpu.get("arch") or "").split(":")[0] in APU_ARCHS


def pick_backend(gpu: dict | None, win: bool) -> str:
    """sd.cpp's backend for setup's GPU dict (vendor, uma, arch): see the module's docstring."""
    if not gpu:
        return "cpu"
    vendor = (gpu.get("vendor") or "nvidia").lower()
    if vendor == "nvidia":
        return "cuda"
    if vendor == "amd":
        return "vulkan" if is_apu(gpu) or win else "hip"
    return "vulkan"                                   # Intel (Arc) and others


def default_args(backend: str) -> list[str]:
    """sd-server's start arguments besides the ones serve/images.py adds (address, port, model folder)."""
    # Vulkan: 256-pixel tiles at a quarter's overlap, not sd.cpp's half: the 780M's 1024x1024 decode took 23 s instead
    # of 45 s at an LPIPS of 0.012 against the half-overlap image (measured with Qwen-Image's VAE)
    return ["--offload-to-cpu"] + (["--vae-tiling", "--vae-tile-overlap", "0.25"] if backend == "vulkan" else [])


def file_url(name: str, endpoint: str = "https://huggingface.co") -> str:
    repo, path = FILES[name][:2]
    return f"{endpoint.rstrip('/')}/{repo}/resolve/{REVISIONS[repo]}/{path}"


def file_id(name: str) -> str:
    """The file's ID in sd-server's model catalog: its path below the model root, with forward slashes."""
    return f"{FILES[name][4]}/{Path(FILES[name][1]).name}"


def local_path(models_dir: Path, name: str) -> Path:
    return Path(models_dir) / MODEL_DIR / FILES[name][4] / Path(FILES[name][1]).name


def make_model_root(models_dir: Path) -> Path:
    """models/images with its category folders (loras/ and the others empty until the user adds files)."""
    root = Path(models_dir) / MODEL_DIR
    for c in CATEGORIES:
        (root / c).mkdir(parents=True, exist_ok=True)
    return root


def cmake_defs(backend: str, arch=None, rocm: Path | None = None, cuda_root: Path | None = None,
               pnpm: str | None = None) -> list[str]:
    """CMake definitions for sd-server on `backend`: arch = the CUDA compute capability ("86") or the gfx target;
    pnpm = the pnpm that builds the Web UI (CMake's own search otherwise)."""
    defs = ["-DSD_BUILD_EXAMPLES=ON", "-DSD_WEBP=OFF", "-DSD_WEBM=OFF", "-DSD_BUILD_SHARED_LIBS=OFF",
            "-DGGML_NATIVE=ON", "-DSD_SERVER_BUILD_FRONTEND=ON"]
    if pnpm:
        defs += [f"-DPNPM_EXECUTABLE={pnpm}"]
    if backend == "cuda":
        defs += ["-DSD_CUDA=ON"] + ([f"-DCMAKE_CUDA_ARCHITECTURES={arch}"] if arch else [])
        if cuda_root:
            defs += [f"-DCUDAToolkit_ROOT={cuda_root}"]
    elif backend == "hip":
        if not rocm:
            raise ValueError("the HIP build needs the ROCm folder")
        gfx = str(arch).split(":")[0]
        llvm = rocm / "lib" / "llvm" if (rocm / "lib" / "llvm" / "bin").is_dir() else rocm / "llvm"
        bitcode = next((p for p in (llvm / "amdgcn" / "bitcode", rocm / "amdgcn" / "bitcode") if p.is_dir()),
                       llvm / "amdgcn" / "bitcode")
        defs += ["-DSD_HIPBLAS=ON", f"-DGPU_TARGETS={gfx}", f"-DAMDGPU_TARGETS={gfx}",
                 f"-DCMAKE_C_COMPILER={llvm / 'bin' / 'clang'}", f"-DCMAKE_CXX_COMPILER={llvm / 'bin' / 'clang++'}",
                 f"-DCMAKE_HIP_COMPILER={llvm / 'bin' / 'clang++'}",
                 f"-DCMAKE_PREFIX_PATH={rocm};{rocm / 'lib' / 'rocm_sysdeps'};{llvm}",
                 f"-DCMAKE_HIP_FLAGS=--rocm-path={rocm} --rocm-device-lib-path={bitcode}"]
    elif backend == "vulkan":
        defs += ["-DSD_VULKAN=ON"]
    elif backend != "cpu":
        raise ValueError(f"unknown backend {backend!r}")
    return defs


def submodules_for(backend: str) -> list[str]:
    """The submodule folders a build on `backend` needs (CUTLASS only for CUDA)."""
    return [s for s in SUBMODULES if s not in CUDA_ONLY or backend == "cuda"]


def pin_lines(subs) -> list[str]:
    return [f"stable-diffusion.cpp {SD_CPP_COMMIT}"] + [f"{s} {SUBMODULES[s][1]}" for s in subs]


def unpack_source(third_party: Path, sd_zip: Path, sub_zips: dict) -> Path:
    """stable-diffusion.cpp's zip with its submodules' zips inside (GitHub's archives leave submodules out; sub_zips:
    folder in the source -> zip) -> third_party/stable-diffusion.cpp, stamped with the commits."""
    dst = Path(third_party) / "stable-diffusion.cpp"
    tmp = Path(third_party) / "_unpack-sd"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(sd_zip) as f:
        f.extractall(tmp / "sd")
    top = next((tmp / "sd").iterdir())
    for i, (sub, z) in enumerate(sub_zips.items()):
        with zipfile.ZipFile(z) as f:
            f.extractall(tmp / f"sub{i}")
        stop = next((tmp / f"sub{i}").iterdir())
        place = top / sub
        shutil.rmtree(place, ignore_errors=True)
        place.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stop), str(place))
    shutil.rmtree(dst, ignore_errors=True)
    shutil.move(str(top), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)
    (dst / ".strata-pin").write_text("\n".join(pin_lines(sub_zips)) + "\n", encoding="utf-8")
    return dst


SUBMODULE_MARKS = {"ggml": "CMakeLists.txt", "examples/server/frontend": "package.json",
                   "thirdparty/cutlass": "include/cute/tensor.hpp"}


def source_ok(src: Path, backend: str = "cpu") -> bool:
    """The unpacked source is the pinned one, with every submodule a build on `backend` needs."""
    pin = Path(src) / ".strata-pin"
    if not pin.is_file():
        return False
    have = set(pin.read_text(encoding="utf-8").splitlines())
    subs = submodules_for(backend)
    return all(line in have for line in pin_lines(subs)) and \
        all((Path(src) / s / SUBMODULE_MARKS[s]).is_file() for s in subs)


def build_stamp(backend: str, arch) -> dict:
    stamp = {"sd_cpp": SD_CPP_COMMIT, "ggml": SD_GGML_COMMIT, "webui": SD_WEBUI_COMMIT, "target": "sd-server",
             "backend": backend, "arch": str(arch or "")}
    if backend == "cuda":
        stamp["cutlass"] = CUTLASS_COMMIT
    return stamp


def image_config(backend: str, quant: str, exe: Path, models_dir: Path, *, webui: Path | None = None,
                 uma: bool = False, gpu=None, lib_dirs=(), policy: str = "auto") -> dict:
    """The run config's "images" section (serve/images.py reads it)."""
    dit, llm, vae = files_for(quant)
    cfg = {"model_name": MODEL_NAME, "exe": str(exe), "backend": backend, "quant": quant,
           "model_dir": str(Path(models_dir) / MODEL_DIR), "recipe": RECIPE,
           "models": {"diffusion_model": file_id(dit), "llm": file_id(llm), "vae": file_id(vae)},
           "args": default_args(backend), "gpu_policy": policy, "uma": bool(uma), "sd_cpp": SD_CPP_COMMIT[:7]}
    if webui is not None:
        cfg["webui"] = str(webui)
    if gpu is not None:
        cfg["gpu"] = gpu
    if lib_dirs:
        cfg["lib_dirs"] = [str(d) for d in lib_dirs]
    return cfg

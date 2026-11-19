# Image generation (FLUX.2 klein 4B in stable-diffusion.cpp's Web UI)

PolyStrata's web app can also make images: its **Images** tab shows the Web UI of
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) ([sdcpp-webui](https://github.com/leejet/sdcpp-webui)),
and PolyStrata's server runs stable-diffusion.cpp's own server (`sd-server`) behind it. The model is
[FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (4 steps, CFG 1) in
[leejet's GGUF quants](https://huggingface.co/leejet/FLUX.2-klein-4B-GGUF). Setup compiles `sd-server` and its Web UI
for your GPU. Image generation is optional and stays off unless you ask for it.

FLUX.2 klein 4B, its VAE and the Qwen3-4B text encoder are under the **Apache 2.0** license.

## Set it up

    ./setup.sh --setup --images flux2-klein          (Linux)
    START-HERE.bat --setup --images flux2-klein      (Windows)

or, for an install that is already set up, without running the whole setup again:

    .venv/bin/python tools/image/install.py --config strata-<model>.json

Setup does four things:

1. Downloads stable-diffusion.cpp release `master-1106-532d2c1` and the commits of its submodules (ggml, the Web UI, and
   for CUDA builds CUTLASS 3.9.2) into `third_party/`. Each comes as a zip; no git is needed.
2. Compiles `sd-server` with its Web UI into `engine-image/`. sd.cpp's CMake builds the Web UI with pnpm, and that
   needs **Node.js 20 or newer**. If pnpm is not on the PATH, setup installs pnpm 10.15.1 into `third_party/pnpm` with
   npm (no sudo). pnpm downloads the Web UI's packages from the npm registry during the build.
3. Downloads three files into the data folder's `models/images/` and checks each against its SHA-256.
4. Writes an `"images"` section into the model's run config.

| piece | file | size |
|---|---|---|
| denoiser (by GPU memory) | `diffusion_models/flux-2-klein-4b-Q4_0.gguf` up to 8 GB, `Q8_0` above | 2.5 / 4.3 GB |
| text encoder (follows the denoiser) | `text_encoders/Qwen3-4B-Q4_K_M.gguf` or `Qwen3-4B-Q8_0.gguf` | 2.5 / 4.3 GB |
| VAE | `vae/flux2-vae.safetensors` (Comfy-Org's split file) | 0.34 GB |

`models/images/` is the model folder `sd-server` gets (`--model-dir`). It also has an empty `loras/`. Files you put
in its category folders (`diffusion_models/`, `text_encoders/`, `vae/`, `loras/`) show up in the Web UI after
**Refresh files & recipes**. For example, FLUX.2 klein base 4B or another model whose
[guide](https://github.com/leejet/stable-diffusion.cpp/tree/master/docs/models) the release covers.

Options:

- `--image-quant Q4_0|Q8_0`: a bigger one than recommended is kept, with a warning.
- `--image-backend cuda|hip|vulkan|cpu`
- `--image-gpu-policy auto|keep|shrink|unload`
- `--images no`: takes the section out of the config.

Backends:

- **CUDA** on NVIDIA. Needs the CUDA Toolkit.
- **HIP** on AMD cards with their own memory (Linux, the same ROCm as the engine).
- **Vulkan** on AMD APUs/iGPUs (Radeon 780M, Strix Point/Halo), on Intel Arc, and on AMD under Windows. A Linux PC
  without the Vulkan development files gets the LunarG Vulkan SDK unpacked into `third_party/` (no sudo). On Windows,
  install the Vulkan SDK first.

On Vulkan, setup adds `--vae-tiling --vae-tile-overlap 0.25`. An untiled 1024x1024 VAE decode on a Radeon 780M is one
GPU job long enough to trip the kernel driver's job timeout. A quarter's overlap decoded in half the time of sd.cpp's
default half overlap (measured with Qwen-Image's VAE, before this change).

An install set up with the earlier Qwen-Image version keeps working for chat. Its `"images"` section (sd-cli) is not
used: the server says so at its start and leaves image generation off until you run `--images flux2-klein`. The old
files in `models/qwen-image-2.1-turbo/` are not used any more, and you can delete them.

## Use it

Open the web app and choose the **Images** tab, or open `http://127.0.0.1:8080/sd/` directly (the tab's **Open in its
own tab**). This is sd.cpp's own Web UI, with all its controls:

- the recipe and model files (the `flux2-klein-4b` recipe is loaded at the start)
- sampler, scheduler, steps, CFG and the recipe's other guidance
- size and batch
- reference images for editing (FLUX.2 klein supports them)
- LoRAs, VAE tiling, caches and live previews

Its upstream guide is the
[server README](https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/server/README.md#web-ui).

The Web UI's **Enhance prompt** sends the prompt to an OpenAI-compatible API. PolyStrata fills in its own `/v1` and
model name (and the web app's API key, if it has one) the first time the page opens, so the button asks the text model
running here. You can change these values in the Web UI's Settings; PolyStrata does not overwrite a value you have set.

What PolyStrata's server passes on is `sd-server`'s native API, `/sd/sdcpp/v1/*`. `sd-server`'s OpenAI
(`/v1/images/generations`) and sdapi routes are not passed on, and PolyStrata's own `/v1/images/generations` is gone
with this version. `sd-server` listens on `127.0.0.1` on a random port.

With an API key (`--api-key`), the page needs it too. When the Images tab opens, the web app sends the key to
`POST /sd/session` and gets a cookie for `/sd/` (HttpOnly, SameSite=Strict), because a frame cannot send the key as a
header. Without a key, a page of another site can neither load `/sd/` in its own frame nor start a job: those requests
are refused (`Sec-Fetch-Site`, `Origin`).

## Starting, stopping and sharing the GPU with the text model

`sd-server` starts the first time the page is used, then loads the config's model (`"recipe"` and `"models"`). It
stops after `"idle_s"` seconds (default 600; 0 = never) without a job, a model change or another action. An open page
that only polls does not keep it running. Stopping gives its RAM back.

While it is stopped, PolyStrata answers the page's catalog and status requests with what `sd-server` said last, so the
page still shows the model as loaded. The next job starts `sd-server` again, loads the model that was loaded last (the
one you chose in the Web UI, if you changed it), and then runs. That job takes the start's time longer. Each start
writes `sd-server`'s log to `sd-server.log` in the images folder (`"out_dir"`, default a temporary folder).

One job at a time: an image job (`img_gen`, `vid_gen`, `upscale`) takes the same turn a chat request takes, from its
submission until `sd-server` reports it completed, failed or cancelled. A chat request waits for the image, and the
other way round. What the text model does meanwhile is set by the config's `images.gpu_policy`:

| policy | the text model during an image job |
|---|---|
| `auto` (default) | `keep` when the image job fits next to it (an APU always), else `shrink` when the engine was started with `--vram-elastic`, else `unload` |
| `keep` | stays loaded and idle |
| `shrink` | gives VRAM from its expert cache (as `POST /v1/vram`) and takes it back afterwards; the conversation stays |
| `unload` | its conversation is saved to a file, the engine is stopped, and the next start restores the conversation (`"reload": "eager"` starts it again right after the image; default `lazy`: with the next chat request) |

`sd-server` runs with `--offload-to-cpu` (the config's `"args"`). The weights stay in RAM, and each part of the model
(text encoder, denoiser, VAE) is copied to the GPU while it runs and released after it, so between jobs `sd-server`
keeps only its GPU context. To keep the weights on the GPU instead, take `--offload-to-cpu` out of `"args"`: images
are then faster, and the VRAM stays taken next to the text model.

The memory a job is planned with (`auto` compares it with the free VRAM) is an estimate that has not been measured
with FLUX.2 klein yet: the larger of the denoiser and the text encoder, plus the VAE, plus 1 GiB and 3 GiB per
megapixel. `"need_mib"` in the config sets it instead.

Other keys of the section:

- `"timeout_s"` (3600): a job running longer is cancelled.
- `"start_timeout_s"` (120)
- `"load_timeout_s"` (1800)
- `"port"`: a fixed port for `sd-server` instead of a random one.
- `"gpu"`: the GPU's number for its backend.
- `"env"`, `"lib_dirs"`

`/v1/status` shows the state under `image_generation`: whether `sd-server` is running, the recipe loaded, the running
job's stage and step, and counts.

## Speed

Not measured yet for FLUX.2 klein 4B in this setup. A report from your hardware is welcome
([docs/COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md)): the GPU, the backend, the quant, the size and steps, the
seconds per image (the Web UI shows them), and whether `sd-server` had to start first.

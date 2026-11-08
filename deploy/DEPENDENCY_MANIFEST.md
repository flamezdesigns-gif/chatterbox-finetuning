# Dependency manifest

Date of the package and wheel lookups: 2026-11-08. Nothing in this manifest was installed in the review environment, and no model body was downloaded.

## Decision

Keep the observed CUDA 12.8 stack. Do not downgrade Torch to 2.6.0.

| Source | Request | Treatment |
| --- | --- | --- |
| `chatterbox-tts==0.1.7` metadata on Python 3.12 | `torch==2.6.0`, `torchaudio==2.6.0` | Install the package with `--no-deps`. Torch 2.6 publishes CUDA 11.8 / 12.4 / 12.6 wheels. Public PyTorch notes and user reports show those builds lack native `sm_120` kernels. RTX 5070 is compute capability 12.0. |
| Installed inventory and `mkvenv` command | `torch 2.11.0+cu128` | Keep. The cu128 Windows cp312 wheel exists and is 2,753,189,216 bytes. CUDA 12.8 builds list Blackwell `sm_120` for Windows. |
| Default PyPI `torch==2.11.0` | depends on `cuda-toolkit==13.0.2` and cu13 libraries | Do not use that wheel. The install index is `https://download.pytorch.org/whl/cu128`, and constraints require `2.11.0+cu128`. |
| Torch 2.11 metadata | `setuptools<82` | Pin `setuptools==81.0.0`. The installed 84.2.1 violates that requirement. |
| Torch 2.11 metadata | `sympy>=1.13.3` | Pin `sympy==1.13.3`. The historical freeze has 1.13.1, which does not satisfy it. |
| Repository `requirements.txt` and the creation script | `peft==0.17.1` | Do not downgrade. `peft==0.21.2` is the version already installed beside `transformers==5.2.0`. Both publish compatible lower bounds (`torch>=1.13`, unpinned `transformers` for that peft release). The training calls are `LoraConfig`, `get_peft_model`, `modules_to_save`, `save_pretrained`, and `PeftModel.from_pretrained`. Those calls were not imported here. |
| `chatterbox-tts==0.1.7` | `transformers==5.2.0`, `diffusers==0.29.0`, `librosa==0.11.0`, `safetensors==0.5.3`, `numpy>=1.24,<2` | Keep those pins. `diffusers==0.29.0` stays because the vendored S3Gen code imports `diffusers.models.lora.LoRACompatibleLinear`. NumPy stays `1.26.4`. |
| `chatterbox-tts==0.1.7` | `gradio==6.8.0` | Omit. Training and inference in this repository do not import Gradio. Installing Chatterbox normally would also pull Torch 2.6. |
| Historical freeze | `accelerate` absent, inventory says `1.15.0` | Pin `accelerate==1.15.0` because Trainer and the peft release require it. |
| `torchvision` | not imported by this repository | Install `0.26.0+cu128` anyway. That is the cu128 build paired with Torch 2.11.0, and the wheel is 9,585,013 bytes. |

`tokenizers==0.22.2` satisfies `transformers==5.2.0` (`tokenizers>=0.22,<=0.23`). `pandas==3.0.6` allows `numpy>=1.26` on Python 3.12. `soundfile==0.14.0` supplies the loader used by the local training patch (`read(..., dtype="float32", always_2d=True)`).

`s3tokenizer==0.3.0` declares `pre-commit` and `onnx`. Pip will install those transitive packages. Their exact versions are not locked because this session did not run pip.

## Install order

1. Upgrade pip in the new virtual environment.
2. Install `setuptools==81.0.0` before Torch.
3. Install `torch`, `torchaudio`, and `torchvision` `+cu128` from the cu128 index only.
4. Install `chatterbox-tts==0.1.7` with `--no-deps`.
5. Install `deploy/requirements-dustin.txt` under `deploy/constraints-dustin.txt` with `--upgrade-strategy only-if-needed` and the cu128 index as an extra index.

`2.11.0+cu128` matches the specifier `==2.11.0+cu128`. A plain PyPI `2.11.0` does not match that specifier. `==2.11.0` would accept either build, so the constraints use the local version.

## Download sizes that were measured

HTTP `Content-Length` only. The response bodies were not saved.

| Artifact | Bytes |
| --- | ---: |
| `torch-2.11.0+cu128-cp312-cp312-win_amd64.whl` | 2,753,189,216 |
| `torchaudio-2.11.0+cu128-cp312-cp312-win_amd64.whl` | 1,671,595 |
| `torchvision-0.26.0+cu128-cp312-cp312-win_amd64.whl` | 9,585,013 |
| Three CUDA wheels together | 2,764,445,824 |
| `ve.safetensors` | 5,695,784 |
| `t3_cfg.safetensors` | 2,129,653,744 |
| `s3gen.safetensors` | 1,056,484,620 |
| `conds.pt` | 107,374 |
| grapheme vocab stored by `setup.py` as `tokenizer.json` | 69,989 |
| Those five standard-model files | 3,192,011,511 |

3,192,011,511 bytes is 3.192 GB decimal, the same figure as the existing base-candidate inventory. The Dustin profile reads the unsuffixed `pretrained_models` directory and does not download a replacement. Turbo and Nano weights were not required for that profile; their sizes were not used. The smoke check prints local file sizes when the files are already present and does not fetch missing ones.

The unpacked environment size was not measured. The inventory's existing environment is 5,461,472,697 bytes and includes Gradio, which this manifest skips. Pip cache reuse was not tested. The installer does not pass `--no-cache-dir` and does not delete the cache. Free space was not remeasured.

The first preprocess constructs `S3Tokenizer("speech_tokenizer_v2_25hz")`. That package may download its own weights. The size was not measured, and the smoke check does not construct it. Silero VAD in `src/utils.py` still uses `torch.hub` on the first trim. Training leaves `is_inference` false, so that download is not part of the Dustin training entry.

`render_book.py` downloads `openai/whisper-large-v3-turbo` only after `--allow-asr-download`. That size was not measured.

## What was run here

- Unit tests for profile isolation, the Lewis refusal, the install plan, pin agreement, and the absence of machine-specific paths.
- `deploy/create_dustin_venv.py` with no `--apply`.
- `deploy/smoke_check.py --paths-only`.
- `setup.py` with `TTS_ALLOW_SETUP` unset, which exits before importing the download stack.
- Render help, and `render_book.py` without `--allow-asr-download`.

## What was not run

- Creating the Windows virtual environment.
- Importing Torch, PEFT, Transformers, or Chatterbox.
- `nvidia-smi`, a CUDA matmul, or any check that the installed 2.11.0+cu128 build on the machine actually executes `sm_120`.
- Training, preprocessing, inference, or a weight download.
- Executing the moved environment or reading its site-packages beyond the supplied inventory.

The Windows smoke command is the check for those GPU assumptions.

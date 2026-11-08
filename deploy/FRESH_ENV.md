# Fresh Python 3.12 environment

This procedure creates `venv-dustin-py312` next to the relocated training tree. It does not modify any other virtual environment, dataset, reference, or checkpoint. It does not download TTS weights and it does not train.

The committed `requirements.txt` pins Torch 2.6.0. That pin matches `chatterbox-tts` 0.1.7 metadata and does not contain RTX 5070 kernels. Do not use it for this GPU, and do not install the historical base-freeze file.

## What you set locally

`TTS_ROOT` is the relocated training directory. The scripts resolve every other path from that variable and from the repository location. They do not contain a machine-specific path.

```bat
set TTS_ROOT=YOUR_RELOCATED_TTS_DIRECTORY
set PYTHON312=YOUR_PYTHON_3_12_BASE_INTERPRETER
```

`PYTHON312` is optional when the `py -3.12` launcher works. Point it at a base interpreter, not at an existing virtual environment.

`TTS_MODEL_DIR` is the Dustin profile's read-only standard base-weight directory. Leave it unset only when `pretrained_models` already sits inside this checkout. A side-by-side checkout must set it to the existing unsuffixed directory and must not copy those files:

```bat
set TTS_MODEL_DIR=%TTS_ROOT%\ft\repo\pretrained_models
```

The variable is refused when it is a symlink, when it is inside `ft\data`, `ft\output`, `ft\logs`, `voice_final`, `demon`, `toadpipe`, or the pre-relocation root, or when the standard files `ve.safetensors`, `t3_cfg.safetensors`, `s3gen.safetensors`, `conds.pt`, and `tokenizer.json` are absent. `setup.py` cannot create or replace anything inside that directory, including when `TTS_ALLOW_SETUP=1`. Keep the dirty checkout at `ft\repo` and put this code in the sibling `ft\repo-dustin`. Run the launchers from `ft\repo-dustin`; they already take their repository root from their own location.

## Create and check

From `scripts\windows` inside this repository:

```bat
new_dustin_venv.cmd
smoke_dustin_env.cmd
```

Creation refuses to run when `venv-dustin-py312` already exists. The smoke check prints the Dustin, upstream, and Lewis path resolutions, compares any standard weight files already on disk with published sizes, and checks the new interpreter.

`--packages --gpu` is the remaining Windows check. It must show:

- Python 3.12
- `torch` `2.11.0+cu128` and `torch.version.cuda` beginning with `12.8`
- an architecture list containing `sm_120`
- compute capability 12.x
- a successful 64x64 bfloat16 matmul

A warning is printed when `nvidia-smi` is missing or the device name does not mention 5070. The driver version is printed and is not compared with a hard-coded minimum. CUDA 12.8 wheels need a driver that supports CUDA 12.8; this repository does not record that driver version.

## Train the Dustin experiment

Put a headerless `filename|raw_text|normalized_text` file at `dustin\data\metadata.csv` under `TTS_ROOT`, and put the wavs in `dustin\data\wavs`. One clip of about 53 seconds is an experiment, not a production audiobook dataset. `max_speech_len` stays 850 tokens, about 34 seconds at 25 tokens per second, so a single longer file is truncated.

```bat
run_dustin_train.cmd
run_dustin_train.cmd train
```

Without the `train` argument the launcher exits. The Lewis profile cannot be selected by that launcher. Logs go to `dustin\logs` under `TTS_ROOT`.

## Profiles

| Profile | How to select | What it uses |
| --- | --- | --- |
| `dustin` | default | Standard Chatterbox, LoRA rank 8, batch 1, its own `dustin` data and output |
| `upstream` | `TTS_PROFILE=upstream` | Committed Nano/Turbo settings, including hyperparameters |
| `lewis` | `TTS_PROFILE=lewis` | Preserved local hyperparameters and relocated `ft\data` / `ft\output` paths |

Training, preprocessing, rendering, and LoRA merge refuse `lewis` and refuse any path whose components contain `ft\data`, `ft\output`, `ft\logs`, `voice_final`, `demon`, or `toadpipe`.

`python setup.py` also exits until `TTS_ALLOW_SETUP=1`. Existing files are still skipped by the downloader when that variable is set. Do not point it at a folder that already holds the base weights if you only want to inspect them.

## Cloud validation

The checks that ran away from the Windows machine are in `deploy/DEPENDENCY_MANIFEST.md`. They did not install this stack, import Torch, or open a GPU.

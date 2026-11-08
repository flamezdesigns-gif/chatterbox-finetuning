"""Path profiles for the relocated training tree.

The default profile is an isolated Dustin experiment. The Lewis profile keeps
the hyperparameters and relative layout from the local training config, and
every training, preprocess, and render entry point refuses to use it.

Protected trees are detected from path components, so a relocated absolute
path is blocked even when ``TTS_ROOT`` is unset. Nothing here creates
directories, downloads weights, or starts a job.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.config import TrainConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

PROFILE_DUSTIN = "dustin"
PROFILE_LEWIS = "lewis"
PROFILE_UPSTREAM = "upstream"
DEFAULT_PROFILE = PROFILE_DUSTIN
PROFILES = (PROFILE_DUSTIN, PROFILE_LEWIS, PROFILE_UPSTREAM)

# Directory-name sequences that belong to existing voice, dataset, or
# checkpoint trees. Matching is case-insensitive and does not depend on the
# drive letter.
PROTECTED_SEQUENCES = (
    ("ft", "data"),
    ("ft", "output"),
    ("ft", "logs"),
    ("voice_final",),
    ("demon",),
    ("toadpipe",),
)

# HTTP Content-Length values observed on 2026-11-08. Bodies were not downloaded.
TORCH_CU128_WIN_CP312_BYTES = 2_753_189_216
TORCHAUDIO_CU128_WIN_CP312_BYTES = 1_671_595
TORCHVISION_CU128_WIN_CP312_BYTES = 9_585_013
STANDARD_WEIGHT_BYTES = {
    "ve.safetensors": 5_695_784,
    "t3_cfg.safetensors": 2_129_653_744,
    "s3gen.safetensors": 1_056_484_620,
    "conds.pt": 107_374,
    # setup.py stores grapheme_mtl_merged_expanded_v1.json as tokenizer.json.
    "tokenizer.json": 69_989,
}

# Distribution names as pip / importlib.metadata see them.
EXPECTED_DIST_VERSIONS = {
    "torch": "2.11.0+cu128",
    "torchaudio": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "setuptools": "81.0.0",
    "numpy": "1.26.4",
    "sympy": "1.13.3",
    "chatterbox-tts": "0.1.7",
    "peft": "0.21.2",
    "transformers": "5.2.0",
    "accelerate": "1.15.0",
    "tokenizers": "0.22.2",
    "huggingface-hub": "1.33.0",
    "safetensors": "0.5.3",
    "diffusers": "0.29.0",
    "librosa": "0.11.0",
    "soundfile": "0.14.0",
    "resemble-perth": "1.0.1",
    "conformer": "0.3.2",
    "einops": "0.8.2",
    "omegaconf": "2.4.0",
    "s3tokenizer": "0.3.0",
    "pykakasi": "2.3.0",
    "spacy-pkuseg": "1.0.1",
    "pyloudnorm": "0.2.0",
    "pandas": "3.0.6",
    "tqdm": "4.70.1",
    "tensorboard": "2.20.0",
    "requests": "2.34.2",
    "silero-vad": "6.2.0",
}

CONSTRAINT_DISTS = (
    "setuptools",
    "numpy",
    "sympy",
    "torch",
    "torchaudio",
    "torchvision",
)
# Installed with --no-deps so its torch==2.6.0 pin cannot replace the cu128 build.
NO_DEPS_DISTS = ("chatterbox-tts",)


class SafetyError(RuntimeError):
    """Raised when a run would touch a protected tree or lacks confirmation."""


@dataclass
class ResolvedRun:
    profile: str
    config: TrainConfig
    tts_root: Path | None
    data_root: Path | None
    blocked_reasons: list[str] = field(default_factory=list)


def _environ(environ):
    return os.environ if environ is None else environ


def profile_name(environ=None) -> str:
    raw = _environ(environ).get("TTS_PROFILE", DEFAULT_PROFILE).strip().lower()
    name = raw or DEFAULT_PROFILE
    if name not in PROFILES:
        raise SafetyError(
            f"Unknown TTS_PROFILE {name!r}. Use dustin, lewis, or upstream."
        )
    return name


def tts_root_from_env(environ=None) -> Path | None:
    raw = _environ(environ).get("TTS_ROOT", "").strip().strip('"')
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def normalized_parts(path) -> list[str]:
    resolved = Path(path).expanduser().resolve()
    skip = {os.sep, "/", "\\", ".", ""}
    return [part.casefold() for part in resolved.parts if part not in skip]


def protected_reason(path) -> str | None:
    if path is None or str(path).strip() == "":
        return None
    parts = normalized_parts(path)
    for sequence in PROTECTED_SEQUENCES:
        width = len(sequence)
        folded = tuple(piece.casefold() for piece in sequence)
        for index in range(0, len(parts) - width + 1):
            if tuple(parts[index:index + width]) == folded:
                label = "/".join(sequence)
                return f"protected tree {label}"
    return None


def _blocked_paths(config: TrainConfig) -> list[str]:
    reasons = []
    for label in (
        "model_dir",
        "csv_path",
        "metadata_path",
        "wav_dir",
        "preprocessed_dir",
        "output_dir",
        "inference_prompt_path",
    ):
        value = getattr(config, label, None)
        if not value:
            continue
        reason = protected_reason(value)
        if reason:
            reasons.append(f"{label}={value} ({reason})")
    return reasons


def _absolutize_existing(config: TrainConfig, repo_root: Path) -> None:
    """Resolve committed relative paths from the repo root without changing them."""
    for label in (
        "model_dir",
        "csv_path",
        "metadata_path",
        "wav_dir",
        "preprocessed_dir",
        "output_dir",
        "inference_prompt_path",
    ):
        value = getattr(config, label, None)
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            setattr(config, label, str((repo_root / path).resolve()))


def _apply_upstream(config: TrainConfig, repo_root: Path) -> None:
    # Keep the committed Nano/Turbo/Standard defaults, including hyperparameters.
    _absolutize_existing(config, repo_root)


def _apply_lewis(config: TrainConfig, tts_root: Path, repo_root: Path) -> None:
    """Hyperparameters copied from the local training config at the attached checkout.

    That checkout is older than fork main's Nano work. Paths follow the
    relocated directory layout. Callers still refuse to train, preprocess,
    or render this profile.
    """
    data = tts_root / "ft" / "data"
    config.is_turbo = False
    config.is_nano = False
    config.extend_vocab = False
    config.is_lora = True
    config.model_dir = str(repo_root / "pretrained_models")
    config.csv_path = str(data / "metadata.csv")
    config.metadata_path = str(data / "metadata.csv")
    config.wav_dir = str(data / "wavs")
    config.preprocessed_dir = str(data / "preprocess")
    config.output_dir = str(tts_root / "ft" / "output")
    config.inference_prompt_path = str(repo_root / "speaker_reference" / "2.wav")
    config.is_inference = False
    config.ljspeech = True
    config.json_format = False
    config.preprocess = True
    config.lora_r = 32
    config.lora_alpha = 64
    config.new_vocab_size = 2454
    config.batch_size = 8
    config.grad_accum = 2
    config.learning_rate = 1e-4
    config.num_epochs = 20
    config.save_steps = 100
    config.logging_steps = 5
    config.save_total_limit = 4
    config.dataloader_num_workers = 0
    config.start_text_token = 255
    config.stop_text_token = 0
    config.max_text_len = 256
    config.max_speech_len = 850
    config.prompt_duration = 3.0


def _apply_dustin(config: TrainConfig, data_root: Path, repo_root: Path) -> None:
    """Short-recording experiment. Separate data and checkpoints from Lewis.

    Rank, batch, and epoch count are smaller than the Lewis profile because
    the source clip is under a minute. ``max_speech_len`` stays 850, which
    at 25 speech tokens per second keeps about 34 seconds of a clip.
    """
    config.is_turbo = False
    config.is_nano = False
    config.extend_vocab = False
    config.is_lora = True
    config.model_dir = str(repo_root / "pretrained_models")
    config.csv_path = str(data_root / "data" / "metadata.csv")
    config.metadata_path = str(data_root / "data" / "metadata.csv")
    config.wav_dir = str(data_root / "data" / "wavs")
    config.preprocessed_dir = str(data_root / "data" / "preprocess")
    config.output_dir = str(data_root / "output")
    config.inference_prompt_path = str(data_root / "reference.wav")
    config.is_inference = False
    config.ljspeech = True
    config.json_format = False
    config.preprocess = True
    config.lora_r = 8
    config.lora_alpha = 16
    config.new_vocab_size = 2454
    config.batch_size = 1
    config.grad_accum = 1
    config.learning_rate = 1e-4
    config.num_epochs = 5
    config.save_steps = 20
    config.logging_steps = 5
    config.save_total_limit = 2
    config.dataloader_num_workers = 0
    config.start_text_token = 255
    config.stop_text_token = 0
    config.max_text_len = 256
    config.max_speech_len = 850
    config.prompt_duration = 3.0


def dustin_data_root(tts_root: Path | None, repo_root: Path = REPO_ROOT) -> Path:
    if tts_root is None:
        return repo_root / "local_workspaces" / "dustin"
    return tts_root / "dustin"


def build_resolved(environ=None, repo_root: Path = REPO_ROOT) -> ResolvedRun:
    profile = profile_name(environ)
    tts_root = tts_root_from_env(environ)
    config = TrainConfig()
    data_root = None

    if profile == PROFILE_UPSTREAM:
        _apply_upstream(config, repo_root)
    elif profile == PROFILE_LEWIS:
        if tts_root is None:
            raise SafetyError(
                "TTS_ROOT is unset, so the relocated Lewis tree cannot be resolved. "
                "Training that profile is refused either way."
            )
        _apply_lewis(config, tts_root, repo_root)
    else:
        data_root = dustin_data_root(tts_root, repo_root)
        _apply_dustin(config, data_root, repo_root)

    return ResolvedRun(
        profile=profile,
        config=config,
        tts_root=tts_root,
        data_root=data_root,
        blocked_reasons=_blocked_paths(config),
    )


def assert_safe_to_load(resolved: ResolvedRun) -> None:
    if resolved.profile == PROFILE_LEWIS:
        raise SafetyError(
            "Refusing to load the Lewis profile. Its dataset, references, "
            "and checkpoints stay untouched."
        )
    if resolved.blocked_reasons:
        raise SafetyError(
            "Refusing to use a protected path: " + "; ".join(resolved.blocked_reasons)
        )


def assert_safe_to_train(resolved: ResolvedRun, environ=None) -> None:
    assert_safe_to_load(resolved)
    if resolved.profile == PROFILE_DUSTIN and _environ(environ).get("TTS_CONFIRM_TRAIN") != "1":
        raise SafetyError(
            "Refusing to start Dustin training without TTS_CONFIRM_TRAIN=1. "
            "The Lewis job is not the default and will not be started."
        )


def assert_config_writable(config: TrainConfig) -> None:
    reasons = _blocked_paths(config)
    if reasons:
        raise SafetyError(
            "Refusing to read or write a protected training path: " + "; ".join(reasons)
        )


def assert_external_paths_allowed(paths) -> None:
    reasons = []
    for path in paths:
        if path is None or str(path).strip() == "":
            continue
        reason = protected_reason(path)
        if reason:
            reasons.append(f"{path} ({reason})")
    if reasons:
        raise SafetyError("Refusing protected path: " + "; ".join(reasons))


def assert_dataset_ready(config: TrainConfig) -> None:
    if config.ljspeech:
        csv_path = getattr(config, "csv_path", "")
        if not csv_path or not os.path.isfile(csv_path):
            raise SafetyError(
                "Metadata CSV not found at "
                f"{csv_path or '(unset)'}. Expected headerless rows of "
                "filename|raw_text|normalized_text. This check does not create files."
            )
    elif config.json_format and not os.path.isfile(config.metadata_path):
        raise SafetyError(
            "JSON metadata not found at "
            f"{config.metadata_path}. This check does not create files."
        )
    if not os.path.isdir(config.wav_dir):
        raise SafetyError(
            f"WAV directory not found at {config.wav_dir}. This check does not create it."
        )


def prepare_training_run(environ=None, repo_root: Path = REPO_ROOT) -> ResolvedRun:
    resolved = build_resolved(environ=environ, repo_root=repo_root)
    assert_safe_to_train(resolved, environ=environ)
    assert_dataset_ready(resolved.config)
    return resolved


def persistent_workers(config: TrainConfig) -> bool:
    """Persistent workers require a non-zero worker count.

    The local training script set this false together with zero workers.
    The committed script used true together with eight workers. Tie the flag
    to the configured worker count so each profile keeps its own behavior.
    """
    return int(config.dataloader_num_workers) > 0


def standard_weight_bytes_total() -> int:
    return sum(STANDARD_WEIGHT_BYTES.values())

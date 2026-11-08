"""Check isolation, and optionally the fresh Python 3.12 CUDA stack.

The default mode only resolves profiles and reports weight files that are
already on disk. It does not import Torch, download weights, or train.

``--packages`` checks installed distributions.
``--gpu`` also requires a CUDA device of compute capability 12.x and runs one
small bfloat16 matmul. That is the RTX 5070 check still required on the
Windows machine.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.deployment import (
    EXPECTED_DIST_VERSIONS,
    PROFILE_DUSTIN,
    PROFILE_LEWIS,
    PROFILE_UPSTREAM,
    STANDARD_WEIGHT_BYTES,
    TORCHAUDIO_CU128_WIN_CP312_BYTES,
    TORCHVISION_CU128_WIN_CP312_BYTES,
    TORCH_CU128_WIN_CP312_BYTES,
    SafetyError,
    build_resolved,
    standard_weight_bytes_total,
)


def _format_bytes(size: int) -> str:
    return f"{size:,} bytes ({size / (1024 ** 3):.2f} GiB)"


def profile_report(environ) -> list[str]:
    lines = []
    for name in (PROFILE_DUSTIN, PROFILE_UPSTREAM, PROFILE_LEWIS):
        probe = dict(environ)
        probe["TTS_PROFILE"] = name
        probe.pop("TTS_CONFIRM_TRAIN", None)
        try:
            resolved = build_resolved(environ=probe)
        except SafetyError as exc:
            lines.append(f"profile {name}: blocked ({exc})")
            continue
        config = resolved.config
        metadata = config.csv_path if getattr(config, "ljspeech", False) else config.metadata_path
        lines.append(
            "profile {name}: turbo={turbo} nano={nano} model={model} model_override={override} "
            "output={output} metadata={metadata} workers={workers} batch={batch} "
            "epochs={epochs} lora_r={rank} logging_steps={logging} blocked={blocked}".format(
                name=name,
                turbo=config.is_turbo,
                nano=config.is_nano,
                model=config.model_dir,
                override=resolved.model_dir_override,
                output=config.output_dir,
                metadata=metadata,
                workers=config.dataloader_num_workers,
                batch=config.batch_size,
                epochs=config.num_epochs,
                rank=config.lora_r,
                logging=config.logging_steps,
                blocked=resolved.blocked_reasons or "no",
            )
        )
    return lines


def weight_report(model_dir: Path) -> list[str]:
    lines = [f"pretrained directory: {model_dir}"]
    present = 0
    for name, expected in STANDARD_WEIGHT_BYTES.items():
        path = model_dir / name
        if not path.is_file():
            lines.append(f"  missing {name}; published size {_format_bytes(expected)}")
            continue
        actual = path.stat().st_size
        present += actual
        relation = "matches published size" if actual == expected else (
            f"size differs from published {_format_bytes(expected)}; left untouched"
        )
        lines.append(f"  present {name}: {_format_bytes(actual)}; {relation}")
    lines.append(
        "published standard-weight set if absent: "
        + _format_bytes(standard_weight_bytes_total())
    )
    lines.append(f"bytes currently present from that set: {present:,}")
    return lines


def package_problems() -> list[str]:
    problems = []
    if sys.version_info[:2] != (3, 12):
        problems.append(
            f"Python {sys.version_info[0]}.{sys.version_info[1]} is running; expected 3.12."
        )
    for dist_name, expected in EXPECTED_DIST_VERSIONS.items():
        try:
            installed = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"missing distribution {dist_name}=={expected}")
            continue
        if installed != expected:
            problems.append(f"{dist_name}=={installed}, expected {expected}")
    return problems


def nvidia_smi_line() -> str:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "INFO nvidia-smi not on PATH"
    try:
        completed = subprocess_run_smi(exe)
    except Exception as exc:  # pragma: no cover - host tool
        return f"INFO nvidia-smi failed: {exc}"
    text = (completed.stdout or completed.stderr).strip()
    return "INFO nvidia-smi " + text


def subprocess_run_smi(exe: str):
    import subprocess

    return subprocess.run(
        [exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def gpu_problems() -> list[str]:
    problems = [nvidia_smi_line()]
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on the fresh env
        return problems + [f"torch import failed: {exc}"]
    version = getattr(torch, "__version__", "")
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    problems.append(f"INFO torch={version} torch.version.cuda={cuda_version}")
    if not str(version).startswith("2.11.0+cu128"):
        problems.append(f"torch version {version} is not 2.11.0+cu128")
    if cuda_version is None or not str(cuda_version).startswith("12.8"):
        problems.append(f"torch CUDA version {cuda_version} is not 12.8")
    arch = []
    get_arch = getattr(torch.cuda, "get_arch_list", None)
    if get_arch is not None:
        try:
            arch = list(get_arch())
        except Exception as exc:
            problems.append(f"could not read torch arch list: {exc}")
    problems.append(f"INFO arch_list={arch}")
    if arch and not any(item.replace("sm_", "").startswith("120") or item == "sm_120" for item in arch):
        problems.append(f"cu128 wheel arch list has no sm_120: {arch}")
    if not torch.cuda.is_available():
        problems.append("torch.cuda.is_available() is false")
        return problems
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    problems.append(f"INFO device={device_name} capability={capability}")
    if "5070" not in device_name.replace(" ", ""):
        problems.append(f"WARNING device name does not mention 5070: {device_name}")
    if capability[0] < 12:
        problems.append(f"compute capability {capability} is below 12.x")
    bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    problems.append(f"INFO bf16_supported={bf16}")
    try:
        left = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        product = left @ left
        torch.cuda.synchronize()
        if not torch.isfinite(product.float()).all():
            problems.append("bfloat16 matmul produced a non-finite value")
        else:
            problems.append("INFO bfloat16 matmul finished")
    except Exception as exc:
        problems.append(f"bfloat16 matmul failed: {exc}")
    return problems


def disk_lines() -> list[str]:
    usage = shutil.disk_usage(Path.cwd())
    return [
        f"disk free at cwd: {_format_bytes(usage.free)}",
        "download estimate, Windows cp312, cache miss, wheels only: "
        + _format_bytes(
            TORCH_CU128_WIN_CP312_BYTES
            + TORCHAUDIO_CU128_WIN_CP312_BYTES
            + TORCHVISION_CU128_WIN_CP312_BYTES
        ),
        "torch wheel: " + _format_bytes(TORCH_CU128_WIN_CP312_BYTES),
        "torchaudio wheel: " + _format_bytes(TORCHAUDIO_CU128_WIN_CP312_BYTES),
        "torchvision wheel: " + _format_bytes(TORCHVISION_CU128_WIN_CP312_BYTES),
        "Installed size of the rest of the env was not measured here. "
        "A comparable existing environment was about 5.09 GiB and included Gradio, "
        "which this manifest does not install.",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--paths-only",
        action="store_true",
        help="Resolve profiles and weights only. This is the default.",
    )
    args = parser.parse_args(argv)
    exit_code = 0
    print(f"python={sys.version.split()[0]} prefix={sys.prefix}")
    print(f"TTS_ROOT={os.environ.get('TTS_ROOT', '')!r} TTS_PROFILE={os.environ.get('TTS_PROFILE', '')!r}")
    try:
        for line in profile_report(os.environ):
            print(line)
        current = build_resolved()
        for line in weight_report(Path(current.config.model_dir)):
            print(line)
    except SafetyError as exc:
        print(f"path check failed: {exc}")
        return 2
    for line in disk_lines():
        print(line)
    if args.packages or args.gpu:
        problems = package_problems()
        if args.gpu:
            problems.extend(gpu_problems())
        for item in problems:
            print(item)
            if not item.startswith("INFO ") and not item.startswith("WARNING "):
                exit_code = 2
            elif item.startswith("WARNING "):
                exit_code = max(exit_code, 1)
    print("Smoke check did not train, did not import a TTS checkpoint, and did not download models.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

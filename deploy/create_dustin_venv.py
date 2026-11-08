"""Print or apply the fresh Dustin environment install. Default is dry-run.

``--apply`` installs only when the running interpreter already lives in a
virtual environment named ``venv-dustin-py312``. It will not create, delete,
or reuse any other environment.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CU128_INDEX = "https://download.pytorch.org/whl/cu128"
VENV_NAME = "venv-dustin-py312"
UNTOUCHED_VENV_NAME = "venv-cbft"


def plan_commands(python_exe: str, repo_root: Path = REPO_ROOT) -> list[list[str]]:
    requirements = repo_root / "deploy" / "requirements-dustin.txt"
    constraints = repo_root / "deploy" / "constraints-dustin.txt"
    return [
        [python_exe, "-m", "pip", "install", "--upgrade", "pip"],
        [python_exe, "-m", "pip", "install", "setuptools==81.0.0"],
        [
            python_exe,
            "-m",
            "pip",
            "install",
            "torch==2.11.0+cu128",
            "torchaudio==2.11.0+cu128",
            "torchvision==0.26.0+cu128",
            "--index-url",
            CU128_INDEX,
        ],
        [python_exe, "-m", "pip", "install", "--no-deps", "chatterbox-tts==0.1.7"],
        [
            python_exe,
            "-m",
            "pip",
            "install",
            "--upgrade-strategy",
            "only-if-needed",
            "--constraint",
            str(constraints),
            "--requirement",
            str(requirements),
            "--extra-index-url",
            CU128_INDEX,
        ],
    ]


def _interpreter_is_dustin_venv(prefix: Path) -> str | None:
    if prefix.name != VENV_NAME:
        return (
            f"Refusing --apply: interpreter prefix {prefix} is not a "
            f"{VENV_NAME} environment."
        )
    if UNTOUCHED_VENV_NAME in {part.lower() for part in prefix.parts}:
        return f"Refusing --apply: interpreter is inside {UNTOUCHED_VENV_NAME}."
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Install into the current venv-dustin-py312 interpreter.",
    )
    args = parser.parse_args(argv)
    python_exe = sys.executable
    commands = plan_commands(python_exe)
    for command in commands:
        print(" ".join(command))
    if not args.apply:
        print("No packages were installed. Creation stays a dry-run without --apply.")
        return 0

    refusal = _interpreter_is_dustin_venv(Path(sys.prefix))
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    for command in commands:
        subprocess.check_call(command)
    print("Package install finished. No training job and no model download was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

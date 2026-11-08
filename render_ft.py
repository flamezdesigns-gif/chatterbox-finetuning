"""Render text with the active profile's LoRA adapter.

Paths under an existing voice, dataset, or checkpoint tree are refused.
No reference or checkpoint path is built in.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref", help="Reference wav for this render.")
    parser.add_argument("text", help="UTF-8 text file.")
    parser.add_argument("out", help="Output path prefix. .wav and .mp3 are appended.")
    parser.add_argument("checkpoint", nargs="?", help="Optional T3 safetensors to load after the adapter.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    from src.deployment import SafetyError, assert_external_paths_allowed

    candidates = [args.ref, args.text, args.out, args.out + ".wav", args.out + ".mp3", args.checkpoint]
    try:
        assert_external_paths_allowed(candidates)
    except SafetyError as exc:
        print(exc, file=sys.stderr)
        return 2

    from src.runtime_patches import apply_training_patches

    apply_training_patches()
    import inference as I

    engine = I.load_finetuned_engine_lora("cuda" if __import__("torch").cuda.is_available() else "cpu")
    if args.checkpoint:
        from safetensors.torch import load_file

        state = {key[3:]: value for key, value in load_file(args.checkpoint).items() if key.startswith("t3.")}
        result = engine.t3.load_state_dict(state, strict=False)
        print("missing", len(result.missing_keys), "unexpected", len(result.unexpected_keys))
    for module in engine.t3.modules():
        module_config = getattr(module, "config", None)
        if module_config is not None and hasattr(module_config, "_attn_implementation"):
            module_config._attn_implementation = "eager"

    text = " ".join(Path(args.text).read_text(encoding="utf-8").split())
    sentences = re.split(r"(?<=[.!?;])\s+", text)
    chunks = []
    current = ""
    limit = int(os.environ.get("CHUNK", "260"))
    for sentence in sentences:
        if len(current) + len(sentence) < limit:
            current = (current + " " + sentence).strip()
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)

    if os.environ.get("NOALIGN"):
        from src.runtime_patches import apply_noalign

        print(apply_noalign(engine))

    import numpy as np
    import soundfile as sf

    I.set_seed(42)
    parts = []
    for chunk in chunks:
        wav = engine.generate(chunk, audio_prompt_path=args.ref, exaggeration=0.4, cfg_weight=0.5)
        if isinstance(wav, tuple):
            wav = wav[0]
        parts += [wav.squeeze().cpu().numpy(), np.zeros(int(engine.sr * 0.35), np.float32)]
    audio = np.concatenate(parts)
    sf.write(args.out + ".wav", audio, engine.sr)
    print("dur", len(audio) / engine.sr)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", args.out + ".wav", "-af", "loudnorm=I=-18:TP=-1.5", "-b:a", "128k", args.out + ".mp3"],
        check=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

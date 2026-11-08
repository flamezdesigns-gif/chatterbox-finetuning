"""Pick low-WER takes for each chunk, then write wav and mp3.

The speech-recognition model is downloaded only with --allow-asr-download.
Reference, text, checkpoint, and output paths are required and cannot point
at an existing voice, dataset, or checkpoint tree.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="UTF-8 text file.")
    parser.add_argument("--out", required=True, help="Output path prefix.")
    parser.add_argument("--ref", required=True, help="Reference wav for this render.")
    parser.add_argument("--checkpoint", required=True, help="T3 safetensors checkpoint.")
    parser.add_argument(
        "--allow-asr-download",
        action="store_true",
        help="Allow the first run to download openai/whisper-large-v3-turbo.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not args.allow_asr_download:
        print(
            "Refusing to start. This renderer downloads openai/whisper-large-v3-turbo "
            "unless that model is already cached. Re-run with --allow-asr-download to permit it.",
            file=sys.stderr,
        )
        return 2

    from src.deployment import SafetyError, assert_external_paths_allowed

    candidates = [
        args.text,
        args.out,
        args.out + ".wav",
        args.out + ".mp3",
        args.out + "_log.json",
        args.ref,
        args.checkpoint,
    ]
    try:
        assert_external_paths_allowed(candidates)
    except SafetyError as exc:
        print(exc, file=sys.stderr)
        return 2

    import difflib
    import json
    import os
    import re
    import subprocess
    import time

    import numpy as np
    import soundfile as sf
    import torch
    from safetensors.torch import load_file
    from transformers import pipeline

    from src.runtime_patches import apply_training_patches

    apply_training_patches()
    import inference as I
    from src.runtime_patches import apply_noalign

    engine = I.load_finetuned_engine_lora("cuda")
    print(apply_noalign(engine))
    for module in engine.t3.modules():
        module_config = getattr(module, "config", None)
        if module_config is not None and hasattr(module_config, "_attn_implementation"):
            module_config._attn_implementation = "eager"
    engine.t3.load_state_dict(
        {key[3:]: value for key, value in load_file(args.checkpoint).items() if key.startswith("t3.")},
        strict=False,
    )
    recognizer = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        device="cuda",
        torch_dtype=torch.float16,
    )

    def normalize(text):
        return re.sub(r"[^a-z' ]", " ", text.lower().replace("-", " ")).split()

    def word_error(reference, hypothesis):
        return 1 - difflib.SequenceMatcher(None, normalize(reference), normalize(hypothesis)).ratio()

    paragraphs = [line.strip() for line in Path(args.text).read_text(encoding="utf-8").splitlines() if line.strip()]
    parts = []
    log = []
    generate_seconds = 0.0
    sample_rate = engine.sr
    tolerance = float(os.environ.get("WERTOL", "0.08"))
    for paragraph_index, paragraph in enumerate(paragraphs):
        sentences = re.split(r"(?<=[.!?;])\s+", paragraph)
        chunks = []
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) > 200:
                chunks.append(current)
                current = sentence
            else:
                current = (current + " " + sentence).strip()
        if current:
            chunks.append(current)
        for chunk_index, chunk in enumerate(chunks):
            best = None
            for attempt in range(4):
                I.set_seed(42 + attempt * 101)
                started = time.time()
                wav = engine.generate(chunk, audio_prompt_path=args.ref, exaggeration=0.4, cfg_weight=0.5)
                if isinstance(wav, tuple):
                    wav = wav[0]
                torch.cuda.synchronize()
                generate_seconds += time.time() - started
                samples = wav.squeeze().cpu().numpy()
                hypothesis = recognizer(
                    {"raw": samples.astype(np.float32), "sampling_rate": sample_rate},
                    generate_kwargs={"language": "en"},
                )["text"]
                error = word_error(chunk, hypothesis)
                log.append({
                    "p": paragraph_index,
                    "c": chunk_index,
                    "try_": attempt,
                    "wer": round(error, 3),
                    "dur": round(len(samples) / sample_rate, 1),
                    "text": chunk,
                    "hyp": hypothesis.strip(),
                })
                print(paragraph_index, chunk_index, attempt, round(error, 3), flush=True)
                if best is None or error < best[0]:
                    best = (error, samples)
                if error <= tolerance:
                    break
            pause = 0.9 if chunk_index == len(chunks) - 1 else 0.4
            parts += [best[1], np.zeros(int(sample_rate * pause), np.float32)]

    audio = np.concatenate(parts)
    sf.write(args.out + ".wav", audio, sample_rate)
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", args.out + ".wav",
            "-af", "loudnorm=I=-18:TP=-1.5:LRA=11", "-b:a", "160k", args.out + ".mp3",
        ],
        check=False,
    )
    Path(args.out + "_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(json.dumps({
        "audio_s": round(len(audio) / sample_rate, 1),
        "gen_s": round(generate_seconds, 1),
        "rtf": round(generate_seconds / (len(audio) / sample_rate), 3),
        "peak_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

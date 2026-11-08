"""Audio-loader and watermarker fallbacks used by the local training script.

These run before Chatterbox builds a watermarker and before preprocessing
calls ``torchaudio.load``. TorchAudio 2.9+ can fail in ``load`` when its
new decoder backend is absent; the local script reads waveforms with
soundfile instead. Some Perth builds ship ``PerthImplicitWatermarker`` as
``None`` and need the dummy watermarker before model construction.
"""


def apply_torchaudio_soundfile_loader():
    import torch
    import torchaudio
    import soundfile as sf

    def _load_with_soundfile(path, *args, **kwargs):
        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T.copy())
        return waveform, sample_rate

    torchaudio.load = _load_with_soundfile


def apply_perth_fallback():
    import perth

    if perth.PerthImplicitWatermarker is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker


def apply_training_patches():
    apply_torchaudio_soundfile_loader()
    apply_perth_fallback()


_ANALYZER_MODULE = "src.chatterbox_.models.t3.inference.alignment_stream_analyzer"


def _identity_step(self, logits, next_token=None):
    return logits


def _model_for_inference(model):
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            pass
    base = getattr(model, "base_model", None)
    inner = getattr(base, "model", None) if base is not None else None
    return inner if inner is not None else model


def _inference_source(model) -> str:
    import inspect

    chunks = []
    for name in ("inference", "inference_turbo"):
        fn = getattr(model, name, None)
        if fn is None:
            continue
        try:
            chunks.append(inspect.getsource(fn))
        except (OSError, TypeError) as exc:
            raise RuntimeError(
                f"NOALIGN cannot read {type(model).__name__}.{name} to verify alignment behavior."
            ) from exc
    if not chunks:
        raise RuntimeError("NOALIGN cannot find T3.inference on the loaded model.")
    return "\n".join(chunks)


def apply_noalign(engine) -> str:
    """Keep sampled speech logits free of alignment-stream edits.

    Older renderers set ``AlignmentStreamAnalyzer.step`` to return its logits
    unchanged. This tree's ``T3.inference`` samples the CFG logits directly and
    does not construct that analyzer. When the class is importable, ``step``
    is still replaced. When sampling code still names the missing class, this
    raises instead of ignoring the request.
    """
    import importlib

    model = _model_for_inference(engine.t3)
    source = _inference_source(model)
    mentions_analyzer = (
        "AlignmentStreamAnalyzer" in source or "alignment_stream_analyzer" in source
    )
    try:
        module = importlib.import_module(_ANALYZER_MODULE)
    except ModuleNotFoundError:
        module = None
    except ImportError as exc:
        raise RuntimeError(f"NOALIGN could not import {_ANALYZER_MODULE}: {exc}") from exc

    if module is not None:
        cls = getattr(module, "AlignmentStreamAnalyzer", None)
        if cls is None or not hasattr(cls, "step"):
            raise RuntimeError(
                f"{_ANALYZER_MODULE} has no AlignmentStreamAnalyzer.step to disable."
            )
        cls.step = _identity_step
        for obj in (engine.t3, model):
            instance = getattr(obj, "alignment_stream_analyzer", None)
            if instance is not None:
                instance.step = lambda logits, next_token=None: logits
        return "patched AlignmentStreamAnalyzer.step to return logits unchanged"

    for obj in (engine.t3, model):
        instance = getattr(obj, "alignment_stream_analyzer", None)
        if instance is not None and hasattr(instance, "step"):
            instance.step = lambda logits, next_token=None: logits
            return "patched the loaded alignment_stream_analyzer.step to return logits unchanged"

    if mentions_analyzer:
        raise RuntimeError(
            "NOALIGN was requested, but T3 sampling still references "
            f"AlignmentStreamAnalyzer and {_ANALYZER_MODULE} is not importable."
        )
    return (
        "NOALIGN: T3.inference does not apply an alignment stream. "
        "Speech logits stay as sampled by the current CFG loop."
    )

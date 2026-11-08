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

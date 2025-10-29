
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import math

# =============================================================================
# AudioLoader
# =============================================================================

@dataclass
class AudioLoaderConfig:
    """
    Author: María Belén
    What: Configuration for AudioLoader; enforces output sample rate, duration and normalization.
    Params:
        sampleRate (int): Target sample rate in Hz.
        durationSec (float): Target duration in seconds.
        mono (bool): If True, convert to mono.
        normalize (str): 'peak' | 'rms' | 'none' normalization strategy.
        rmsTargetDbfs (float): RMS target in dBFS (only if normalize='rms').
        padSide (str): 'both' | 'end' | 'start' for zero padding.
        crop (str): 'center' | 'start' | 'random' crop strategy when longer.
    Returns: N/A (data container).
    """
    sampleRate: int = 16000
    durationSec: float = 10.0
    mono: bool = True
    normalize: str = "peak"
    rmsTargetDbfs: float = -20.0
    padSide: str = "end"
    crop: str = "center"


def safeRms(x: np.ndarray) -> float:
    """
    Author: María Belén
    What: Compute numerically stable RMS.
    Params: x (np.ndarray) – audio waveform.
    Returns: float – RMS value.
    """
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12))


def dbfsToLinear(dbfs: float) -> float:
    """
    Author: María Belén
    What: Convert dBFS value to linear amplitude scale (0 dBFS -> 1.0).
    Params: dbfs (float) – level in dBFS.
    Returns: float – linear gain.
    """
    return 10.0 ** (dbfs / 20.0)


def toMono(x: np.ndarray) -> np.ndarray:
    """
    Author: María Belén
    What: Convert multi-channel audio to mono by averaging channels.
    Params: x (np.ndarray) – waveform of shape [n] or [n, ch].
    Returns: np.ndarray – mono waveform [n].
    """
    if x.ndim == 2:
        return np.mean(x, axis=1, dtype=np.float32)
    return x.astype(np.float32, copy=False)


def resampleAudio(x: np.ndarray, srcSr: int, dstSr: int) -> np.ndarray:
    """
    Author: María Belén
    What: High-quality polyphase resampling from srcSr to dstSr.
    Params:
        x (np.ndarray) – mono waveform.
        srcSr (int) – original sample rate.
        dstSr (int) – desired sample rate.
    Returns: np.ndarray – resampled waveform.
    """
    if srcSr == dstSr:
        return x.astype(np.float32, copy=False)
    g = math.gcd(srcSr, dstSr)
    up, down = dstSr // g, srcSr // g
    y = resample_poly(x, up, down).astype(np.float32, copy=False)
    return y


def normalizePeak(x: np.ndarray) -> np.ndarray:
    """
    Author: María Belén
    What: Peak normalize to max |x| = 0.99.
    Params: x (np.ndarray) – waveform.
    Returns: np.ndarray – normalized waveform.
    """
    peak = float(np.max(np.abs(x)))
    if peak < 1e-9:
        return x
    return (0.99 * x / peak).astype(np.float32, copy=False)


def normalizeRms(x: np.ndarray, targetDbfs: float) -> np.ndarray:
    """
    Author: María Belén
    What: RMS normalize to a target level in dBFS, clipping to [-1,1].
    Params:
        x (np.ndarray) – waveform.
        targetDbfs (float) – target RMS in dBFS.
    Returns: np.ndarray – normalized waveform.
    """
    target = dbfsToLinear(targetDbfs)
    cur = safeRms(x)
    if cur < 1e-9:
        return x
    y = (target / cur) * x
    y = np.clip(y, -1.0, 1.0, out=y)
    return y.astype(np.float32, copy=False)


def padOrCrop(x: np.ndarray, nTarget: int, padSide: str, crop: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Author: María Belén
    What: Enforce fixed-length by padding zeros or cropping the signal.
    Params:
        x (np.ndarray) – waveform.
        nTarget (int) – desired number of samples.
        padSide (str) – 'both'|'end'|'start' padding strategy.
        crop (str) – 'center'|'start'|'random' crop strategy.
    Returns:
        y (np.ndarray) – length-fixed waveform.
        mask (np.ndarray) – 1 where real audio, 0 where padding.
    """
    n = x.shape[0]
    mask = np.ones(nTarget, dtype=np.float32)
    if n == nTarget:
        return x.astype(np.float32, copy=False), mask

    if n > nTarget:
        if crop == "start":
            start = 0
        elif crop == "random":
            start = np.random.randint(0, n - nTarget + 1)
        else:  # center
            start = (n - nTarget) // 2
        x = x[start:start + nTarget]
        return x.astype(np.float32, copy=False), mask

    y = np.zeros(nTarget, dtype=np.float32)
    if padSide == "both":
        totalPad = nTarget - n
        left = totalPad // 2
        right = totalPad - left
        y[left:left + n] = x
        mask[:left] = 0.0
        mask[left + n:] = 0.0
    elif padSide == "start":
        y[(nTarget - n):] = x
        mask[:(nTarget - n)] = 0.0
    else:  # "end"
        y[:n] = x
        mask[n:] = 0.0
    return y, mask


class AudioLoader:
    """
    Author: María Belén
    What: Read audio, convert to mono float32 at a fixed sample rate and duration, and return a padding mask.
    Params:
        cfg (AudioLoaderConfig): configuration dataclass. If None, defaults are used.
    Returns:
        load(path) -> (wave (np.ndarray [N]), meta (dict with 'mask' and fields)).
    """

    def __init__(self, cfg: Optional[AudioLoaderConfig] = None):
        self.cfg = cfg or AudioLoaderConfig()
        self.nTarget = int(self.cfg.sampleRate * self.cfg.durationSec)

    def load(self, path: str, *, crop: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
        """
        Author: María Belén
        What: Load one audio file and normalize/pad it according to config.
        Params:
            path (str): path to audio file.
            crop (Optional[str]): override crop mode for this call.
        Returns:
            y (np.ndarray): waveform [N] float32 in [-1,1].
            meta (dict): metadata including 'mask' [N].
        """
        x, sr = sf.read(str(path), always_2d=False, dtype="float32")
        x = np.asarray(x, dtype=np.float32)

        if self.cfg.mono:
            x = toMono(x)

        x = np.nan_to_num(x, copy=False)
        x = resampleAudio(x, sr, self.cfg.sampleRate)

        if self.cfg.normalize == "peak":
            x = normalizePeak(x)
        elif self.cfg.normalize == "rms":
            x = normalizeRms(x, self.cfg.rmsTargetDbfs)

        cropMode = crop or self.cfg.crop
        y, mask = padOrCrop(x, self.nTarget, self.cfg.padSide, cropMode)

        meta = {
            "path": str(path),
            "origSr": int(sr),
            "targetSr": int(self.cfg.sampleRate),
            "origLen": int(x.shape[0]),
            "targetLen": int(self.nTarget),
            "durationSec": float(self.cfg.durationSec),
            "normalize": self.cfg.normalize,
            "rmsTargetDbfs": float(self.cfg.rmsTargetDbfs) if self.cfg.normalize == "rms" else None,
            "padSide": self.cfg.padSide,
            "cropMode": cropMode,
            "mask": mask.astype(np.float32),
        }
        return y, meta


def loadFolder(folder: str, cfg: Optional[AudioLoaderConfig] = None) -> Iterable[Tuple[np.ndarray, Dict]]:
    """
    Author: María Belén
    What: Iterate over .wav files in a folder yielding (wave, meta).
    Params:
        folder (str): root folder to search recursively.
        cfg (Optional[AudioLoaderConfig]): configuration for loader.
    Returns: Iterable[Tuple[np.ndarray, Dict]] with (wave, meta) per file.
    """
    loader = AudioLoader(cfg)
    for p in sorted(Path(folder).rglob("*.wav")):
        yield loader.load(str(p))

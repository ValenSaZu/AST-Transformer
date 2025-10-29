
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
from scipy.signal import stft

# =============================================================================
# MelSpectrogramGenerator (Styled Version)
# =============================================================================

def hzToMel(hz: np.ndarray) -> np.ndarray:
    """
    Author: María Belén
    What: Convert Hz to Mel scale (HTK formula).
    Params: hz (np.ndarray)
    Returns: np.ndarray – mels.
    """
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def melToHz(mel: np.ndarray) -> np.ndarray:
    """
    Author: María Belén
    What: Convert Mel to Hz (HTK inverse).
    Params: mel (np.ndarray)
    Returns: np.ndarray – Hz.
    """
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def triangularMelFilterbank(sr: int, nFft: int, nMels: int, fMin: float, fMax: Optional[float]) -> np.ndarray:
    """
    Author: María Belén
    What: Build triangular Mel filterbank (Slaney-style normalization).
    Params:
        sr (int): sample rate.
        nFft (int): FFT size.
        nMels (int): number of mel bands.
        fMin (float): minimum frequency.
        fMax (Optional[float]): maximum frequency (defaults to Nyquist).
    Returns: np.ndarray – filterbank [nMels, nFft//2+1].
    """
    if fMax is None:
        fMax = sr / 2.0
    nFreqs = nFft // 2 + 1

    melMin = hzToMel(np.array([fMin]))[0]
    melMax = hzToMel(np.array([fMax]))[0]
    melPoints = np.linspace(melMin, melMax, nMels + 2, dtype=np.float64)
    hzPoints = melToHz(melPoints)
    binPoints = np.floor((nFft + 1) * hzPoints / sr).astype(int)

    fbanks = np.zeros((nMels, nFreqs), dtype=np.float64)
    for m in range(1, nMels + 1):
        fLeft = binPoints[m - 1]
        fCenter = binPoints[m]
        fRight = binPoints[m + 1]

        if fLeft == fCenter:
            fCenter = min(fCenter + 1, nFreqs - 1)
        if fCenter == fRight:
            fRight = min(fRight + 1, nFreqs - 1)

        if fLeft < fCenter:
            fbanks[m - 1, fLeft:fCenter] = (np.arange(fLeft, fCenter) - fLeft) / (fCenter - fLeft)
        if fCenter < fRight:
            fbanks[m - 1, fCenter:fRight] = (fRight - np.arange(fCenter, fRight)) / (fRight - fCenter)

    enorm = 2.0 / (hzPoints[2:nMels + 2] - hzPoints[:nMels])
    fbanks *= enorm[:, np.newaxis]
    return fbanks.astype(np.float32)

def powerToDb(S: np.ndarray, topDb: float = 80.0) -> np.ndarray:
    """
    Author: María Belén
    What: Convert power spectrogram to decibels (max-referenced).
    Params:
        S (np.ndarray): power spectrogram.
        topDb (float): dynamic range limit.
    Returns: np.ndarray – dB-scaled spectrogram.
    """
    S = np.maximum(S, 1e-10)
    logS = 10.0 * np.log10(S)
    logS -= np.max(logS)
    if topDb is not None:
        logS = np.maximum(logS, logS.max() - topDb)
    return logS.astype(np.float32)

def ceilDiv(a: int, b: int) -> int:
    """
    Author: María Belén
    What: Ceiling integer division.
    Params: a (int), b (int)
    Returns: int – ceil(a/b).
    """
    return (a + b - 1) // b

@dataclass
class MelSpecConfig:
    """
    Author: María Belén
    What: Configuration for MelSpectrogramGenerator and patchification.
    Params:
        sampleRate (int): sample rate in Hz.
        nFft (int): FFT size.
        winLength (int): window length in samples.
        hopLength (int): hop length in samples.
        nMels (int): number of mel bands.
        fMin (float): minimum frequency.
        fMax (Optional[float]): maximum frequency (Nyquist by default).
        center (bool): if True, pad with zeros in STFT like librosa.
        window (str): scipy window name ('hann').
        topDb (float): dynamic range clamp for dB scale.
        padTimeToMultiple (int): pad time frames to a multiple (e.g., 16).
        targetNumFrames (Optional[int]): exact number of frames to pad to.
    Returns: N/A (data container).
    """
    sampleRate: int = 16000
    nFft: int = 512
    winLength: int = 400
    hopLength: int = 160
    nMels: int = 128
    fMin: float = 0.0
    fMax: Optional[float] = 8000.0
    center: bool = False
    window: str = "hann"
    topDb: float = 80.0
    padTimeToMultiple: int = 16
    targetNumFrames: Optional[int] = None

class MelSpectrogramGenerator:
    """
    Author: María Belén
    What: Generate Mel spectrograms in dB and split them into fixed-size patches for AST-style transformers.
    Params:
        cfg (MelSpecConfig): configuration dataclass.
    Returns:
        melSpectrogram(...) -> (M_db [F,T], frameMask [T], info dict)
        patchify(...) -> (patches [N,pF,pT], patchMask [N], info dict)
    """

    def __init__(self, cfg: Optional[MelSpecConfig] = None):
        self.cfg = cfg or MelSpecConfig()
        self.melFilter = triangularMelFilterbank(
            self.cfg.sampleRate, self.cfg.nFft, self.cfg.nMels, self.cfg.fMin, self.cfg.fMax
        )

    def stftPower(self, y: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Author: María Belén
        What: Compute one-sided STFT power spectrogram [freq, time].
        Params: y (np.ndarray) – mono waveform.
        Returns: (S (np.ndarray), nFrames (int))
        """
        noverlap = self.cfg.winLength - self.cfg.hopLength
        boundary = None if not self.cfg.center else "zeros"
        _, _, Zxx = stft(
            y,
            fs=self.cfg.sampleRate,
            window=self.cfg.window,
            nperseg=self.cfg.winLength,
            noverlap=noverlap,
            nfft=self.cfg.nFft,
            boundary=boundary,
            padded=False,
            return_onesided=True,
        )
        S = (np.abs(Zxx) ** 2).astype(np.float32)
        return S, S.shape[1]

    def framesFromMask(self, sampleMask: np.ndarray) -> int:
        """
        Author: María Belén
        What: Estimate how many STFT frames contain real audio using the sample mask.
        Params: sampleMask (np.ndarray) – mask over samples [N].
        Returns: int – number of valid frames.
        """
        nReal = int(np.sum(sampleMask).item())
        if nReal <= 0:
            return 0
        if nReal < self.cfg.winLength:
            return 1
        return 1 + (nReal - self.cfg.winLength) // self.cfg.hopLength

    def padTime(self, X: np.ndarray, frameMask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Author: María Belén
        What: Pad time axis to targetNumFrames or to a multiple of padTimeToMultiple.
        Params: X (np.ndarray) – spectrogram [F,T]; frameMask (np.ndarray) – [T].
        Returns: (Xpad, frameMaskPad)
        """
        _, T = X.shape
        if self.cfg.targetNumFrames is not None:
            tTarget = self.cfg.targetNumFrames
        else:
            mult = self.cfg.padTimeToMultiple
            tTarget = ceilDiv(T, mult) * mult if mult > 0 else T
        if tTarget == T:
            return X, frameMask
        pad = tTarget - T
        Xpad = np.pad(X, ((0,0),(0,pad)), mode="constant")
        fpad = np.pad(frameMask, (0,pad), mode="constant")
        return Xpad, fpad

    def melSpectrogram(self, y: np.ndarray, sampleMask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Author: María Belén
        What: Compute Mel spectrogram in dB with optional frame mask generation and time padding.
        Params:
            y (np.ndarray): mono waveform [N] in [-1,1].
            sampleMask (Optional[np.ndarray]): sample-level mask [N].
        Returns:
            M_db (np.ndarray): [nMels, Tpad]
            frameMask (np.ndarray): [Tpad], 1.0 where frames contain real audio
            info (dict): shapes and parameters
        """
        assert y.ndim == 1, "waveform must be mono [N]"
        S, nFrames = self.stftPower(y)

        M = np.dot(self.melFilter, S)  # [nMels, T]
        Mdb = powerToDb(M, topDb=self.cfg.topDb)

        if sampleMask is None:
            frameMask = np.ones(nFrames, dtype=np.float32)
        else:
            frameMask = np.zeros(nFrames, dtype=np.float32)
            nRealFrames = int(min(self.framesFromMask(sampleMask.astype(np.float32)), nFrames))
            if nRealFrames > 0:
                frameMask[:nRealFrames] = 1.0

        Mdb, frameMask = self.padTime(Mdb, frameMask)
        info = {
            "nFramesRaw": int(nFrames),
            "nFramesPadded": int(Mdb.shape[1]),
            "nMels": int(Mdb.shape[0]),
        }
        return Mdb.astype(np.float32), frameMask.astype(np.float32), info

    def patchify(self, Mdb: np.ndarray, frameMask: np.ndarray, patchSize: Tuple[int, int]=(16,16)) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Author: María Belén
        What: Split Mel spectrogram into non-overlapping patches.
        Params:
            Mdb (np.ndarray): [F, T]
            frameMask (np.ndarray): [T]
            patchSize (Tuple[int,int]): (pF, pT), typically (16, 16)
        Returns:
            patches (np.ndarray): [N, pF, pT]
            patchMask (np.ndarray): [N], 1 if any frame in time-block is real audio
            info (dict): grid and patch metadata
        """
        F, T = Mdb.shape
        pF, pT = patchSize

        Ftarget = ceilDiv(F, pF) * pF
        Ttarget = ceilDiv(T, pT) * pT

        if Ftarget != F:
            Mdb = np.pad(Mdb, ((0, Ftarget - F), (0, 0)), mode="constant")
        if Ttarget != T:
            Mdb = np.pad(Mdb, ((0, 0), (0, Ttarget - T)), mode="constant")
            frameMask = np.pad(frameMask, (0, Ttarget - T), mode="constant")

        Fg = Ftarget // pF
        Tg = Ttarget // pT

        grid = Mdb.reshape(Fg, pF, Tg, pT).transpose(2, 0, 1, 3)
        patches = grid.reshape(Tg * Fg, pF, pT).astype(np.float32)

        fmGrid = frameMask.reshape(Tg, pT)
        timeBlockMask = (np.any(fmGrid > 0.0, axis=1)).astype(np.float32)  # [Tg]
        patchMask = np.repeat(timeBlockMask, Fg).astype(np.float32)        # [Tg*Fg]

        info = {
            "gridT": int(Tg),
            "gridF": int(Fg),
            "numPatches": int(Tg * Fg),
            "patchSize": (int(pF), int(pT)),
        }
        return patches, patchMask, info

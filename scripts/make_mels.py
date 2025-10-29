#!/usr/bin/env python
"""
Author: María Belén
What: Batch pre-processing: waveform -> mel dB -> 16x16 patches + masks.
Params: ver argparse.
Returns: Archivos .npz por audio + manifest.jsonl.
"""
import argparse, json
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

# usa tus módulos estilados
from python.audioloader_styled import AudioLoader, AudioLoaderConfig
from python.melspec_generator_styled import MelSpectrogramGenerator, MelSpecConfig

def findAudioFiles(root: Path) -> List[Path]:
    """Author: María Belén | What: List .wav recursively."""
    return sorted(root.rglob("*.wav"))

def makeDirs(path: Path) -> None:
    """Author: María Belén | What: Create parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)

def processOneFile(
    wavPath: Path,
    outDir: Path,
    loader: AudioLoader,
    gen: MelSpectrogramGenerator,
    patchSize: Tuple[int, int],
) -> dict:
    """Author: María Belén | What: Process a single WAV and save .npz."""
    wave, meta = loader.load(str(wavPath))
    Mdb, frameMask, infoMel = gen.melSpectrogram(wave, sampleMask=meta["mask"])
    patches, patchMask, infoPatch = gen.patchify(Mdb, frameMask, patchSize)

    rel = wavPath.with_suffix(".npz").name
    outPath = outDir / rel
    makeDirs(outPath)
    np.savez_compressed(
        outPath,
        mel=Mdb.astype(np.float32),
        frameMask=frameMask.astype(np.float32),
        patches=patches.astype(np.float32),
        patchMask=patchMask.astype(np.float32),
    )
    return {
        "in": str(wavPath),
        "out": str(outPath),
        "melShape": [int(Mdb.shape[0]), int(Mdb.shape[1])],
        "numPatches": int(patches.shape[0]),
        "patchSize": list(patchSize),
        "grid": [int(infoPatch["gridF"]), int(infoPatch["gridT"])],
        "sr": int(meta["targetSr"]),
        "durationSec": float(meta["durationSec"]),
    }

def saveManifest(rows: List[dict], manifestPath: Path) -> None:
    """Author: María Belén | What: Save JSONL manifest with one line per file."""
    makeDirs(manifestPath)
    with open(manifestPath, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def run() -> None:
    """Author: María Belén | What: Parse args and run batch pre-processing."""
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Folder with WAV files")
    p.add_argument("--dst", required=True, help="Output folder (npz + manifest)")
    p.add_argument("--durationSec", type=float, default=10.0)
    p.add_argument("--normalize", choices=["peak", "rms", "none"], default="peak")
    p.add_argument("--rmsTargetDbfs", type=float, default=-20.0)
    p.add_argument("--nMels", type=int, default=128)
    p.add_argument("--targetNumFrames", type=int, default=1024, help="Frames after padding (AST-like)")
    p.add_argument("--patch", type=int, default=16, help="Patch size (pF=pT)")
    args = p.parse_args()

    loaderCfg = AudioLoaderConfig(
        sampleRate=16000,
        durationSec=args.durationSec,
        normalize=args.normalize,
        rmsTargetDbfs=args.rmsTargetDbfs,
        padSide="end",
        crop="center",
    )
    melCfg = MelSpecConfig(
        sampleRate=16000, nFft=512, winLength=400, hopLength=160,
        nMels=args.nMels, fMin=0.0, fMax=8000.0,
        center=False, topDb=80.0, padTimeToMultiple=16,
        targetNumFrames=args.targetNumFrames
    )
    loader = AudioLoader(loaderCfg)
    gen = MelSpectrogramGenerator(melCfg)
    patchSize = (args.patch, args.patch)

    src = Path(args.src)
    dst = Path(args.dst)
    wavs = findAudioFiles(src)
    rows = []
    for w in tqdm(wavs, desc="Processing WAVs"):
        rows.append(processOneFile(w, dst, loader, gen, patchSize))

    saveManifest(rows, dst / "manifest.jsonl")
    print(f"Done: {len(rows)} files → {dst}")

if __name__ == "__main__":
    run()

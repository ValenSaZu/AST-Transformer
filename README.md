# AST-Transformer

Pipeline para convertir audios de tráfico en **log-mel** y **parches 16×16 (AST-style)** listos para modelos tipo Audio Spectrogram Transformer.

---

## Dataset principal
- Artículo (Nature Scientific Data): https://www.nature.com/articles/s41597-025-04689-3  
  *(El artículo enlaza al repositorio de datos — se descargan más abajo en el Quickstart).*

**Datasets extra (si hay tiempo):**
- Accident & crime (Kaggle): https://www.kaggle.com/datasets/afisarsy/raw-audio-of-accident-and-crime-detection  
- UrbanSound8K (Zenodo): https://zenodo.org/records/3338727

---

## Quickstart (Colab · rama principal)

### 1) Clonar el repo (rama `main`)
```bash
!git clone --branch main --single-branch https://github.com/ValenSaZu/AST-Transformer.git /content/AST-Transformer
%cd /content/AST-Transformer
````

### 2) Herramientas para extraer `.rar`

```bash
!apt-get -y update -qq
!apt-get -y install -qq unar unrar p7zip-full
```

### 3) Descargar y extraer el dataset (MELAUDIS) a `/content/melaudis/wav`

> El artículo de Nature enlaza los archivos de datos. El bloque siguiente los descarga y extrae.

```python
import os, requests, glob, zipfile, subprocess

# carpetas destino
ROOT = "/content/melaudis"
RAW  = f"{ROOT}/raw"
WAV  = f"{ROOT}/wav"
os.makedirs(RAW, exist_ok=True)
os.makedirs(WAV, exist_ok=True)

# ⚠️ Reemplaza esta URL/ID si el artículo publica un enlace distinto
# (Este bloque espera que el artículo apunte a archivos descargables.)
# Ejemplo genérico: una lista de archivos con URLs directas:
file_urls = []
# Si tienes las URLs exactas, añádelas:
# file_urls = ["https://…/MELAUDIS_BG.rar", "https://…/MELAUDIS_Vehicles.rar"]

if not file_urls:
    print("Añade las URLs de los .rar del dataset según el enlace del artículo.")
else:
    for url in file_urls:
        name = os.path.basename(url)
        out = f"{RAW}/{name}"
        if os.path.exists(out):
            print("✓ ya existe:", name); continue
        print("↓", name)
        with requests.get(url, stream=True) as r, open(out, "wb") as w:
            r.raise_for_status()
            for ch in r.iter_content(1024*1024):
                if ch: w.write(ch)

# Extraer .rar (primero 'unar'; si falla, 'unrar')
for rar in glob.glob(f"{RAW}/*.rar"):
    print("Extract:", os.path.basename(rar))
    res = subprocess.run(["unar","-force-overwrite","-o",WAV,rar])
    if res.returncode != 0:
        subprocess.run(["unrar","x","-o+","-y",rar,WAV+"/"], check=True)

# (Por si también hay .zip)
for z in glob.glob(f"{RAW}/*.zip"):
    print("Unzip:", os.path.basename(z))
    with zipfile.ZipFile(z) as zf:
        zf.extractall(WAV)

# Conteo de WAVs
wavs = glob.glob(f"{WAV}/**/*.wav", recursive=True) + glob.glob(f"{WAV}/**/*.WAV", recursive=True)
print("WAVs encontrados:", len(wavs))
print("Ejemplos:", wavs[:3])
```

### 4) Instalar dependencias del proyecto

```bash
%pip install -q -r python/requirements.txt
```

### 5) Preprocesar (loader → log-mel dB → patches 16×16)

```bash
# Ejecutar desde la raíz del repo para que 'python/…' sea importable
PYTHONPATH=$PWD python scripts/make_mels.py \
  --src /content/melaudis/wav \
  --dst /content/melaudis/processed \
  --durationSec 2.0 \
  --normalize peak \
  --targetNumFrames 256 \
  --patch 16
```

### 6) Verificar la salida (formas esperadas)

```python
import glob, numpy as np
outs = glob.glob("/content/melaudis/processed/*.npz")
print("NPZ creados:", len(outs))
if outs:
    z = np.load(outs[0])
    print({k: v.shape for k, v in z.items()})
    # Esperado:
    # mel -> (128, 256)
    # frameMask -> (256,)
    # patches -> (N, 16, 16)
    # patchMask -> (N,)
```

---

## Estructura relevante del repo

```
python/
  audioloader.py
  melspec_generator.py
scripts/
  make_mels.py       # CLI: convierte WAV → mel dB → patches 16×16
```

---

## Notas y reglas rápidas

* **No subir datos** al repo (usa Colab/Drive/descargas locales). Añade `data/`, `*.wav`, `*.npz` a `.gitignore`.
* Ejecuta los scripts desde la **raíz del repo** con `PYTHONPATH=$PWD` para que los imports funcionen.
* Convenciones de colaboración:

  * Ramas: `feat/...`, `fix/...`, `docs/...`
  * Commits (Conventional Commits): `feat(audio): ...`, `fix(mels): ...`

```

::contentReference[oaicite:0]{index=0}
```

"""Build colab/train_round5_colab.ipynb (bundle route, no GitHub/token)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


cells = [
    md(
        """# Grammar Corrector - Round 5 GPU training

Trains the project's grammar model on the generated error corpus.

## Before you start
1. **Runtime -> Change runtime type -> GPU** (a T4 is enough).
2. Upload **one** bundle to `/content` (drag it into the Files panel on the left):
   - **`code_bundle.zip` (~0.5 MB)** - trains `google-t5/t5-small` from
     Hugging Face on your data. Fast, no token, recommended.
   - **`colab_bundle.zip` (~311 MB)** - continues the project's custom 77M
     checkpoint instead of starting from t5-small.
3. Run the cells top to bottom.
4. Download `round5_results.zip` at the end.

The code bundle contains the generator, the training/evaluation scripts, the
9,703 synthetic pairs, the 5,000 CoEdit pairs, and the three held-out sets.
No personal text is included.
"""
    ),
    md("## 1. Check the GPU"),
    code(
        """!nvidia-smi
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
"""
    ),
    md("## 2. Install the libraries"),
    code(
        """!pip install -q "transformers>=4.46,<6" "datasets>=2.18" sentencepiece accelerate
"""
    ),
    md("## 3. Unzip the bundle"),
    code(
        """from pathlib import Path
import shutil, zipfile

WORK = Path("/content/grammartool")
if not Path("/content").exists():
    WORK = Path("/kaggle/working/grammartool")

NAMES = ["code_bundle.zip", "colab_bundle.zip"]
CANDIDATES = (
    [Path("/content") / n for n in NAMES]
    + [Path("/kaggle/working") / n for n in NAMES]
    + [Path("/kaggle/input/grammartool") / n for n in NAMES]
)

BUNDLE = next((p for p in CANDIDATES if p.exists()), None)
if BUNDLE is None:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        drive_dir = Path("/content/drive/MyDrive")
        BUNDLE = next((drive_dir / n for n in NAMES
                       if (drive_dir / n).exists()), None)
    except Exception:
        pass

assert BUNDLE is not None, (
    "Upload code_bundle.zip (or colab_bundle.zip) to /content"
)

if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
with zipfile.ZipFile(BUNDLE) as z:
    z.extractall(WORK)

weights = WORK / "model" / "model.safetensors"
has_weights = weights.exists() and weights.stat().st_size > 100e6
print("bundle:", BUNDLE.name, round(BUNDLE.stat().st_size / 1e6, 2), "MB")
print("workdir:", WORK)
print("top level:", sorted(p.name for p in WORK.iterdir()))
print("custom checkpoint in bundle:", has_weights)
"""
    ),
    md(
        """## 4. Train

- If the bundle contains `model/` (the 311 MB one), this continues the
  project's own 77M checkpoint at LR 2e-5 for 2 epochs.
- Otherwise it fine-tunes `google-t5/t5-small` from Hugging Face at LR 2e-4
  for 3 epochs. That is still your model: your data, your error generator,
  your training pipeline.
"""
    ),
    code(
        """import subprocess, sys
from pathlib import Path

weights = WORK / "model" / "model.safetensors"
has_weights = weights.exists() and weights.stat().st_size > 100e6

if has_weights:
    BASE, LR, EPOCHS = "model", "2e-5", "2"
else:
    BASE, LR, EPOCHS = "google-t5/t5-small", "2e-4", "3"

cmd = [
    sys.executable, str(WORK / "app" / "train_round5.py"),
    "--base", BASE,
    "--out", "model_round5_gpu",
    "--epochs", EPOCHS,
    "--lr", LR,
    "--batch", "32",
    "--synthetic-limit", "9703",
    "--coedit-limit", "5001",
]
print("base model:", BASE, "| lr:", LR, "| epochs:", EPOCHS)
print(" ".join(cmd))
subprocess.run(cmd, cwd=WORK, check=True)
"""
    ),
    md(
        """## 5. Compare decoding strategies

Greedy vs beam-4 vs beam-4 + minimal-edit reranking, on all three held-out
sets. Tells us whether the extra beams are worth the runtime cost.
"""
    ),
    code(
        """subprocess.run(
    [sys.executable, str(WORK / "app" / "eval_decoding.py"),
     "--model", "model_round5_gpu"],
    cwd=WORK, check=True,
)
"""
    ),
    md("## 6. Per-error-family breakdown"),
    code(
        """subprocess.run(
    [sys.executable, str(WORK / "app" / "analyze_round5.py")],
    cwd=WORK, check=True,
)
"""
    ),
    md("## 7. Package and download the results"),
    code(
        """import zipfile
from pathlib import Path

out = Path("/content/round5_results.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in (WORK / "model_round5_gpu").rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(WORK))
    for pattern in ("round5_*", "decoding_model_round5_gpu*"):
        for p in (WORK / "eval").glob(pattern):
            if p.is_file():
                z.write(p, p.relative_to(WORK))
print("wrote", out, round(out.stat().st_size / 1e6, 1), "MB")

try:
    shutil.copy(out, "/content/drive/MyDrive/round5_results.zip")
    print("copied to Drive: MyDrive/round5_results.zip")
except Exception as exc:
    print("Drive copy skipped:", exc)

try:
    from google.colab import files
    files.download(str(out))
except Exception as exc:
    print("Browser download skipped:", exc)
"""
    ),
    md(
        """## 8. Back on the project machine

1. Download `round5_results.zip`.
2. Extract it into `C:\\Users\\BlueSpirit\\Documents\\GrammerTool\\` so that
   `model_round5_gpu\\` sits next to `model\\`.
3. Run:
   ```
   python app/compare_models.py --models model model_round5_gpu
   ```
   If the new model wins on the 300-pair benchmark, it becomes the app model.
"""
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = HERE / "train_round5_colab.ipynb"
path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", path)

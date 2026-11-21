# Round 5 on a GPU (Colab / Kaggle)

The local machine has no GPU (`torch.cuda.is_available()` is `False`), so the
CPU run takes ~45 minutes. This folder runs the same round on a free GPU in a
few minutes.

## Two bundles (pick one)

| Bundle | Size | What it trains |
|---|---|---|
| `code_bundle.zip` | **~0.5 MB** | Fine-tunes `google-t5/t5-small` from Hugging Face on your data. **Recommended** - tiny upload, no token. |
| `colab_bundle.zip` | ~311 MB | Continues the project's custom 77M checkpoint (`model/`). |

Both contain the generator, the training/evaluation scripts, the 9,703
synthetic pairs, the 5,000 CoEdit pairs, and the three held-out sets. No
personal text is included.

## Colab steps (no token)

1. Go to <https://colab.research.google.com> -> **Upload** ->
   `train_round5_colab.ipynb`.
2. **Runtime -> Change runtime type -> GPU** (T4 is enough).
3. Drag `code_bundle.zip` into the **Files** panel on the left (it lands in
   `/content`). If you prefer the custom checkpoint, drag `colab_bundle.zip`
   instead.
4. Run the cells top to bottom.
5. The last cell downloads `round5_results.zip`.

Expected time on a T4: training a few minutes, the evaluations and decoding
comparison a few more.

## Kaggle steps

1. Create a new Notebook, **Settings -> Accelerator -> GPU T4 x2**.
2. Upload `train_round5_colab.ipynb` and run it.
3. Upload the bundle to `/kaggle/working/` (or as a dataset under
   `/kaggle/input/grammartool/`). The unzip cell checks both.

## What the notebook does

1. Unzips `code_bundle.zip` (or `colab_bundle.zip`).
2. Trains:
   - `google-t5/t5-small` at LR 2e-4 for 3 epochs, or
   - the custom `model/` at LR 2e-5 for 2 epochs.
   Both use 9,703 synthetic + 5,001 CoEdit + the earlier targeted pairs.
3. Evaluates exact-match on the 60-pair, 700-pair synthetic, and 300-pair
   held-out sets.
4. Compares greedy / beam-4 / beam-4 + minimal-edit reranking.
5. Breaks the results down by error family.
6. Zips `model_round5_gpu/` plus the metrics into `round5_results.zip`.

## Bringing the result back

1. Download `round5_results.zip`.
2. Extract it into `C:\Users\BlueSpirit\Documents\GrammerTool\` so that
   `model_round5_gpu\` sits next to `model\`.
3. Run:
   ```
   python app/compare_models.py --models model model_round5_gpu
   ```
4. If the new model wins on the 300-pair benchmark, it becomes the app model.

## Changing the run

The training command is in cell 4. Useful flags:

- `--epochs 1` - quicker, less risk of overfitting
- `--lr 1e-5` - gentler continuation
- `--batch 16` - if the GPU runs out of memory
- `--smoke` - 100 pairs, validates the pipeline in under a minute

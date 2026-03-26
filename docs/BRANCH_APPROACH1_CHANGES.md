# Branch changes (Approach 1 / PARSeq jersey aux)

This document summarizes what this branch adds and changes, in order, so you can follow the story from **problem → design → files → how to run**.

---

## 1. Core idea: PARSeq + encoder auxiliary heads

- **Training:** In addition to the usual PARSeq loss, optional **linear heads** on **mean-pooled encoder memory** predict jersey-aligned targets (0–99, tens, units, digit-count), with weights `aux_alpha_*` in config. This matches the same digit conventions as `JerseyNumberMultitaskDataset` in the main repo.
- **Inference:** Unchanged **PARSeq** forward/decode. The aux heads are training-only regularizers; the pipeline still runs `str.py` with a PARSeq checkpoint unless you enable the ResNet path.

**Files (PARSeq subtree):**

| File | Role |
|------|------|
| `str/parseq/configs/model/parseq.yaml` | `use_jersey_aux_heads` (default `false`) and `aux_alpha_full`, `aux_alpha_tens`, `aux_alpha_units`, `aux_alpha_count`. |
| `str/parseq/configs/experiment/parseq-jersey-aux.yaml` | Hydra experiment: `use_jersey_aux_heads: true`. |
| `str/parseq/strhub/models/parseq/system.py` | Builds aux heads when enabled; adds `_jersey_aux_loss` in `training_step`; decoder path unchanged. |
| `str/parseq/train.py` | Loads public PARSeq weights with **shape filtering** and **`strict=False`** so new heads (and shorter `max_label_length`) do not break checkpoint load. |

---

## 2. `main.py`: how you train and what the pipeline defaults to

- **`--train_str`** still launches PARSeq `train.py` via subprocess (using `configuration.str_python`).
- **`--jersey_aux_str`** (use with `--train_str`): passes Hydra **`+experiment=parseq-jersey-aux`** instead of plain `parseq`.
- **Default SoccerNet pipeline** sets **`use_multitask_classifier: False`** so STR inference uses **PARSeq** again.
- **`multitask_model_path`** is resolved with `os.path.dirname(os.path.abspath(__file__))` so it works even if the current working directory is not the repo root.
- **LMDB preflight:** Before starting training, **`_parseq_training_lmdb_ready`** checks that `data.mdb` exists under `train/` or `train/real/` (and capitalized `Train/` variants). If not, it prints instructions and **returns** (avoids a long Hydra traceback when data is missing).
- **`JERSEY_STR_LMDB_ROOT`:** If set, overrides the default LMDB root for STR training (absolute path, or path relative to the shell’s cwd). Same mechanism works for Hockey STR training if you point it at the Hockey LMDB tree.

---

## 3. `configuration.py`: which Python runs PARSeq

- **`str_env`** defaults to **`jersey`** (not `parseq2`), because many developers use one main conda env. PARSeq train/inference subprocesses use `str_python` built from that env name.
- **`JERSEY_STR_ENV`:** Override env name if you still want a dedicated `parseq2` (or other) interpreter.
- **Comment** on SoccerNet `numbers_data`: clarifies that PARSeq training needs the **separate LMDB download** from the README, not only raw jersey-2023 images.
- **Pipeline outputs (related branch work):** `JERSEY_PIPELINE_OUTPUT_SLUG`, `JERSEY_USE_SHARED_PIPELINE_OUTPUT`, and git-branch-based `out/pipeline_runs/<slug>/SoccerNetResults` for non-`main`/`master` branches (see `configuration.py`).

---

## 4. Robustness fixes for PARSeq training on your machine

These are not “jersey math” changes; they fix **environment and data-layout** issues you hit when running from the **`jersey`** env (Python 3.14, NumPy 2, Windows, shorter labels, missing LMDB).

### 4.1 Pretrained checkpoint vs `max_label_length=2`

- Public PARSeq weights use a **longer** `pos_queries` than a 2-character jersey setup.
- **`train.py`** defines **`_filter_pretrained_state_dict`**: only load tensors whose **shapes match** the current model; skip mismatches (e.g. `pos_queries`) so they stay randomly initialized and learn during fine-tuning.

### 4.2 Python 3.14 + Hydra argparse

- **`str/parseq/train.py`:** Small patch before importing Hydra: on Python ≥ 3.14, coerce non-string `argparse` help for Hydra’s `--shell-completion` so `train.py` starts.

### 4.3 NumPy 2.0 + imgaug

- **`str/parseq/strhub/data/augment.py`:** If `np.sctypes` is missing, define a minimal dict **before** `import imgaug`, because imgaug 0.4.x expects `np.sctypes` at import time.

### 4.4 LMDB layout (SoccerNet zip variants)

- **`str/parseq/strhub/data/dataset.py`:** Skip LMDBs with **0 samples** after charset/length filtering; raise a clear **`FileNotFoundError`** if nothing usable remains (instead of an empty `ConcatDataset`).
- **`str/parseq/strhub/data/module.py`:** Try **`train/real`** then **`train`**, and **`Train/`** / **`Val/`** capitalized variants, so common unzip layouts work.

---

## 5. Documentation touch-up

- **`docs/MODEL_SPECS.md`:** Short note on training with **`--jersey_aux_str`** / `parseq-jersey-aux`.

---

## 6. What you need on disk for STR training

PARSeq training is **not** reading full-frame SoccerNet images from `jersey-2023/train/...`. It needs the **weakly labelled jersey number crops in LMDB** linked from the project **README** (“Weakly-labelled jersey number crops … LMDB”).

Expected layout under your LMDB root (or under `data/SoccerNet/jersey-2023/lmdb` by default):

- `train/.../data.mdb` or `train/real/.../data.mdb`
- `val/.../data.mdb` (for validation)

If you unpacked elsewhere:

```text
JERSEY_STR_LMDB_ROOT=<folder that contains train/ and val/>
```

---

## 7. Commands cheat sheet

### Switching STR training mode (SoccerNet)

| Mode | Hydra experiment | Command |
|------|-------------------|---------|
| **Standard / baseline** — PARSeq only, no encoder aux heads | `parseq` | `python main.py SoccerNet train --train_str` |
| **Approach 1 (this branch)** — PARSeq + jersey aux heads on the encoder | `parseq-jersey-aux` | `python main.py SoccerNet train --train_str --jersey_aux_str` |

The only difference is **`--jersey_aux_str`**: without it, `main.py` uses **`+experiment=parseq`**; with it, **`+experiment=parseq-jersey-aux`** (see `train_parseq` in `main.py`).


**After training:** Point `configuration.py` → `str_model` (SoccerNet or Hockey) to your new checkpoint under `str/parseq/outputs/...`, or copy the `.ckpt` into `models/` and update the path. For SoccerNet you can also use **`JERSEY_SOCCERNET_STR_CKPT`** or **`JERSEY_STR_MODEL`** (see `configuration.py`) to override the checkpoint path without editing the file.


**Standard run**
```bash
$env:JERSEY_USE_SHARED_PIPELINE_OUTPUT = "1"

```
**approach1**
```bash
Remove-Item Env:JERSEY_USE_SHARED_PIPELINE_OUTPUT -ErrorAction SilentlyContinue

```

**Full pipeline (STR = PARSeq by default):**

```bash
python main.py SoccerNet test
```

---

## 8. File checklist (quick reference)

- `main.py` — `--jersey_aux_str`, LMDB preflight, `JERSEY_STR_LMDB_ROOT`, pipeline defaults
- `configuration.py` — `str_env` / `JERSEY_STR_ENV`, LMDB comment, pipeline output slug
- `str/parseq/train.py` — argparse 3.14 shim, `_filter_pretrained_state_dict`, `strict=False` load
- `str/parseq/strhub/data/augment.py` — NumPy 2 `sctypes` shim
- `str/parseq/strhub/data/dataset.py` — safer `build_tree_dataset`
- `str/parseq/strhub/data/module.py` — train/val path fallbacks
- `str/parseq/configs/model/parseq.yaml` — aux flags and alphas
- `str/parseq/configs/experiment/parseq-jersey-aux.yaml` — enable aux
- `str/parseq/strhub/models/parseq/system.py` — aux heads + loss
- `docs/MODEL_SPECS.md` — training note

---

## 9. Known rough edges

- **`main.py`** still prints **“Done training”** even if the subprocess fails; check the exit code / traceback in the terminal.
- **`trainer.val_check_interval=1`** (from `train_parseq`) validates **every batch**, which is very slow; consider loosening that in `main.py` once training is stable.

If you merge this branch, reviewers can use this file as the single narrative for “what changed and why.”

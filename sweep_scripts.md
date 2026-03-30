# Sweep scripts reference

This document describes the three Python sweep utilities in the project root. They all **reuse existing scene-text recognition (STR) outputs** (`jersey_id_result` JSON) and **do not retrain** models. They differ in *what* they sweep and *how* predictions are aggregated per tracklet.

---

## Shared building blocks

All three scripts duplicate the same small pipeline pieces (they could be factored into a module, but are kept self-contained).

### `list_dirs(path)`

Returns subdirectory names under `path`. Used to enumerate **tracklet IDs** that have image folders in the SoccerNet split.

### `consolidated_results(image_dir, results_dict, illegible_path, soccer_ball_list=None)`

Turns per-tracklet numeric predictions from the STR pipeline into a **final dictionary** aligned with the evaluation setup:

1. **Start** from `results_dict` (tracklet ID → predicted jersey string).
2. **Soccer ball tracks**: If `soccer_ball_list` is provided, every track ID listed there is forced to label **`1`** (convention for “ball” in this project).
3. **Illegible list**: Entries in `illegible_result` JSON get **`-1`** if not already present (illegible / abstain).
4. **Coverage**: Every tracklet folder under `image_dir` gets an entry: missing → **`-1`**, existing → **integer** prediction.

So evaluation always sees one prediction per tracklet folder, with special handling for balls and illegible tracks.

### `evaluate(consolidated_dict, gt_dict)`

Compares consolidated predictions to ground truth (`gt` JSON): counts **correct**, **total** (over GT keys), and **accuracy** as `100 * correct / total`. A prediction matches if `str(gt[id]) == str(predicted)`.

### Data paths (all scripts)

- **Split**: `part = 'test'` (SoccerNet test).
- **Inputs**: `configuration.dataset['SoccerNet']` for `root_dir`, `working_dir`, and keys for images, `jersey_id_result`, `soccer_ball_list`, `illegible_result`, and `gt`.

---

## `sweep_consolidation.py`

**Purpose:** Compare **many consolidation / aggregation strategies** on the *same* STR result file, then report test accuracy. This answers: “Given fixed per-frame STR outputs, which way of combining them into one jersey per tracklet works best?”

**How to run:** `python sweep_consolidation.py` (from project root, with env that has deps and `configuration` / `helpers`).

### What it sweeps

1. **Original weighted-sum (`helpers.process_jersey_id_predictions`)**  
   - With `useBias=True` and `useBias=False`.  
   - This is the baseline STR post-processing in `helpers.py`.

2. **Bayesian variant (`helpers.process_jersey_id_predictions_bayesian`)**  
   - Nested loop over `useTS`, `useBias`, `useTh` each in `{True, False}` → **8 combinations**.  
   - Flags control temperature scaling, digit bias, and thresholding inside that implementation.

3. **Raw probability averaging (`helpers.process_jersey_id_predictions_raw`)**  
   - Once with `useTS=False`, once with `useTS=True`.

4. **Bayesian temperature sweep**  
   - Temporarily sets `helpers.TS` to each of `[1.0, 1.5, 2.0, 2.367, 3.0, 4.0, 5.0]`.  
   - Runs Bayesian with `useTS=True`, `useBias=True`, `useTh=False`.  
   - Restores `helpers.TS` at the end.

### Output

A table: method label, correct count, total, accuracy (%). Every row uses the same `consolidated_results` + `evaluate` path so numbers are comparable.

---

## `sweep_thresholds.py`

**Purpose:** **Grid search** over two numeric thresholds used in a **weighted-sum style** aggregator (implemented *inline* in this file as `process_with_thresholds`, not by calling `process_jersey_id_predictions` directly). Optionally toggles **digit bias**. Use this to find good `FILTER_THRESHOLD` / `SUM_THRESHOLD` style hyperparameters for the classic pipeline.

**How to run:** `conda run -n jersey python sweep_thresholds.py` (as noted in the docstring).

### `process_with_thresholds(file_path, filter_th, sum_th, useBias=True)`

1. **Load** `jersey_id_result` JSON: keys are composite names; **tracklet** = first token before `_`.
2. For each detection: keep only **valid numeric** labels (`helpers.is_valid_number`).
3. **Per-detection weight**: product of confidence values `confidence[:-1]` (same idea as multiplying per-character probs in STR).
4. **Filter threshold (`filter_th`)**: If `filter_th > 0`, any row with weight **below** `filter_th` has its weight set to **0** (drops weak evidence).
5. **Per digit class**: For each unique predicted digit value, sum **adjusted** weights = `weight * helpers.get_bias(value)` if `useBias`, else `weight * 1`.
6. **Sum threshold (`sum_th`)**: Pick the class with **maximum** total weight; if that maximum is **not** greater than `sum_th`, predict **`-1`** (abstain). Otherwise predict that digit.

So `filter_th` prunes low-confidence frames; `sum_th` requires enough accumulated evidence before committing to a number.

### Grid

- `filter_th ∈ {0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5}`
- `sum_th ∈ {0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0}`
- `use_bias ∈ {True, False}`

### Output and filtering

- Tracks **best** `(filter_th, sum_th, use_bias, correct)` by accuracy.
- **Prints only** combinations with accuracy **≥ 86.5%** (reduces noise).
- Final lines print the **best** parameters and accuracy.

---

## `sweep_thresholds2.py`

**Purpose:** **Refinement sweep** around a promising region of `(filter_th, sum_th)` after the coarse search in `sweep_thresholds.py`. Docstring: “Extended sweep around the best region.”

**How to run:** Same idea as the other sweeps—`python sweep_thresholds2.py` with project on `PYTHONPATH` and correct conda env (docstring is minimal; mirror `sweep_thresholds.py` if needed).

### Differences from `sweep_thresholds.py`

- **Fixed `useBias=True`** — no bias on/off loop; assumes bias helps and focuses on thresholds.
- **Narrower grids** (finer steps where coarse search suggested good performance):
  - `filter_th ∈ [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]`
  - `sum_th ∈ [1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 5.0]`
- **Print threshold** for rows: accuracy **≥ 87.0%** (slightly stricter than the first sweep’s 86.5%).
- `process_with_thresholds`, `consolidated_results`, and `evaluate` are the **same logic** as in `sweep_thresholds.py`.

### Output

Table columns: `FILTER_TH`, `SUM_TH`, `Correct`, `Total`, `Accuracy`; then **BEST** `filter_th`, `sum_th`, correct count, and accuracy.

---

## Summary comparison

| Script | Sweeps | Bias toggle | Typical use |
|--------|--------|-------------|-------------|
| `sweep_consolidation.py` | Methods: weighted-sum, Bayesian (flags + TS values), raw averaging | Part of method definitions in `helpers` | Pick a **consolidation algorithm** |
| `sweep_thresholds.py` | `filter_th`, `sum_th`, `use_bias` (coarse grid) | Yes | **Coarse** threshold tuning for weighted-sum path |
| `sweep_thresholds2.py` | `filter_th`, `sum_th` (dense around prior best) | No (`True` only) | **Fine** threshold tuning |

All three assume **SoccerNet test** paths in `configuration.py` are valid and that `jersey_id_result` and auxiliary JSONs already exist from an upstream pipeline run.

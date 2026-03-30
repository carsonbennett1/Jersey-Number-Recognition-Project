#!/usr/bin/env python3
"""
Report oracle upper bounds after stage 6 (crops fixed): STR (stage 7) + combine can change.

1) Perfect STR oracle: synthetic jersey_id_results where each crop predicts the GT number
   (when GT is a valid jersey label per helpers.is_valid_number), or omits crops when GT is -1.
   Then runs the same combine as main.py (Top-L by default, or Bayesian with --combine-bayesian)
   plus consolidated_results.

2) Aggregation-only ceiling: fraction of tracklets where GT appears among at least one valid
   crop label in the current STR output (reweighting could in principle pick GT via find_best_prediction).

Usage (from repo root, with project env activated):
  python oracle_max_after_stage6.py
  python oracle_max_after_stage6.py --part test --working-dir out/pipeline_runs/approach1/SoccerNetResults
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Optional, Set, Tuple

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import configuration as config  # noqa: E402
import helpers  # noqa: E402


def _norm_gt(v: Any) -> Optional[int]:
    s = str(v).strip()
    if s == "-1":
        return -1
    try:
        return int(s)
    except ValueError:
        return None


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _consolidated_results(
    image_dir: str,
    pred_dict: Dict[str, str],
    illegible_path: str,
    soccer_ball_list: Optional[str],
) -> Dict[str, int]:
    """Same logic as main.consolidated_results (avoid importing main)."""
    dict_out = {str(k): v for k, v in pred_dict.items()}
    balls_list: list = []
    if soccer_ball_list is not None and os.path.isfile(soccer_ball_list):
        with open(soccer_ball_list, "r", encoding="utf-8") as sf:
            balls_json = json.load(sf)
        balls_list = balls_json["ball_tracks"]
        for entry in balls_list:
            dict_out[str(entry)] = 1

    with open(illegible_path, "r", encoding="utf-8") as f:
        illegile_dict = json.load(f)
    all_illegible = illegile_dict["illegible"]
    for entry in all_illegible:
        if str(entry) not in dict_out.keys():
            dict_out[str(entry)] = -1

    all_tracks = helpers.list_dirs(image_dir)
    for t in all_tracks:
        if t not in dict_out.keys():
            dict_out[t] = -1
        else:
            dict_out[t] = int(dict_out[t])
    return dict_out


def _accuracy(consolidated: Dict[str, int], gt_dict: Dict[str, Any]) -> Tuple[int, int, float]:
    correct = 0
    total = 0
    for tid in gt_dict.keys():
        tid_s = str(tid)
        try:
            predicted = consolidated[tid_s]
        except KeyError:
            predicted = -1
        gt = _norm_gt(gt_dict[tid])
        if gt is None:
            continue
        if str(gt) == str(predicted):
            correct += 1
        total += 1
    acc = 100.0 * correct / total if total else 0.0
    return correct, total, acc


def _build_perfect_str_json(
    jersey_id_results: Dict[str, Any],
    gt_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Oracle: for each crop key, if track GT is 1..99 (is_valid_number), set label to GT and keep
    confidence structure from the real run. If GT is -1, drop all crop keys for that track (empty
    aggregation -> -1). If GT is 0 or invalid for is_valid_number, omit oracle fix (keep real entry
    to avoid mis-simulating).
    """
    by_track: Dict[str, list] = defaultdict(list)
    for name in jersey_id_results.keys():
        tr = name.split("_")[0]
        by_track[tr].append(name)

    out: Dict[str, Any] = {}
    for tr, names in by_track.items():
        gt = None
        if str(tr) in gt_dict:
            gt = _norm_gt(gt_dict[str(tr)])
        elif tr in gt_dict:
            gt = _norm_gt(gt_dict[tr])
        if gt is None:
            for n in names:
                out[n] = jersey_id_results[n]
            continue

        if gt == -1:
            # No crop entries -> process never sees this track; consolidated fills from image_dir
            continue

        lab = str(gt)
        if not helpers.is_valid_number(lab):
            # e.g. 0 — cannot be emitted by current STR validity rules
            for n in names:
                out[n] = jersey_id_results[n]
            continue

        for name in names:
            entry = dict(jersey_id_results[name])
            entry["label"] = lab
            out[name] = entry

    return out


def _aggregation_label_sets(jersey_id_results: Dict[str, Any]) -> Dict[str, Set[int]]:
    """Valid integer labels per track from current STR file."""
    by_track: Dict[str, Set[int]] = defaultdict(set)
    for name, meta in jersey_id_results.items():
        tr = name.split("_")[0]
        lab = meta.get("label")
        if lab is None:
            continue
        if helpers.is_valid_number(str(lab)):
            by_track[tr].add(int(lab))
    return dict(by_track)


def _aggregation_ceiling(
    jersey_id_results: Dict[str, Any],
    gt_dict: Dict[str, Any],
) -> Tuple[int, int, float]:
    """
    Upper bound if you could only reweight existing crop predictions (labels fixed per crop):
    GT must appear as a valid label on at least one crop for that track (GT > 0).
    For GT == -1: achievable if that track has no valid crop labels in the JSON (empty aggregation → -1).
    """
    label_sets = _aggregation_label_sets(jersey_id_results)
    tracks_with_keys = set(label_sets.keys())

    ok = 0
    total = 0
    for tid in gt_dict.keys():
        tid_s = str(tid)
        gt = _norm_gt(gt_dict[tid])
        if gt is None:
            continue
        total += 1
        if gt == -1:
            if tid_s not in tracks_with_keys or len(label_sets.get(tid_s, set())) == 0:
                ok += 1
            continue
        if gt in label_sets.get(tid_s, set()):
            ok += 1

    acc = 100.0 * ok / total if total else 0.0
    return ok, total, acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle accuracy bounds after stage 6 (fixed crops)")
    parser.add_argument(
        "--part",
        default="test",
        choices=("test", "val", "train", "challenge"),
        help="SoccerNet split (default: test)",
    )
    parser.add_argument(
        "--working-dir",
        default=None,
        help="Pipeline working_dir (default: configuration.py SoccerNet working_dir)",
    )
    parser.add_argument(
        "--jersey-id",
        default=None,
        help="Path to jersey_id_results JSON (default: <working-dir>/<part jersey_id_result>)",
    )
    parser.add_argument(
        "--combine-bayesian",
        action="store_true",
        default=False,
        help="Use Bayesian combine like process_jersey_id_predictions (default: Top-L, same as main.py)",
    )
    args = parser.parse_args()

    sn = config.dataset["SoccerNet"]
    working_dir = args.working_dir or sn["working_dir"]
    part_cfg = sn[args.part]

    root_dir = sn["root_dir"]
    image_dir = os.path.normpath(os.path.join(root_dir, part_cfg["images"]))
    gt_path = os.path.normpath(os.path.join(root_dir, part_cfg["gt"]))
    illegible_path = os.path.join(working_dir, part_cfg["illegible_result"])
    soccer_ball_list = os.path.join(working_dir, part_cfg["soccer_ball_list"])
    jersey_path = args.jersey_id or os.path.join(working_dir, part_cfg["jersey_id_result"])
    raw_legible_path = os.path.join(working_dir, part_cfg["raw_legible_result"])
    gauss_filtered_path = os.path.join(working_dir, part_cfg["gauss_filtered"])

    if not os.path.isfile(jersey_path):
        print(f"ERROR: Missing jersey_id results: {jersey_path}")
        sys.exit(1)
    if not os.path.isfile(gt_path):
        print(f"ERROR: Missing GT: {gt_path}")
        sys.exit(1)
    if not os.path.isdir(image_dir):
        print(f"ERROR: Missing image dir: {image_dir}")
        sys.exit(1)

    gt_dict = _load_json(gt_path)
    jersey_id_results = _load_json(jersey_path)

    # --- Aggregation-only ceiling (current crop labels) ---
    agg_ok, agg_tot, agg_acc = _aggregation_ceiling(jersey_id_results, gt_dict)
    print("=== After stage 6: oracle bounds ===")
    print(f"Working dir: {working_dir}")
    print(f"Jersey ID file: {jersey_path}")
    print()
    print(
        "Aggregation-only ceiling (GT must appear on at least one crop as a valid STR label; "
        "GT=-1 only if no valid labels on any crop for that track):"
    )
    print(f"  Achievable tracklets: {agg_ok} / {agg_tot}  ({agg_acc:.6f}%)")
    print()

    # --- Perfect STR oracle (synthetic JSON + same combine as pipeline) ---
    tmp_path = os.path.join(working_dir, "_oracle_perfect_str_jersey_id_results.json")
    perfect = _build_perfect_str_json(jersey_id_results, gt_dict)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(perfect, f)
        if args.combine_bayesian:
            pred_from_process, _ = helpers.process_jersey_id_predictions(tmp_path, useBias=True)
        else:
            if not os.path.isfile(raw_legible_path):
                print(f"ERROR: Top-L combine needs raw legibility JSON: {raw_legible_path}")
                sys.exit(1)
            pred_from_process, _ = helpers.process_jersey_id_predictions_top_L(
                tmp_path,
                raw_legibility_path=raw_legible_path,
                filtered_results_path=gauss_filtered_path,
                useBias=True,
            )
        consolidated = _consolidated_results(
            image_dir,
            pred_from_process,
            illegible_path,
            soccer_ball_list if os.path.isfile(soccer_ball_list) else None,
        )
        p_ok, p_tot, p_acc = _accuracy(consolidated, gt_dict)
        combine_name = "process_jersey_id_predictions (Bayesian)" if args.combine_bayesian else "process_jersey_id_predictions_top_L"
        print(
            f"Perfect STR oracle (every crop predicts GT when GT is 1-99; GT=-1: no crop entries; "
            f"then {combine_name} + consolidated_results as in main.py):"
        )
        print(f"  Achievable tracklets: {p_ok} / {p_tot}  ({p_acc:.6f}%)")
    finally:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    print()
    print(
        "Note: GT=0 is excluded by helpers.is_valid_number; tracks with no crops in jersey_id "
        "cannot recover a positive GT without upstream changes. Ball tracks are forced to 1 in "
        "consolidated_results when soccer_ball_list is present."
    )


if __name__ == "__main__":
    main()

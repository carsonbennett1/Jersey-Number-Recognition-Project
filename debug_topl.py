import argparse
import json
import os
import random

import numpy as np

import configuration as config
import helpers


def list_dirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def consolidated_results_local(image_dir, pred_dict, illegible_path, soccer_ball_list=None):
    out = dict(pred_dict)
    balls_list = []
    if soccer_ball_list and os.path.isfile(soccer_ball_list):
        with open(soccer_ball_list, "r") as sf:
            balls_json = json.load(sf)
        balls_list = balls_json.get("ball_tracks", [])
        for entry in balls_list:
            out[str(entry)] = 1

    with open(illegible_path, "r") as f:
        illegible_dict = json.load(f)
    for entry in illegible_dict.get("illegible", []):
        if str(entry) not in out:
            out[str(entry)] = -1

    for t in list_dirs(image_dir):
        if t not in out:
            out[t] = -1
        else:
            out[t] = int(out[t])
    return out


def sliced_metrics(consolidated_dict, gt_dict):
    def _acc(items):
        if not items:
            return 0.0, 0, 0
        correct = 0
        for k in items:
            pred = consolidated_dict.get(k, -1)
            if int(pred) == int(gt_dict[k]):
                correct += 1
        return 100.0 * correct / len(items), correct, len(items)

    all_ids = list(gt_dict.keys())
    legible = [k for k, v in gt_dict.items() if int(v) != -1]
    illegible = [k for k, v in gt_dict.items() if int(v) == -1]

    overall = _acc(all_ids)
    leg = _acc(legible)
    ill = _acc(illegible)
    return {
        "overall": overall,
        "legible_only": leg,
        "illegible_only": ill,
    }


def print_hist(pred_dict, top_n=15):
    counts = {}
    for v in pred_dict.values():
        key = str(v)
        counts[key] = counts.get(key, 0) + 1
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    print("\nPrediction histogram (top labels):")
    for k, c in items[:top_n]:
        print(f"  label={k:>3} count={c}")


def run_eval(name, pred_fn, image_dir, illegible_path, soccer_ball_path, gt_path):
    pred_dict, full = pred_fn()
    consolidated = consolidated_results_local(
        image_dir, pred_dict, illegible_path, soccer_ball_list=soccer_ball_path
    )
    with open(gt_path, "r") as f:
        gt = json.load(f)
    m = sliced_metrics(consolidated, gt)
    print(f"\n[{name}]")
    print(f"  overall: {m['overall'][0]:.4f}% ({m['overall'][1]}/{m['overall'][2]})")
    print(f"  legible-only: {m['legible_only'][0]:.4f}% ({m['legible_only'][1]}/{m['legible_only'][2]})")
    print(
        f"  illegible-only: {m['illegible_only'][0]:.4f}% "
        f"({m['illegible_only'][1]}/{m['illegible_only'][2]})"
    )
    print_hist(consolidated)
    return consolidated, full, gt


def alignment_sanity(str_result_file, raw_legibility_file, filtered_file, sample_n=3):
    with open(str_result_file, "r") as f:
        str_results = json.load(f)
    with open(raw_legibility_file, "r") as f:
        raw_scores = json.load(f)
    with open(filtered_file, "r") as f:
        filtered = json.load(f)

    by_track = {}
    for name in str_results.keys():
        tmp = name.split("_", 1)
        track = tmp[0]
        frame = tmp[1] if len(tmp) > 1 else ""
        by_track.setdefault(track, []).append(frame)

    tracks = [t for t in by_track.keys() if t in raw_scores and t in filtered]
    random.shuffle(tracks)
    tracks = tracks[:sample_n]
    print("\nAlignment sanity:")
    for t in tracks:
        ordered = sorted(by_track[t], key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
        source = filtered[t]
        q = raw_scores[t]
        frame_to_score = {frame: q[idx] for idx, frame in enumerate(source) if idx < len(q)}
        pairs = [(f, frame_to_score.get(f, None)) for f in ordered[:10]]
        print(f"  track={t} len_ordered={len(ordered)} len_source={len(source)} len_q={len(q)}")
        print(f"    ordered[:10]={ordered[:10]}")
        print(f"    source[:10]={source[:10]}")
        print(f"    pairs[:10]={pairs}")


def _frame_sort_key(frame_name: str):
    stem = os.path.splitext(frame_name)[0]
    return int(stem) if stem.isdigit() else stem


def compute_qt_match_coverage(str_result_file, raw_legibility_file, filtered_file, gt_path, min_ordered_frames=3):
    """
    q_t mapping match coverage:
    For each tracklet, count how many ordered STR frames successfully map to a q_t score
    using the same normalized-key mapping idea as Top-L scoring:
      - if filtered list is available for that track: map filtered frame -> raw score using
        helpers._norm_frame_key, then check how many ordered STR frames produce keys found.
      - else: positional fallback => mapped_count = min(len(ordered), len(raw_scores)).
    """
    with open(str_result_file, "r") as f:
        str_results = json.load(f)
    with open(raw_legibility_file, "r") as f:
        raw_scores = json.load(f)
    with open(filtered_file, "r") as f:
        filtered = json.load(f)
    with open(gt_path, "r") as f:
        gt = json.load(f)

    # Group frames per tracklet (store frame_name only; logits are not needed for mapping coverage).
    by_track = {}
    for full_name in str_results.keys():
        tmp = full_name.split("_", 1)
        track = tmp[0]
        frame = tmp[1] if len(tmp) > 1 else ""
        if not frame:
            continue
        by_track.setdefault(track, []).append(frame)

    cover_rows = []
    for tracklet, frames in by_track.items():
        ordered = sorted(frames, key=_frame_sort_key)
        if len(ordered) < min_ordered_frames:
            continue

        raw_tracklet_scores = raw_scores.get(tracklet, [])

        if tracklet in filtered and tracklet in raw_scores:
            frame_to_score = {}
            src_list = filtered.get(tracklet, [])
            for idx, frame in enumerate(src_list):
                if idx >= len(raw_tracklet_scores):
                    break
                k = helpers._norm_frame_key(frame, tracklet)
                frame_to_score[k] = float(raw_tracklet_scores[idx])

            mapped = 0
            for frame in ordered:
                k = helpers._norm_frame_key(frame, tracklet)
                if k in frame_to_score:
                    mapped += 1
            coverage = mapped / float(len(ordered)) if ordered else 0.0
            cover_rows.append((tracklet, coverage, mapped, len(ordered), int(gt.get(tracklet, -1))))
        else:
            mapped = min(len(ordered), len(raw_tracklet_scores))
            coverage = mapped / float(len(ordered)) if ordered else 0.0
            cover_rows.append((tracklet, coverage, mapped, len(ordered), int(gt.get(tracklet, -1))))

    # Summaries
    legible_rows = [r for r in cover_rows if r[4] != -1]
    illegible_rows = [r for r in cover_rows if r[4] == -1]

    def _mean(rows):
        return sum(x[1] for x in rows) / len(rows) if rows else 0.0

    print("\nQ_t mapping match coverage:")
    print(f"  tracks checked: {len(cover_rows)} (legible={len(legible_rows)}, illegible={len(illegible_rows)})")
    print(f"  avg coverage (legible GT!= -1):   {_mean(legible_rows):.4f}")
    print(f"  avg coverage (illegible GT== -1): {_mean(illegible_rows):.4f}")

    worst = sorted(cover_rows, key=lambda x: x[1])[:10]
    print("  worst 10 tracklets (coverage, mapped/ordered, GT):")
    for tracklet, cov, mapped, ordered_n, gt_v in worst:
        print(f"    track={tracklet} cov={cov:.3f} mapped={mapped}/{ordered_n} GT={gt_v}")


def diagnose_L_frame_selection(str_result_file, raw_legibility_file, filtered_file, gt_path, useBias=True, sample_n=5, L_small=1, L_large=16):
    """
    Verify L changes selected frames for the best candidate:
    For a few tracklets, recompute candidate_probs and vt(k) then:
      - choose best k under L_small and print selected top-L frame indices
      - choose best k under L_large and print selected top-L frame indices
    """
    with open(str_result_file, "r") as f:
        str_results = json.load(f)
    with open(raw_legibility_file, "r") as f:
        raw_scores = json.load(f)
    with open(filtered_file, "r") as f:
        filtered = json.load(f)
    with open(gt_path, "r") as f:
        gt = json.load(f)

    # Collect per-track (ordered frames + logits)
    by_track = {}
    for full_name in str_results.keys():
        tmp = full_name.split("_", 1)
        track = tmp[0]
        frame = tmp[1] if len(tmp) > 1 else ""
        if not frame:
            continue
        by_track.setdefault(track, []).append((frame, full_name))

    candidates = [t for t in by_track.keys() if t in raw_scores and t in gt]
    random.shuffle(candidates)
    candidates = candidates[:sample_n]

    eps = 1e-9

    print("\nL frame selection diagnostics (best k and top-L frames):")
    for tracklet in candidates:
        frame_full = by_track[tracklet]
        ordered_pairs = sorted(frame_full, key=lambda x: _frame_sort_key(x[0]))
        ordered_frames = [p[0] for p in ordered_pairs]

        # logits with same ordering as ordered_frames
        logits_list = []
        for _, full_name in ordered_pairs:
            raw_result = str_results[full_name]["logits"]
            logits_list.append(helpers.apply_ts(raw_result))
        results = np.array(logits_list)

        # q_t mapping (same normalization-based approach as Top-L scoring in helpers.py)
        raw_tracklet_scores = raw_scores[tracklet]
        if tracklet in filtered:
            frame_to_score = {}
            src_list = filtered.get(tracklet, [])
            for idx, frame in enumerate(src_list):
                if idx >= len(raw_tracklet_scores):
                    break
                k = helpers._norm_frame_key(frame, tracklet)
                frame_to_score[k] = float(raw_tracklet_scores[idx])

            qt = []
            for frame in ordered_frames:
                k = helpers._norm_frame_key(frame, tracklet)
                q = frame_to_score.get(k, 0.0)
                qt.append(q)
            qt = np.array(qt, dtype=float)
        else:
            qt = np.array(raw_tracklet_scores[:len(ordered_frames)], dtype=float)
            if len(qt) < len(ordered_frames):
                qt = np.concatenate([qt, np.zeros(len(ordered_frames) - len(qt), dtype=float)])

        # Candidate probs build
        tens_priors, unit_priors = helpers.initialize_priors(useBias)
        tens_likelihood, unit_likelihood = helpers.split_predictions_by_digit(results, priors=(tens_priors, unit_priors))
        candidate_probs = helpers.build_candidate_probs_1_to_99(tens_likelihood, unit_likelihood)  # [T,99]

        gt_v = gt.get(tracklet, -1)
        print(f"\n  track={tracklet} GT={gt_v} len_frames={len(ordered_frames)} max_qt={float(np.max(qt)):.4f}")

        for L in [L_small, L_large]:
            best_k, _, _ = helpers.top_l_candidate_score(candidate_probs, qt, L=L, eps=eps)
            best_idx = int(best_k) - 1
            vt = qt * np.log(candidate_probs[:, best_idx] + eps)
            L_eff = min(L, len(vt))
            top_idx = np.argsort(vt)[-L_eff:]
            # sort by vt descending for readability
            top_idx = top_idx[np.argsort(vt[top_idx])[::-1]]

            selected_frames = [ordered_frames[i] for i in top_idx.tolist()]
            print(f"    L={L}: best_k={best_k} top_idx={top_idx.tolist()} selected_frames={selected_frames}")

        print("    (If the selected_frames sets are identical, L sweep may be effectively flat.)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default="test", choices=["test", "val", "train", "challenge"])
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wrong_n", type=int, default=50)
    args = parser.parse_args()
    random.seed(args.seed)

    ds = config.dataset["SoccerNet"][args.part]
    work = config.dataset["SoccerNet"]["working_dir"]
    root = config.dataset["SoccerNet"]["root_dir"]

    str_result_file = os.path.join(work, ds["jersey_id_result"])
    illegible_path = os.path.join(work, ds["illegible_result"])
    soccer_ball_path = os.path.join(work, ds["soccer_ball_list"])
    gt_path = os.path.join(root, ds["gt"])
    image_dir = os.path.join(root, ds["images"])
    filtered_file = os.path.join(work, ds["gauss_filtered"])
    raw_legibility_file = os.path.join(work, ds.get("raw_legible_result", ""))

    # A0: baseline combine
    run_eval(
        "A0 baseline",
        lambda: helpers.process_jersey_id_predictions(str_result_file, useBias=True),
        image_dir, illegible_path, soccer_ball_path, gt_path
    )

    # A1: alignment-only-ish (Top-L with very large L => use almost all frames)
    run_eval(
        "A1 TopL all-frames",
        lambda: helpers.process_jersey_id_predictions_top_L(
            str_result_file, raw_legibility_file, filtered_results_path=filtered_file,
            useBias=True, L=10**9
        ),
        image_dir, illegible_path, soccer_ball_path, gt_path
    )

    # A2: Top-L no thresholds
    _, full_a2, gt = run_eval(
        f"A2 TopL L={args.L}",
        lambda: helpers.process_jersey_id_predictions_top_L(
            str_result_file, raw_legibility_file, filtered_results_path=filtered_file,
            useBias=True, L=args.L, return_debug=True
        ),
        image_dir, illegible_path, soccer_ball_path, gt_path
    )

    # A3: Top-L + thresholds
    run_eval(
        f"A3 TopL L={args.L} + thresholds",
        lambda: helpers.process_jersey_id_predictions_top_L(
            str_result_file, raw_legibility_file, filtered_results_path=filtered_file,
            useBias=True, L=args.L, max_qt_gate=0.15, best_score_gate=-12.0
        ),
        image_dir, illegible_path, soccer_ball_path, gt_path
    )

    # L sweep
    print("\nL sweep:")
    for L in [1, 2, 4, 8, 16]:
        pred, _ = helpers.process_jersey_id_predictions_top_L(
            str_result_file, raw_legibility_file, filtered_results_path=filtered_file,
            useBias=True, L=L
        )
        consolidated = consolidated_results_local(
            image_dir, pred, illegible_path, soccer_ball_list=soccer_ball_path
        )
        m = sliced_metrics(consolidated, gt)
        print(f"  L={L:<2} overall={m['overall'][0]:.4f}% legible={m['legible_only'][0]:.4f}% illegible={m['illegible_only'][0]:.4f}%")

    # Threshold sweeps (separate)
    print("\nmax_qt_gate sweep:")
    for g in [None, 0.05, 0.1, 0.15, 0.2]:
        pred, _ = helpers.process_jersey_id_predictions_top_L(
            str_result_file, raw_legibility_file, filtered_results_path=filtered_file,
            useBias=True, L=args.L, max_qt_gate=g
        )
        consolidated = consolidated_results_local(
            image_dir, pred, illegible_path, soccer_ball_list=soccer_ball_path
        )
        m = sliced_metrics(consolidated, gt)
        print(f"  gate={g} legible={m['legible_only'][0]:.4f}% illegible={m['illegible_only'][0]:.4f}%")

    print("\nbest_score_gate sweep:")
    for g in [None, -18.0, -15.0, -12.0, -10.0, -8.0]:
        pred, _ = helpers.process_jersey_id_predictions_top_L(
            str_result_file, raw_legibility_file, filtered_results_path=filtered_file,
            useBias=True, L=args.L, best_score_gate=g
        )
        consolidated = consolidated_results_local(
            image_dir, pred, illegible_path, soccer_ball_list=soccer_ball_path
        )
        m = sliced_metrics(consolidated, gt)
        print(f"  gate={g} legible={m['legible_only'][0]:.4f}% illegible={m['illegible_only'][0]:.4f}%")

    # Top candidate diagnostics for wrong predictions
    pred_a2, _ = helpers.process_jersey_id_predictions_top_L(
        str_result_file, raw_legibility_file, filtered_results_path=filtered_file,
        useBias=True, L=args.L, return_debug=True
    )
    consolidated_a2 = consolidated_results_local(
        image_dir, pred_a2, illegible_path, soccer_ball_list=soccer_ball_path
    )
    wrong = [k for k in gt.keys() if int(consolidated_a2.get(k, -1)) != int(gt[k])]
    random.shuffle(wrong)
    sample = wrong[:args.wrong_n]
    print(f"\nTop candidate diagnostics (sample wrong={len(sample)}):")
    _, full_dbg = helpers.process_jersey_id_predictions_top_L(
        str_result_file, raw_legibility_file, filtered_results_path=filtered_file,
        useBias=True, L=args.L, return_debug=True
    )
    for t in sample:
        dbg = full_dbg.get(t, {})
        print(
            f"  track={t} gt={gt[t]} pred={consolidated_a2.get(t, -1)} "
            f"max_qt={dbg.get('max_qt', None)} top5={dbg.get('top5', [])}"
        )

    alignment_sanity(str_result_file, raw_legibility_file, filtered_file, sample_n=3)
    compute_qt_match_coverage(str_result_file, raw_legibility_file, filtered_file, gt_path)
    diagnose_L_frame_selection(str_result_file, raw_legibility_file, filtered_file, gt_path, useBias=True, sample_n=5, L_small=1, L_large=16)


if __name__ == "__main__":
    main()

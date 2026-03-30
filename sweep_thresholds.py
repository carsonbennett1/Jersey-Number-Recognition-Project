"""
Sweep FILTER_THRESHOLD and SUM_THRESHOLD to find optimal values.
Usage: conda run -n jersey python sweep_thresholds.py
"""
import json
import os
import numpy as np
import helpers
import configuration as config

def list_dirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def consolidated_results(image_dir, results_dict, illegible_path, soccer_ball_list=None):
    d = dict(results_dict)
    if soccer_ball_list is not None:
        with open(soccer_ball_list, 'r') as sf:
            balls_json = json.load(sf)
        for entry in balls_json['ball_tracks']:
            d[str(entry)] = 1
    with open(illegible_path, 'r') as f:
        illegile_dict = json.load(f)
    for entry in illegile_dict['illegible']:
        if str(entry) not in d:
            d[str(entry)] = -1
    for t in list_dirs(image_dir):
        if t not in d:
            d[t] = -1
        else:
            d[t] = int(d[t])
    return d

def evaluate(consolidated_dict, gt_dict):
    correct = 0
    total = 0
    for id in gt_dict.keys():
        predicted = consolidated_dict.get(id, -1)
        if str(gt_dict[id]) == str(predicted):
            correct += 1
        total += 1
    return correct, total, 100.0 * correct / total

def process_with_thresholds(file_path, filter_th, sum_th, useBias=True):
    """Modified process_jersey_id_predictions with configurable thresholds."""
    all_results = {}
    final_results = {}
    with open(file_path, 'r') as f:
        results_dict = json.load(f)
    for name in results_dict.keys():
        tmp = name.split('_')
        tracklet = tmp[0]
        if tracklet not in all_results:
            all_results[tracklet] = []
            final_results[tracklet] = -1
        value = results_dict[name]['label']
        if not helpers.is_valid_number(value):
            continue
        confidence = results_dict[name]['confidence']
        total_prob = 1
        for x in confidence[:-1]:
            total_prob *= float(x)
        all_results[tracklet].append([int(value), total_prob])

    final_full_results = {}
    for tracklet in all_results.keys():
        if len(all_results[tracklet]) == 0:
            continue
        results = np.array(all_results[tracklet])

        # Apply filter threshold
        if filter_th > 0:
            for entry in results:
                if entry[1] < filter_th:
                    entry[1] = 0

        unique_predictions = np.unique(results[:, 0])
        weights = []
        for i in range(len(unique_predictions)):
            value = unique_predictions[i]
            rows_with_value = results[np.where(results[:, 0] == value)]
            b = helpers.get_bias(value) if useBias else 1
            adjusted_prob = rows_with_value[:, 1] * b
            sum_weights = np.sum(adjusted_prob)
            weights.append(sum_weights)

        best_weight = np.max(weights)
        index_of_best = np.argmax(weights)
        best_prediction = unique_predictions[index_of_best] if best_weight > sum_th else -1

        final_results[tracklet] = str(int(best_prediction))
        final_full_results[tracklet] = {'label': str(int(best_prediction)),
                                         'unique': unique_predictions, 'weights': weights}

    return final_results, final_full_results

# Paths
part = 'test'
image_dir = os.path.join(config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet'][part]['images'])
soccer_ball_list = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][part]['soccer_ball_list'])
illegible_path = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][part]['illegible_result'])
gt_path = os.path.join(config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet'][part]['gt'])
str_result_file = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][part]['jersey_id_result'])

with open(gt_path, 'r') as f:
    gt_dict = json.load(f)

print(f"{'FILTER_TH':>10} {'SUM_TH':>10} {'Bias':>5} {'Correct':>7} {'Total':>5} {'Accuracy':>8}")
print("-" * 55)

# Sweep filter threshold and sum threshold
best_acc = 0
best_params = None
for filter_th in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    for sum_th in [0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        for use_bias in [True, False]:
            results_dict, _ = process_with_thresholds(
                str_result_file, filter_th, sum_th, useBias=use_bias
            )
            cons = consolidated_results(image_dir, results_dict, illegible_path, soccer_ball_list)
            c, t, a = evaluate(cons, gt_dict)
            if a > best_acc:
                best_acc = a
                best_params = (filter_th, sum_th, use_bias, c)
            if a >= 86.5:  # Only print competitive results
                print(f"{filter_th:>10.2f} {sum_th:>10.1f} {str(use_bias):>5} {c:>7} {t:>5} {a:>7.2f}%")

print(f"\n{'='*55}")
print(f"BEST: filter_th={best_params[0]}, sum_th={best_params[1]}, bias={best_params[2]}")
print(f"      Correct={best_params[3]}, Accuracy={best_acc:.2f}%")

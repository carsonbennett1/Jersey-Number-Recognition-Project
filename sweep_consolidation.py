"""
Quick parameter sweep for consolidation methods.
Only re-runs the combine + eval steps (no retraining needed).
Usage: python sweep_consolidation.py
"""
import json
import os
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

# Paths
part = 'test'
image_dir = os.path.join(config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet'][part]['images'])
soccer_ball_list = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][part]['soccer_ball_list'])
illegible_path = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][part]['illegible_result'])
gt_path = os.path.join(config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet'][part]['gt'])
str_result_file = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][part]['jersey_id_result'])

with open(gt_path, 'r') as f:
    gt_dict = json.load(f)

print(f"{'Method':<50} {'Correct':>7} {'Total':>5} {'Accuracy':>8}")
print("-" * 75)

# 1. Original weighted-sum method (baseline)
results_dict, _ = helpers.process_jersey_id_predictions(str_result_file, useBias=True)
cons = consolidated_results(image_dir, results_dict, illegible_path, soccer_ball_list)
c, t, a = evaluate(cons, gt_dict)
print(f"{'Original (weighted-sum, bias=True)':<50} {c:>7} {t:>5} {a:>7.2f}%")

# 2. Original without bias
results_dict, _ = helpers.process_jersey_id_predictions(str_result_file, useBias=False)
cons = consolidated_results(image_dir, results_dict, illegible_path, soccer_ball_list)
c, t, a = evaluate(cons, gt_dict)
print(f"{'Original (weighted-sum, bias=False)':<50} {c:>7} {t:>5} {a:>7.2f}%")

# 3. Bayesian with all options
for useTS in [True, False]:
    for useBias in [True, False]:
        for useTh in [True, False]:
            label = f"Bayesian (TS={useTS}, Bias={useBias}, Th={useTh})"
            results_dict, _ = helpers.process_jersey_id_predictions_bayesian(
                str_result_file, useTS=useTS, useBias=useBias, useTh=useTh
            )
            cons = consolidated_results(image_dir, results_dict, illegible_path, soccer_ball_list)
            c, t, a = evaluate(cons, gt_dict)
            print(f"{label:<50} {c:>7} {t:>5} {a:>7.2f}%")

# 4. Raw probability averaging
results_dict, _ = helpers.process_jersey_id_predictions_raw(str_result_file, useTS=False)
cons = consolidated_results(image_dir, results_dict, illegible_path, soccer_ball_list)
c, t, a = evaluate(cons, gt_dict)
print(f"{'Raw averaging (no TS)':<50} {c:>7} {t:>5} {a:>7.2f}%")

results_dict, _ = helpers.process_jersey_id_predictions_raw(str_result_file, useTS=True)
cons = consolidated_results(image_dir, results_dict, illegible_path, soccer_ball_list)
c, t, a = evaluate(cons, gt_dict)
print(f"{'Raw averaging (TS=True)':<50} {c:>7} {t:>5} {a:>7.2f}%")

# 5. Bayesian with different temperature values
original_TS = helpers.TS
for ts_val in [1.0, 1.5, 2.0, 2.367, 3.0, 4.0, 5.0]:
    helpers.TS = ts_val
    results_dict, _ = helpers.process_jersey_id_predictions_bayesian(
        str_result_file, useTS=True, useBias=True, useTh=False
    )
    cons = consolidated_results(image_dir, results_dict, illegible_path, soccer_ball_list)
    c, t, a = evaluate(cons, gt_dict)
    print(f"{'Bayesian (Bias=True, TS=' + str(ts_val) + ')':<50} {c:>7} {t:>5} {a:>7.2f}%")
helpers.TS = original_TS

import argparse
import os
import sys
import subprocess
import legibility_classifier as lc
import numpy as np
import json
import helpers
from tqdm import tqdm
import configuration as config
from pathlib import Path
import torch
import torch.nn.functional as torchF
from PIL import Image
from jersey_number_dataset import data_transforms
from networks import JerseyNumberMulticlassClassifier

def list_dirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def _run_cmd(args_list, cwd=None):
    """Run command via subprocess (avoids Windows cmd.exe path parsing issues)."""
    try:
        result = subprocess.run(args_list, cwd=cwd or os.getcwd())
        return result.returncode == 0
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return False

def get_soccer_net_legibility_results_with_raw(args, use_filtered=True, filter='gauss', exclude_balls=True,
                                               legibility_threshold=0.5):
    """One forward pass per image: save sigmoid scores (Top-L qt), legible paths, and illegible list.

    Writes raw_legible_result, legible_result, and illegible_result under the SoccerNet working_dir.
    """
    root_dir = config.dataset['SoccerNet']['root_dir']
    image_dir = config.dataset['SoccerNet'][args.part]['images']
    path_to_images = os.path.join(root_dir, image_dir)
    tracklets = list_dirs(path_to_images)
    results_raw = {x: [] for x in tracklets}
    filtered = None

    if use_filtered:
        if filter == 'sim':
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['sim_filtered'])
        else:
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['gauss_filtered'])
        with open(path_to_filter_results, 'r') as f:
            filtered = json.load(f)

    if exclude_balls:
        updated_tracklets = []
        soccer_ball_list = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                        config.dataset['SoccerNet'][args.part]['soccer_ball_list'])
        with open(soccer_ball_list, 'r') as f:
            ball_json = json.load(f)
        ball_list = ball_json['ball_tracks']
        for track in tracklets:
            if track not in ball_list:
                updated_tracklets.append(track)
        tracklets = updated_tracklets

    model_path = config.dataset['SoccerNet']['legibility_model']
    model_arch = config.dataset['SoccerNet']['legibility_model_arch']
    leg_model, leg_device = lc.load_legibility_model(model_path, arch=model_arch)

    legible_tracklets = {}
    illegible_tracklets = []

    for directory in tqdm(tracklets):
        track_dir = os.path.join(path_to_images, directory)
        if use_filtered:
            images = filtered[directory]
        else:
            images = os.listdir(track_dir)
        images_full_path = [os.path.join(track_dir, x) for x in images]
        track_results = lc.run(
            images_full_path, model_path, arch=model_arch, threshold=-1,
            model=leg_model, device=leg_device,
        )
        results_raw[directory] = track_results
        scores = np.asarray(track_results, dtype=np.float64)
        legible = np.nonzero(scores > legibility_threshold)[0]
        if len(legible) == 0:
            illegible_tracklets.append(directory)
        else:
            legible_tracklets[directory] = [images_full_path[i] for i in legible]

    working = config.dataset['SoccerNet']['working_dir']
    part_cfg = config.dataset['SoccerNet'][args.part]
    raw_path = os.path.join(working, part_cfg['raw_legible_result'])
    with open(raw_path, 'w', encoding='utf-8') as outfile:
        json.dump(results_raw, outfile)

    full_legible_path = os.path.join(working, part_cfg['legible_result'])
    with open(full_legible_path, 'w', encoding='utf-8') as outfile:
        outfile.write(json.dumps(legible_tracklets, indent=4))

    full_illegible_path = os.path.join(working, part_cfg['illegible_result'])
    with open(full_illegible_path, 'w', encoding='utf-8') as outfile:
        outfile.write(json.dumps({'illegible': illegible_tracklets}, indent=4))

    return legible_tracklets, illegible_tracklets

def get_soccer_net_legibility_results(args, use_filtered = False, filter = 'sim', exclude_balls=True):
    root_dir = config.dataset['SoccerNet']['root_dir']
    image_dir = config.dataset['SoccerNet'][args.part]['images']
    path_to_images = os.path.join(root_dir, image_dir)
    tracklets = list_dirs(path_to_images)

    if use_filtered:
        if filter == 'sim':
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['sim_filtered'])
        else:
            path_to_filter_results = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                                  config.dataset['SoccerNet'][args.part]['gauss_filtered'])
        with open(path_to_filter_results, 'r') as f:
            filtered = json.load(f)

    legible_tracklets = {}
    illegible_tracklets = []

    if exclude_balls:
        updated_tracklets = []
        soccer_ball_list = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                        config.dataset['SoccerNet'][args.part]['soccer_ball_list'])
        with open(soccer_ball_list, 'r') as f:
            ball_json = json.load(f)
        ball_list = ball_json['ball_tracks']
        for track in tracklets:
            if not track in ball_list:
                updated_tracklets.append(track)
        tracklets = updated_tracklets


    model_path = config.dataset['SoccerNet']['legibility_model']
    model_arch = config.dataset['SoccerNet']['legibility_model_arch']
    leg_model, leg_device = lc.load_legibility_model(model_path, arch=model_arch)

    for directory in tqdm(tracklets):
        track_dir = os.path.join(path_to_images, directory)
        if use_filtered:
            images = filtered[directory]
        else:
            images = os.listdir(track_dir)
        images_full_path = [os.path.join(track_dir, x) for x in images]
        track_results = lc.run(images_full_path, model_path, arch=model_arch, threshold=0.5, model=leg_model, device=leg_device)
        legible = list(np.nonzero(track_results))[0]
        if len(legible) == 0:
            illegible_tracklets.append(directory)
        else:
            legible_images = [images_full_path[i] for i in legible]
            legible_tracklets[directory] = legible_images

    # save results
    json_object = json.dumps(legible_tracklets, indent=4)
    full_legibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['legible_result'])
    with open(full_legibile_path, "w") as outfile:
        outfile.write(json_object)

    full_illegibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['illegible_result'])
    json_object = json.dumps({'illegible': illegible_tracklets}, indent=4)
    with open(full_illegibile_path, "w") as outfile:
        outfile.write(json_object)

    return legible_tracklets, illegible_tracklets


def generate_json_for_pose_estimator(args, legible = None):
    all_files = []
    if not legible is None:
        for key in legible.keys():
            for entry in legible[key]:
                all_files.append(os.path.join(os.getcwd(), entry))
    else:
        root_dir = os.path.join(os.getcwd(), config.dataset['SoccerNet']['root_dir'])
        image_dir = config.dataset['SoccerNet'][args.part]['images']
        path_to_images = os.path.join(root_dir, image_dir)
        tracks = list_dirs(path_to_images)
        for tr in tracks:
            track_dir = os.path.join(path_to_images, tr)
            imgs = [f for f in os.listdir(track_dir) if not f.startswith('.')]
            for img in imgs:
                all_files.append(os.path.join(track_dir, img))

    output_json = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['pose_input_json'])
    helpers.generate_json(all_files, output_json)


def consolidated_results(image_dir, dict, illegible_path, soccer_ball_list=None):
    if not soccer_ball_list is None:
        with open(soccer_ball_list, 'r') as sf:
            balls_json = json.load(sf)
        balls_list = balls_json['ball_tracks']
        for entry in balls_list:
            dict[str(entry)] = 1

    with open(illegible_path, 'r') as f:
        illegile_dict = json.load(f)
    all_illegible = illegile_dict['illegible']
    for entry in all_illegible:
        if not str(entry) in dict.keys():
            dict[str(entry)] = -1

    all_tracks = list_dirs(image_dir)
    for t in all_tracks:
        if not t in dict.keys():
            dict[t] = -1
        else:
            dict[t] = int(dict[t])
    return dict

class _CropDataset(torch.utils.data.Dataset):
    """Minimal dataset for loading crop images by filename."""
    def __init__(self, imgs_dir, filenames, transform):
        self.imgs_dir = imgs_dir
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.imgs_dir, self.filenames[idx])).convert('RGB')
        return self.transform(image), idx

def run_multitask_inference(crops_dir, result_file, model_path):
    """Run multi-task ResNet classifier on torso crops, output PARSeq-compatible JSON."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else
                          "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
                          "cpu")
    model = JerseyNumberMulticlassClassifier()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    transform = data_transforms['test']['resnet']
    imgs_dir = os.path.join(crops_dir, 'imgs')
    filenames = sorted(os.listdir(imgs_dir))
    results = {}

    batch_size = 32 if device.type in ('cuda', 'mps') else 1
    num_workers = 0 if os.name == 'nt' else 4

    dataset = _CropDataset(imgs_dir, filenames, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    with torch.no_grad():
        for batch_images, batch_indices in tqdm(dataloader, desc="Multi-task inference"):
            batch_images = batch_images.to(device)
            h1, h2, h3, h4 = model(batch_images)
            probs = torchF.softmax(h1, dim=1)

            for i in range(len(batch_indices)):
                idx = batch_indices[i].item()
                filename = filenames[idx]
                img_probs = probs[i]
                pred_class = int(torch.argmax(img_probs).item())
                pred_conf = float(img_probs[pred_class].item())
                label = str(pred_class) if pred_class > 0 else "-1"
                results[filename] = {
                    'label': label,
                    'confidence': [pred_conf, 1.0],
                    'raw': img_probs.cpu().tolist(),
                    'logits': h1[i].cpu().tolist()
                }

    with open(result_file, 'w') as f:
        json.dump(results, f)
    print(f"Multi-task inference complete: {len(results)} predictions saved")
    return True

def _parseq_training_lmdb_ready(data_root: str) -> bool:
    """PARSeq expects **/data.mdb under train/real or train (case variants for splits)."""
    root = Path(data_root)
    if not root.is_dir():
        return False
    for rel in ('train/real', 'train', 'Train/real', 'Train'):
        p = root / rel
        if p.is_dir() and any(p.rglob('data.mdb')):
            return True
    return False


def train_parseq(args):
    parseq_dir = config.str_home
    current_dir = os.getcwd()
    experiment = 'parseq-jersey-aux' if getattr(args, 'jersey_aux_str', False) else 'parseq'
    print(f"PARSeq Hydra experiment: {experiment}")
    if args.dataset == 'Hockey':
        print("Train PARSeq for Hockey")
    else:
        print("Train PARSeq for Soccer")
    env_lm = os.environ.get('JERSEY_STR_LMDB_ROOT', '').strip()
    if env_lm:
        data_root = str(Path(env_lm).resolve() if os.path.isabs(env_lm) else (Path(current_dir) / env_lm).resolve())
        print(f"Using JERSEY_STR_LMDB_ROOT -> {data_root}")
    elif args.dataset == 'Hockey':
        data_root = os.path.join(current_dir, config.dataset['Hockey']['root_dir'], config.dataset['Hockey']['numbers_data'])
    else:
        data_root = os.path.join(current_dir, config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet']['numbers_data'])
    if not _parseq_training_lmdb_ready(data_root):
        print('\n' + '=' * 72)
        if args.dataset == 'SoccerNet':
            print('PARSeq STR training needs a separate SoccerNet jersey-number LMDB (cropped digits).')
            print('The main jersey-2023 image folders are not enough — you must download the LMDB pack.')
            print('README → Data → "Weakly-labelled jersey number crops ... LMDB" (Google Drive).')
        else:
            print('PARSeq STR training needs the Hockey jersey-numbers LMDB under your data/Hockey tree.')
        print(f'Expected: `data.mdb` under train/ or train/real/ inside:\n  {data_root}')
        print('If the LMDB lives elsewhere, set JERSEY_STR_LMDB_ROOT to that folder (absolute or relative to cwd).')
        print('=' * 72 + '\n')
        return
    success = _run_cmd([
        config.str_python, 'train.py',
        f'+experiment={experiment}', 'dataset=real', f'data.root_dir={data_root}',
        'trainer.max_epochs=25', 'pretrained=parseq', 'trainer.devices=1',
        # float 1.0 = validate once per epoch; int 1 would validate after every batch (very slow)
        'trainer.val_check_interval=1.0', 'data.batch_size=128', 'data.max_label_length=2'
    ], cwd=parseq_dir)
    print("Done training")


def hockey_pipeline(args):
    # actions = {"legible": True,
    #            "pose": False,
    #            "crops": False,
    #            "str": True}
    success = True
    # test legibility classification
    _project_root = os.path.dirname(os.path.abspath(__file__))
    if args.pipeline['legible']:
        root_dir = os.path.join(config.dataset["Hockey"]["root_dir"], config.dataset["Hockey"]["legibility_data"])
        print("Test legibility classifier")
        legibility_script = os.path.join(_project_root, 'legibility_classifier.py')
        success = _run_cmd([
            sys.executable, legibility_script, '--data', root_dir,
            '--arch', 'resnet34', '--trained_model', config.dataset['Hockey']['legibility_model']
        ], cwd=_project_root)
        print("Done legibility classifier")

    if success and args.pipeline['str']:
        print("Predict numbers")
        data_root = os.path.join(_project_root, config.dataset['Hockey']['root_dir'], config.dataset['Hockey']['numbers_data'])
        str_script = os.path.join(_project_root, 'str.py')
        success = _run_cmd([
            config.str_python, str_script, config.dataset["Hockey"]["str_model"],
            '--data_root', data_root
        ], cwd=_project_root)
        print("Done predict numbers")

def _output_exists(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0

def soccer_net_pipeline(args):
    legible_dict = None
    legible_results = None
    consolidated_dict = None
    Path(config.dataset['SoccerNet']['working_dir']).mkdir(parents=True, exist_ok=True)
    success = True
    _project_root = os.path.dirname(os.path.abspath(__file__))
    if getattr(config, 'pipeline_output_slug', ''):
        print(f"Isolated pipeline output (branch run): {config.dataset['SoccerNet']['working_dir']}")
    else:
        print(f"Pipeline output directory: {config.dataset['SoccerNet']['working_dir']}")

    image_dir = os.path.join(config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet'][args.part]['images'])
    soccer_ball_list = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                      config.dataset['SoccerNet'][args.part]['soccer_ball_list'])
    features_dir = config.dataset['SoccerNet'][args.part]['feature_output_folder']
    full_legibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                      config.dataset['SoccerNet'][args.part]['legible_result'])
    illegible_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                  config.dataset['SoccerNet'][args.part]['illegible_result'])
    gt_path = os.path.join(config.dataset['SoccerNet']['root_dir'], config.dataset['SoccerNet'][args.part]['gt'])

    input_json = os.path.join(config.dataset['SoccerNet']['working_dir'],
                              config.dataset['SoccerNet'][args.part]['pose_input_json'])
    output_json = os.path.join(config.dataset['SoccerNet']['working_dir'],
                               config.dataset['SoccerNet'][args.part]['pose_output_json'])

    gauss_filtered_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                       config.dataset['SoccerNet'][args.part]['gauss_filtered'])
    
    raw_legibile_path = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                     config.dataset['SoccerNet'][args.part]['raw_legible_result'])

    # 1. Filter out soccer ball based on images size
    if args.pipeline['soccer_ball_filter']:
        if _output_exists(soccer_ball_list):
            print("Determine soccer ball: SKIPPED (output exists)")
        else:
            print("Determine soccer ball")
            success = helpers.identify_soccer_balls(image_dir, soccer_ball_list)
            print("Done determine soccer ball")

    # 2. generate and store features for each image in each tracklet
    if args.pipeline['feat']:
        print("Generate features")
        # Use subprocess (avoids Windows "filename/directory syntax incorrect" from os.system + cmd.exe path parsing)
        if not os.path.isfile(config.reid_script_path):
            print(f"ERROR: centroid_reid.py not found at {config.reid_script_path}. Ensure project is complete.")
            success = False
        else:
            success = _run_cmd([
                sys.executable, config.reid_script_path,
                '--tracklets_folder', image_dir,
                '--output_folder', features_dir
            ], cwd=_project_root)
        print("Done generating features")

    # 3. identify and remove outliers based on features
    if args.pipeline['filter'] and success:
        if _output_exists(gauss_filtered_path):
            print("Identify and remove outliers: SKIPPED (output exists)")
        else:
            print("Identify and remove outliers")
            gaussian_script = os.path.join(_project_root, 'gaussian_outliers.py')
            success = _run_cmd([
                sys.executable, gaussian_script,
                '--tracklets_folder', image_dir,
                '--output_folder', features_dir
            ], cwd=_project_root)
            print("Done removing outliers")

    # 4. pass all images through legibility classifier and record results
    if args.pipeline['legible'] and success:
        if (_output_exists(full_legibile_path) and _output_exists(illegible_path)
                and _output_exists(raw_legibile_path)):
            print("Classifying Legibility: SKIPPED (output exists)")
            with open(full_legibile_path, 'r') as f:
                legible_dict = json.load(f)
        else:
            print("Classifying Legibility:")
            try:
                legible_dict, illegible_tracklets = get_soccer_net_legibility_results_with_raw(
                    args, use_filtered=True, filter='gauss', exclude_balls=True)
            except Exception as error:
                print(f'Failed to run legibility classifier:{error}')
                success = False
            print("Done classifying legibility")

    # 4.5 evaluate tracklet legibility results
    if args.pipeline['legible_eval'] and success:
        print("Evaluate Legibility results:")
        try:
            if legible_dict is None:
                 with open(full_legibile_path, 'r') as openfile:
                    legible_dict = json.load(openfile)

            helpers.evaluate_legibility(gt_path, illegible_path, legible_dict, soccer_ball_list=soccer_ball_list)
        except Exception as e:
            print(e)
            success = False
        print("Done evaluating legibility")


    # 5. generate json for pose-estimation + run pose
    if args.pipeline['pose'] and success:
        if _output_exists(output_json):
            print("Generating json for pose: SKIPPED (output exists)")
            print("Detecting pose: SKIPPED (output exists)")
        else:
            print("Generating json for pose")
            try:
                if legible_dict is None:
                    with open(full_legibile_path, 'r') as openfile:
                        legible_dict = json.load(openfile)
                generate_json_for_pose_estimator(args, legible = legible_dict)
            except Exception as e:
                print(e)
                success = False
            print("Done generating json for pose")

            if success:
                print("Detecting pose")
                pose_config = os.path.join(config.pose_home, 'configs', 'body', '2d_kpt_sview_rgb_img', 'topdown_heatmap', 'coco', 'ViTPose_huge_coco_256x192.py')
                pose_checkpoint = os.path.join(config.pose_home, 'checkpoints', 'vitpose-h.pth')
                pose_script = os.path.join(_project_root, 'pose.py')
                pose_py = config.pose_python
                if not os.path.isfile(pose_py):
                    print(
                        f"ERROR: Pose interpreter not found: {pose_py}\n"
                        f"Create the '{config.pose_env}' conda environment (mmcv + ViTPose + addict), e.g. run: python setup.py"
                    )
                    success = False
                else:
                    success = _run_cmd([
                        pose_py, pose_script,
                        pose_config, pose_checkpoint,
                        '--img-root', '/', '--json-file', input_json, '--out-json', output_json
                    ], cwd=_project_root)
                print("Done detecting pose")


    # 6. generate cropped images
    if args.pipeline['crops'] and success:
        crops_destination_dir = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['crops_folder'], 'imgs')
        if os.path.isdir(crops_destination_dir) and len(os.listdir(crops_destination_dir)) > 0:
            print(f"Generate crops: SKIPPED ({len(os.listdir(crops_destination_dir))} crop files already exist)")
        else:
            print("Generate crops")
            try:
                Path(crops_destination_dir).mkdir(parents=True, exist_ok=True)
                if legible_results is None:
                    with open(full_legibile_path, "r") as outfile:
                        legible_results = json.load(outfile)
                helpers.generate_crops(output_json, crops_destination_dir, legible_results)
            except Exception as e:
                print(e)
                success = False
            print("Done generating crops")

    str_result_file = os.path.join(config.dataset['SoccerNet']['working_dir'],
                                   config.dataset['SoccerNet'][args.part]['jersey_id_result'])
    # 7. run STR system on all crops
    if args.pipeline['str'] and success:
        if _output_exists(str_result_file):
            print("Predict numbers: SKIPPED (output exists)")
        else:
            crops_dir = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['crops_folder'])
            if args.pipeline.get('use_multitask_classifier', False):
                # Approach 1: Multi-task ResNet classifier (runs in-process)
                print("Predict numbers (multi-task classifier)")
                multitask_model_path = args.pipeline.get('multitask_model_path',
                    os.path.join(_project_root, 'experiments', 'multitask_resnet34.pth'))
                success = run_multitask_inference(crops_dir, str_result_file, multitask_model_path)
            else:
                # Original PARSeq STR (subprocess; interpreter from configuration.str_python / JERSEY_STR_ENV)
                print("Predict numbers (PARSeq)")
                str_script = os.path.join(_project_root, 'str.py')
                str_py = config.str_python
                if not os.path.isfile(str_py):
                    print(
                        f"ERROR: STR interpreter not found: {str_py}\n"
                        f"Install PARSeq deps in the '{config.str_env}' env or set JERSEY_STR_ENV, e.g. parseq2. "
                        f"See README / python setup.py"
                    )
                    success = False
                else:
                    success = _run_cmd([
                        str_py, str_script, config.dataset["SoccerNet"]["str_model"],
                        '--data_root', crops_dir, '--inference', '--result_file', str_result_file
                    ], cwd=_project_root)
            print("Done predict numbers")

    final_results_path = os.path.join(config.dataset['SoccerNet']['working_dir'], config.dataset['SoccerNet'][args.part]['final_result'])

    # 8. combine tracklet results
    analysis_results = None  # Initialize to avoid UnboundLocalError when skipped
    if args.pipeline['combine'] and success:
        if _output_exists(final_results_path):
            print("Combine results: SKIPPED (output exists)")
            with open(final_results_path, 'r') as f:
                consolidated_dict = json.load(f)
        else:
            if args.pipeline.get('combine_bayesian'):
                results_dict, analysis_results = helpers.process_jersey_id_predictions(
                    str_result_file, useBias=True)
            else:
                results_dict, analysis_results = helpers.process_jersey_id_predictions_top_L(
                    str_result_file,
                    raw_legibility_path=raw_legibile_path,
                    filtered_results_path=gauss_filtered_path,
                    useBias=True,
                )

            consolidated_dict = consolidated_results(image_dir, results_dict, illegible_path, soccer_ball_list=soccer_ball_list)

            with open(final_results_path, 'w') as f:
                json.dump(consolidated_dict, f)

    # 9. evaluate accuracy
    if args.pipeline['eval'] and success:
        if consolidated_dict is None:
            with open(final_results_path, 'r') as f:
                consolidated_dict = json.load(f)
        with open(gt_path, 'r') as gf:
            gt_dict = json.load(gf)
        print(len(consolidated_dict.keys()), len(gt_dict.keys()))
        helpers.evaluate_results(consolidated_dict, gt_dict, full_results = analysis_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help="Options: 'SoccerNet', 'Hockey'")
    parser.add_argument('part', help="Options: 'test', 'val', 'train', 'challenge")
    parser.add_argument('--train_str', action='store_true', default=False, help="Run training of jersey number recognition")
    parser.add_argument('--jersey_aux_str', action='store_true', default=False,
                        help="With --train_str: enable PARSeq encoder multi-task aux loss (parseq-jersey-aux); inference still PARSeq decode")
    parser.add_argument('--combine_bayesian', action='store_true', default=False,
                        help="SoccerNet: use Bayesian combine instead of Top-L (default: Top-L + raw legibility)")
    args = parser.parse_args()

    if not args.train_str:
        if args.dataset == 'SoccerNet':
            actions = {"soccer_ball_filter": True,
                       "feat": True,
                       "filter": True,
                       "legible": True,
                       "legible_eval": True,
                       "pose": True,
                       "crops": True,
                       "str": True,
                       "combine": True,
                       "eval": True,
                       "use_multitask_classifier": False,
                       "multitask_model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "multitask_resnet34.pth"),
                       "combine_bayesian": args.combine_bayesian}
            args.pipeline = actions
            soccer_net_pipeline(args)
        elif args.dataset == 'Hockey':
            actions = {"legible": True,
                       "str": True}
            args.pipeline = actions
            hockey_pipeline(args)
        else:
            print("Unknown dataset")
    else:
        train_parseq(args)



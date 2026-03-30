import os
import sys

# Detect the project root - derive from this file's location
_main_repo = os.environ.get('JERSEY_PROJECT_ROOT')
if not _main_repo:
    _main_repo = os.path.dirname(os.path.abspath(__file__))

# Detect conda base - allow override via environment variable
_conda_base = os.environ.get('JERSEY_CONDA_BASE')
if not _conda_base:
    _home = os.path.expanduser('~')
    # Try common conda locations (miniconda3, anaconda3, conda3)
    for _conda_name in ('miniconda3', 'anaconda3', 'conda3'):
        _candidate = os.path.join(_home, _conda_name, 'envs')
        if os.path.isdir(_candidate):
            _conda_base = _candidate
            break
    else:
        _conda_base = os.path.join(_home, 'miniconda3', 'envs')

# Determine Python executable name based on OS
_python_exe = 'python.exe' if os.name == 'nt' else 'python'
_python_bin_dir = '' if os.name == 'nt' else 'bin'

# Helper to build Python executable path for each environment
def _get_python_path(env_name):
    if os.name == 'nt':
        return os.path.join(_conda_base, env_name, _python_exe)
    else:
        return os.path.join(_conda_base, env_name, _python_bin_dir, _python_exe)

pose_home = os.path.join(_main_repo, 'pose/ViTPose')
# Prefer vitpose2 (current setup.py); fall back to vitpose for older installs
_pose_env_candidates = ('vitpose2', 'vitpose')
pose_env = 'vitpose2'
pose_python = _get_python_path(pose_env)
if not os.path.isfile(pose_python):
    for _alt in _pose_env_candidates[1:]:
        _p = _get_python_path(_alt)
        if os.path.isfile(_p):
            pose_env = _alt
            pose_python = _p
            break

str_home = os.path.join(_main_repo, 'str/parseq/')
str_env = 'parseq2'
str_python = _get_python_path(str_env)
str_platform = 'cu113'

# centroids
reid_env = 'centroids'
reid_python = _get_python_path('jersey')
reid_script = 'centroid_reid.py'
# Full path to reid script - use this when spawning subprocesses to avoid CWD/path issues across devices
reid_script_path = os.path.join(_main_repo, 'centroid_reid.py')

reid_home = os.path.join(_main_repo, 'reid/')


dataset = {'SoccerNet':
                {'root_dir': os.path.join(_main_repo, 'data/SoccerNet/jersey-2023'),
                 'working_dir': os.path.join(_main_repo, 'out/SoccerNetResults'),
                 'test': {
                        'images': 'test/test/images',
                        'gt': 'test/test/test_gt.json',
                        'feature_output_folder': os.path.join(_main_repo, 'out/SoccerNetResults/test'),
                        'illegible_result': 'illegible.json',
                        'soccer_ball_list': 'soccer_ball.json',
                        'sim_filtered': 'test/main_subject_0.4.json',
                        'gauss_filtered': 'test/main_subject_gauss_th=3.5_r=3.json',
                        'legible_result': 'legible.json',
                        'raw_legible_result': 'raw_legible_resnet34.json',
                        'pose_input_json': 'pose_input.json',
                        'pose_output_json': 'pose_results.json',
                        'crops_folder': 'crops',
                        'jersey_id_result': 'jersey_id_results.json',
                        'final_result': 'final_results.json'
                    },
                 'val': {
                        'images': 'val/val/images',
                        'gt': 'val/val/val_gt.json',
                        'feature_output_folder': os.path.join(_main_repo, 'out/SoccerNetResults/val'),
                        'illegible_result': 'illegible_val.json',
                        'legible_result': 'legible_val.json',
                        'soccer_ball_list': 'soccer_ball_val.json',
                        'crops_folder': 'crops_val',
                        'sim_filtered': 'val/main_subject_0.4.json',
                        'gauss_filtered': 'val/main_subject_gauss_th=3.5_r=3.json',
                        'pose_input_json': 'pose_input_val.json',
                        'pose_output_json': 'pose_results_val.json',
                        'jersey_id_result': 'jersey_id_results_validation.json'
                    },
                 'train': {
                     'images': 'train/train/images',
                     'gt': 'train/train/train_gt.json',
                     'feature_output_folder': os.path.join(_main_repo, 'out/SoccerNetResults/train'),
                     'illegible_result': 'illegible_train.json',
                     'legible_result': 'legible_train.json',
                     'soccer_ball_list': 'soccer_ball_train.json',
                     'sim_filtered': 'train/main_subject_0.4.json',
                     'gauss_filtered': 'train/main_subject_gauss_th=3.5_r=3.json',
                     'pose_input_json': 'pose_input_train.json',
                     'pose_output_json': 'pose_results_train.json',
                     'raw_legible_result': 'train_raw_legible_combined.json'
                 },
                 'challenge': {
                        'images': 'challenge/challenge/images',
                        'feature_output_folder': os.path.join(_main_repo, 'out/SoccerNetResults/challenge'),
                        'gt': '',
                        'illegible_result': 'challenge_illegible.json',
                        'soccer_ball_list': 'challenge_soccer_ball.json',
                        'sim_filtered': 'challenge/main_subject_0.4.json',
                        'gauss_filtered': 'challenge/main_subject_gauss_th=3.5_r=3.json',
                        'legible_result': 'challenge_legible.json',
                        'pose_input_json': 'challenge_pose_input.json',
                        'pose_output_json': 'challenge_pose_results.json',
                        'crops_folder': 'challenge_crops',
                        'jersey_id_result': 'challenge_jersey_id_results.json',
                        'final_result': 'challenge_final_results.json',
                        'raw_legible_result': 'challenge_raw_legible_vit.json'
                 },
                 'numbers_data': 'lmdb',

                 'legibility_model': os.path.join(_main_repo, "models/legibility_resnet34_soccer_20240215.pth"),
                 'legibility_model_arch': "resnet34",

                 'legibility_model_url':  "https://drive.google.com/uc?id=18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw",
                 'pose_model_url': 'https://drive.google.com/uc?id=1A3ftF118IcxMn_QONndR-8dPWpf7XzdV',
                 'str_model': os.path.join(_main_repo, 'str/parseq/outputs/parseq/2026-03-29_14-02-59/checkpoints/epoch=11-step=1236-val_accuracy=95.6731-val_NED=96.4629.ckpt'),

                 #'str_model': 'pretrained=parseq',
                 'str_model_url': "https://drive.google.com/uc?id=1uRln22tlhneVt3P6MePmVxBWSLMsL3bm",
                },
           "Hockey": {
                 'root_dir': os.path.join(_main_repo, 'data/Hockey'),
                 'legibility_data': 'legibility_dataset',
                 'numbers_data': 'jersey_number_dataset/jersey_numbers_lmdb',
                 'legibility_model':  os.path.join(_main_repo, 'models/legibility_resnet34_hockey_20240201.pth'),
                 'legibility_model_url':  "https://drive.google.com/uc?id=1RfxINtZ_wCNVF8iZsiMYuFOP7KMgqgDp",
                 'str_model': os.path.join(_main_repo, 'models/parseq_epoch=3-step=95-val_accuracy=98.7903-val_NED=99.3952.ckpt'),
                 'str_model_url': "https://drive.google.com/uc?id=1FyM31xvSXFRusN0sZH0EWXoHwDfB9WIE",
            }
        }
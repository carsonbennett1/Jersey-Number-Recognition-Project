import os
import shutil
import configuration as cfg
import json
import urllib.request
import gdown
import argparse


def check_conda():
    """Verify conda is installed and in PATH. Exit with helpful message if not."""
    conda_path = shutil.which("conda")
    if conda_path is None:
        print("\n" + "=" * 60)
        print("ERROR: Conda is required but not found in your PATH.")
        print("=" * 60)
        print("\nThis setup script uses Conda to create separate environments for:")
        print("  - ViTPose (pose estimation)")
        print("  - PARSeq (scene text recognition)")
        print("  - Centroid-ReID (re-identification)")
        print("\nPlease install Miniconda or Anaconda, then:")
        print("  1. Restart your terminal (or run 'conda init' and restart)")
        print("  2. Ensure conda is in PATH")
        print("\nDownload Miniconda: https://docs.conda.io/en/latest/miniconda.html")
        print("=" * 60 + "\n")
        raise SystemExit(1)


###### common setup utils ##############

def make_conda_env(env_name, libs=""):
    os.system(f"conda create -n {env_name} -y "+libs)

def activate_conda_env(env_name):
    os.system(f"conda activate {env_name}")

def deactivate_conda_env(env_name):
    os.system(f"conda deactivate")

def conda_pyrun(env_name, exec_file, args):
    os.system(f"conda run -n {env_name} --live-stream python3 \"{exec_file}\" '{json.dumps(dict(vars(args)))}'")


def get_conda_envs():
    """Get list of conda environment names. Returns empty list if conda fails."""
    stream = os.popen("conda env list")
    output = stream.read()
    if not output or "conda" not in output.lower():
        return []
    a = output.split()
    for item in ["*", "#", "conda", "environments:"]:
        if item in a:
            a.remove(item)
    # Remove duplicate "#" if present
    while "#" in a:
        a.remove("#")
    return a[::2] if a else []
###########################################


def setup_reid(root):
    os.chdir(root)
    env_name  = cfg.reid_env
    repo_name = "centroids-reid"
    src_url   = "https://github.com/mikwieczorek/centroids-reid.git"
    rep_path  = "./reid"

    if not repo_name in os.listdir(rep_path):
        # clone source repo
        os.system(f"git clone --recurse-submodules {src_url} {os.path.join(rep_path, repo_name)}")

    # create models folder and download weights (always check - may be missing if repo was cloned earlier)
    models_folder_path = os.path.join(root, "reid", repo_name, "models")
    os.makedirs(models_folder_path, exist_ok=True)
    market_path = os.path.join(models_folder_path, "market1501_resnet50_256_128_epoch_120.ckpt")
    duke_path = os.path.join(models_folder_path, "dukemtmcreid_resnet50_256_128_epoch_120.ckpt")
    if not os.path.isfile(market_path) or os.path.getsize(market_path) < 10_000_000:
        if os.path.isfile(market_path):
            os.remove(market_path)
        gdown.download("https://drive.google.com/uc?id=1ZFywKEytpyNocUQd2APh2XqTe8X0HMom", market_path)
    if not os.path.isfile(duke_path) or os.path.getsize(duke_path) < 10_000_000:
        if os.path.isfile(duke_path):
            os.remove(duke_path)
        gdown.download("https://drive.google.com/uc?id=1w9yzdP_5oJppGIM4gs3cETyLujanoHK8", duke_path)

    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.8")
        cwd = os.getcwd()
        os.chdir(os.path.join(rep_path, repo_name))
        os.system(f"conda run --live-stream -n {env_name} conda install --name {env_name} pip")
        os.system(f"conda run --live-stream -n {env_name} pip install -r requirements.txt")

        os.chdir(cwd)

# clone and install vitpose
# download the model
def setup_pose(root):
    env_name  = cfg.pose_env
    repo_name = "ViTPose"
    src_url   = "https://github.com/ViTAE-Transformer/ViTPose.git"
    rep_path  = "./pose"
    os.chdir(root)

    if not repo_name in os.listdir(rep_path):
       # clone source repo
        os.chdir(root)
        os.system(f"git clone --recurse-submodules {src_url} {os.path.join(rep_path,repo_name)}")

    os.chdir(root)
    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.8")

        os.system(f"conda run --live-stream -n {env_name} conda install --name {env_name} pip")
        os.system(f"conda run --live-stream -n {env_name} pip install  mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html")

        os.chdir(os.path.join(root, rep_path, "ViTPose"))
        os.system(f"conda run --live-stream -n {env_name} pip install -v -e .")
        os.system(f"conda run --live-stream -n {env_name} pip install timm==0.4.9 einops")


# clone and install str
# download the model
def setup_str(root):
    env_name  = cfg.str_env
    repo_name = "parseq"
    src_url   = "https://github.com/baudm/parseq.git"
    rep_path  = "./str"
    os.chdir(root)

    if not repo_name in os.listdir(rep_path):
        # clone source repo
        os.system(f"git clone --recurse-submodules {src_url} {os.path.join(rep_path, repo_name)}")

    parseq_dir = os.path.join(root, rep_path, repo_name)
    os.chdir(parseq_dir)

    # Create core.cu117.txt from core.txt (equivalent to: make torch-cu117)
    # Windows doesn't have make; the Makefile does: sed 's|cpu|cu117|' core.txt > core.cu117.txt
    core_cu117 = os.path.join(parseq_dir, "requirements", "core.cu117.txt")
    if not os.path.isfile(core_cu117):
        core_txt = os.path.join(parseq_dir, "requirements", "core.txt")
        if os.path.isfile(core_txt):
            with open(core_txt, "r") as f:
                content = f.read()
            with open(core_cu117, "w") as f:
                f.write(content.replace("cpu", "cu117"))

    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.9")
        req_file = core_cu117 if os.path.isfile(core_cu117) else os.path.join(parseq_dir, "requirements", "core.txt")
        os.system(f"conda run --live-stream -n {env_name} conda install --name {env_name} pip")
        os.system(f'conda run --live-stream -n {env_name} pip install -r "{req_file}" -e .[train,test]')

    os.chdir(root)

def download_models_common(root_dir):
    os.chdir(root_dir)
    repo_name = "ViTPose"
    rep_path = "./pose"

    url = cfg.dataset['SoccerNet']['pose_model_url']
    models_folder_path = os.path.join(rep_path, repo_name, "checkpoints")
    os.makedirs(models_folder_path, exist_ok=True)
    save_path = os.path.join(rep_path, "ViTPose", "checkpoints", "vitpose-h.pth")
    if not os.path.isfile(save_path):
        gdown.download(url, save_path)

def download_models(root_dir, dataset):
    os.chdir(root_dir)
    # download and save fine-tuned model
    save_path = os.path.join(root_dir, cfg.dataset[dataset]['str_model'])
    if not os.path.isfile(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        source_url = cfg.dataset[dataset]['str_model_url']
        gdown.download(source_url, save_path)

    save_path = os.path.join(root_dir, cfg.dataset[dataset]['legibility_model'])
    if not os.path.isfile(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        source_url = cfg.dataset[dataset]['legibility_model_url']
        gdown.download(source_url, save_path)

def setup_sam(root_dir):
    os.chdir(root_dir)
    repo_name = 'sam2'
    src_url = 'https://github.com/davda54/sam'

    if not repo_name in os.listdir(root_dir):
        # clone source repo
        os.system(f"git clone --recurse-submodules {src_url} {os.path.join(root_dir, repo_name)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?', default='all', help="Options: all, SoccerNet, Hockey")

    args = parser.parse_args()

    # Verify conda is available before starting
    check_conda()

    root_dir = os.getcwd()

    # common for both datasets
    setup_sam(root_dir)
    setup_pose(root_dir)
    download_models_common(root_dir)
    setup_str(root_dir)

    #SoccerNet only
    if not args.dataset == 'Hockey':
        setup_reid(root_dir)
        download_models(root_dir, 'SoccerNet')

    if not args.dataset == 'SoccerNet':
        download_models(root_dir, 'Hockey')

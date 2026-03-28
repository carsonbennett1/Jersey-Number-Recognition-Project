"""Sweep all aux_alpha presets for PARSeq jersey-aux training.

Trains PARSeq with each preset from aux_alpha_presets.yaml sequentially,
saves top-3 checkpoints per run, then ranks all checkpoints globally
and reports the top-5 best for pipeline evaluation.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent
_PARSEQ_DIR = _REPO_ROOT / 'str' / 'parseq'
_OUTPUTS_DIR = _PARSEQ_DIR / 'outputs' / 'parseq'
_DEFAULT_PRESETS = _PARSEQ_DIR / 'configs' / 'aux_alpha_presets.yaml'
_CKPT_PATTERN = re.compile(r'val_accuracy=(\d+\.\d+)')
_EPOCH_PATTERN = re.compile(r'epoch=(\d+)')

# ---------------------------------------------------------------------------
# Configuration (mirrors main.py / configuration.py)
# ---------------------------------------------------------------------------

def _get_str_python() -> str:
    sys.path.insert(0, str(_REPO_ROOT))
    import configuration as config
    return config.str_python


def _get_data_root() -> str:
    env_lm = os.environ.get('JERSEY_STR_LMDB_ROOT', '').strip()
    if env_lm:
        return str(Path(env_lm).resolve() if os.path.isabs(env_lm) else (Path.cwd() / env_lm).resolve())
    sys.path.insert(0, str(_REPO_ROOT))
    import configuration as config
    return os.path.join(
        str(_REPO_ROOT),
        config.dataset['SoccerNet']['root_dir'],
        config.dataset['SoccerNet']['numbers_data'],
    )

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _existing_run_dirs() -> set:
    if not _OUTPUTS_DIR.exists():
        return set()
    return {d.name for d in _OUTPUTS_DIR.iterdir() if d.is_dir()}


def _train_one_preset(preset: dict, str_python: str, data_root: str, max_epochs: int) -> str | None:
    """Launch a single PARSeq training run. Returns the new output directory name or None on failure."""
    before = _existing_run_dirs()

    cmd = [
        str_python, 'train.py',
        '+experiment=parseq-jersey-aux', 'dataset=real',
        f'data.root_dir={data_root}',
        f'trainer.max_epochs={max_epochs}', 'pretrained=parseq',
        'trainer.devices=1', 'trainer.val_check_interval=1.0',
        'data.batch_size=128', 'data.max_label_length=2',
        f'model.aux_alpha_full={preset["aux_alpha_full"]}',
        f'model.aux_alpha_tens={preset["aux_alpha_tens"]}',
        f'model.aux_alpha_units={preset["aux_alpha_units"]}',
        f'model.aux_alpha_count={preset["aux_alpha_count"]}',
    ]

    print(f'\n{"=" * 72}')
    print(f'Preset {preset["id"]}: {preset["name"]}')
    print(f'  aux_alpha_full={preset["aux_alpha_full"]}, tens={preset["aux_alpha_tens"]}, '
          f'units={preset["aux_alpha_units"]}, count={preset["aux_alpha_count"]}')
    print(f'{"=" * 72}')

    try:
        result = subprocess.run(cmd, cwd=str(_PARSEQ_DIR))
        if result.returncode != 0:
            print(f'WARNING: training returned exit code {result.returncode} for preset {preset["id"]}')
    except FileNotFoundError as exc:
        print(f'ERROR: {exc}')
        return None

    after = _existing_run_dirs()
    new_dirs = sorted(after - before)
    if not new_dirs:
        print(f'WARNING: no new output directory detected for preset {preset["id"]}')
        return None
    return new_dirs[-1]

# ---------------------------------------------------------------------------
# Checkpoint scanning
# ---------------------------------------------------------------------------

def _scan_checkpoints(run_dirs: dict) -> list[dict]:
    """Parse val_accuracy from checkpoint filenames across all runs.

    Returns a list of dicts sorted by val_accuracy descending.
    """
    results = []
    for preset_id, info in run_dirs.items():
        ckpt_dir = _OUTPUTS_DIR / info['run_dir'] / 'checkpoints'
        if not ckpt_dir.exists():
            continue
        for f in ckpt_dir.iterdir():
            if f.name == 'last.ckpt' or not f.suffix == '.ckpt':
                continue
            m_acc = _CKPT_PATTERN.search(f.name)
            m_epoch = _EPOCH_PATTERN.search(f.name)
            if not m_acc:
                continue
            results.append({
                'preset_id': preset_id,
                'preset_name': info['name'],
                'alphas': info['alphas'],
                'val_accuracy': float(m_acc.group(1)),
                'epoch': int(m_epoch.group(1)) if m_epoch else -1,
                'checkpoint': str(f),
                'filename': f.name,
            })
    results.sort(key=lambda r: r['val_accuracy'], reverse=True)
    return results

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_top_n(ranked: list[dict], n: int) -> list[dict]:
    top = ranked[:n]
    print(f'\n{"=" * 72}')
    print(f'  TOP {n} CHECKPOINTS (across all presets)')
    print(f'{"=" * 72}\n')
    for i, r in enumerate(top, 1):
        a = r['alphas']
        print(f'  Rank {i}: Preset "{r["preset_name"]}" (id={r["preset_id"]}), '
              f'epoch={r["epoch"]}, val_accuracy={r["val_accuracy"]:.4f}')
        print(f'    alphas: full={a["full"]}, tens={a["tens"]}, units={a["units"]}, count={a["count"]}')
        print(f'    checkpoint: {r["checkpoint"]}')
        print()
    return top


def _print_pipeline_instructions(top: list[dict]) -> None:
    print(f'\n{"=" * 72}')
    print('  HOW TO USE EACH CHECKPOINT IN THE PIPELINE')
    print(f'{"=" * 72}\n')
    for i, r in enumerate(top, 1):
        ckpt_path = r['checkpoint']
        print(f'--- Rank {i}: {r["preset_name"]} (val_accuracy={r["val_accuracy"]:.4f}) ---\n')
        print(f'  PowerShell:')
        print(f'    $env:JERSEY_SOCCERNET_STR_CKPT="{ckpt_path}"')
        print(f'    python main.py SoccerNet test')
        print(f'    Remove-Item Env:\\JERSEY_SOCCERNET_STR_CKPT')
        print()

# ---------------------------------------------------------------------------
# Results JSON
# ---------------------------------------------------------------------------

def _write_results_json(run_dirs: dict, ranked: list[dict], top: list[dict], out_path: Path) -> None:
    payload = {
        'presets': {},
        'global_top_5': [],
    }
    for preset_id, info in run_dirs.items():
        best_for_preset = next((r for r in ranked if r['preset_id'] == preset_id), None)
        payload['presets'][str(preset_id)] = {
            'name': info['name'],
            'alphas': info['alphas'],
            'run_dir': info['run_dir'],
            'best_val_accuracy': best_for_preset['val_accuracy'] if best_for_preset else None,
            'best_checkpoint': best_for_preset['checkpoint'] if best_for_preset else None,
        }
    for r in top:
        payload['global_top_5'].append({
            'rank': top.index(r) + 1,
            'preset_id': r['preset_id'],
            'preset_name': r['preset_name'],
            'val_accuracy': r['val_accuracy'],
            'epoch': r['epoch'],
            'checkpoint': r['checkpoint'],
            'alphas': r['alphas'],
        })
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\nResults written to {out_path}')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Sweep aux_alpha presets for PARSeq jersey-aux training')
    parser.add_argument('--presets-file', type=str, default=str(_DEFAULT_PRESETS),
                        help='Path to aux_alpha_presets.yaml')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of best checkpoints to report (default: 5)')
    parser.add_argument('--max-epochs', type=int, default=25,
                        help='Training epochs per preset (default: 25)')
    args = parser.parse_args()

    with open(args.presets_file, 'r') as f:
        presets = yaml.safe_load(f)['presets']
    print(f'Loaded {len(presets)} presets from {args.presets_file}')

    str_python = _get_str_python()
    data_root = _get_data_root()
    print(f'Python: {str_python}')
    print(f'Data root: {data_root}')

    run_dirs: dict[int, dict] = {}
    t0 = time.time()

    for idx, preset in enumerate(presets):
        pid = preset['id']
        print(f'\n[{idx + 1}/{len(presets)}] Starting preset {pid} ({preset["name"]})...')
        run_name = _train_one_preset(preset, str_python, data_root, args.max_epochs)
        if run_name:
            run_dirs[pid] = {
                'name': preset['name'],
                'run_dir': run_name,
                'alphas': {
                    'full': preset['aux_alpha_full'],
                    'tens': preset['aux_alpha_tens'],
                    'units': preset['aux_alpha_units'],
                    'count': preset['aux_alpha_count'],
                },
            }
        elapsed = time.time() - t0
        remaining = (elapsed / (idx + 1)) * (len(presets) - idx - 1)
        print(f'  Elapsed: {elapsed / 60:.1f} min | Est. remaining: {remaining / 60:.1f} min')

    if not run_dirs:
        print('ERROR: No successful training runs. Exiting.')
        sys.exit(1)

    print(f'\nAll training complete ({(time.time() - t0) / 60:.1f} min total). Scanning checkpoints...')

    ranked = _scan_checkpoints(run_dirs)
    if not ranked:
        print('ERROR: No checkpoints found. Exiting.')
        sys.exit(1)

    top = _print_top_n(ranked, args.top_n)
    _print_pipeline_instructions(top)

    results_path = _REPO_ROOT / 'sweep_results.json'
    _write_results_json(run_dirs, ranked, top, results_path)

    print('\nDone.')


if __name__ == '__main__':
    main()

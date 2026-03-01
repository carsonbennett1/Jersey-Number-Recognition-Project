# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
import json
import sys

import torch

import configuration as _cfg
ROOT = os.path.join(_cfg._main_repo, 'pose/ViTPose/')
sys.path.append(str(ROOT))  # add ROOT to PATH

from argparse import ArgumentParser

from xtcocotools.coco import COCO

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo


def main():
    """Visualize the demo images.

    Require the json_file containing boxes.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default='',
        help='Json file containing image info.')
    parser.add_argument(
        '--out-json',
        type=str,
        default='',
        help='Json file containing results.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='Root of the output img file. '
             'Default not saving the visualization images.')
    # Improved device detection: CUDA > MPS > CPU
    default_device = 'cpu'
    if torch.cuda.is_available():
        default_device = 'cuda:0'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        default_device = 'mps'
    
    parser.add_argument(
        '--device', default=default_device, help='Device used for inference (cuda, mps, or cpu)')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()

    print(f"Pose device: {args.device}")

    # cudnn.benchmark only works on CUDA
    if 'cuda' in args.device:
        torch.backends.cudnn.benchmark = True

    coco = COCO(args.json_file)
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    img_keys = list(coco.imgs.keys())

    return_heatmap = False
    output_layer_names = None

    results = []
    save_vis = args.out_img_root != ''

    from tqdm import tqdm
    with torch.inference_mode():
        for i in tqdm(range(len(img_keys)), desc="Pose estimation"):
            image_id = img_keys[i]
            image = coco.loadImgs(image_id)[0]
            image_name = os.path.join(args.img_root, image['file_name'])
            ann_ids = coco.getAnnIds(image_id)

            person_results = []
            for ann_id in ann_ids:
                person = {}
                ann = coco.anns[ann_id]
                person['bbox'] = ann['bbox']
                person_results.append(person)

            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                image_name,
                person_results,
                bbox_thr=None,
                format='xywh',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)

            results.append(
                {"img_name": image['file_name'], "id": image_id, "keypoints": pose_results[0]['keypoints'].tolist()})

            if save_vis:
                os.makedirs(args.out_img_root, exist_ok=True)
                out_file = os.path.join(args.out_img_root, f'vis_{i}.jpg')
                vis_pose_result(
                    pose_model,
                    image_name,
                    pose_results,
                    dataset=dataset,
                    dataset_info=dataset_info,
                    kpt_score_thr=args.kpt_thr,
                    radius=args.radius,
                    thickness=args.thickness,
                    show=args.show,
                    out_file=out_file)

    if args.out_json != '':
        with open(args.out_json, 'w') as fp:
            json.dump({"pose_results": results}, fp)


if __name__ == '__main__':
    main()
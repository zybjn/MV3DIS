#!/usr/bin/env python3
"""Generate per-scene mask .pth files with YOLO-World/GroundingDINO + SAM.

GroundingDINO+SAM
CUDA_VISIBLE_DEVICES=0 python mask_generator/run.py \
  --detector grounding \
  --data-root /defaultShare/archive/user_name/Dataset/scannet/RGB_D/validation \
  --scene-list meta_data/scannetv2_val.txt \
  --classes-file mask_generator/scannet200_classes.txt \
  --output-dir /nfs/user_name/Dataset/scannet/open3dis_processed/exp_scannet200/version_ramyoloworld/maskyoloworldsam/grounding_sam2_mv3dis/2d_pth_score \
  --grounding-dino-root /defaultShare/archive/user_name/private/Open3DIS-main/segmenter2d/GroundingDINO \
  --detector-config /defaultShare/archive/user_name/private/Open3DIS-main/segmenter2d/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --detector-checkpoint /defaultShare/archive/user_name/private/checkpoint/groundingdino_swint_ogc.pth \
  --sam2-root /defaultShare/archive/user_name/private/Open3DIS-main/segmenter2d/SAM2 \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-checkpoint /defaultShare/archive/user_name/private/checkpoint/sam2.1_hiera_large.pt \
  --image-interval 10


YOLO-World+SAM

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 python mask_generator/run.py \
  --detector yolo \
  --data-root /defaultShare/archive/user_name/Dataset/scannet/RGB_D/validation \
  --scene-list meta_data/scannetv2_val.txt \
  --classes-file mask_generator/scannet200_classes.txt \
  --output-dir /nfs/user_name/Dataset/scannet/open3dis_processed/exp_scannet200/version_ramyoloworld/maskyoloworldsam/yoloworld_sam2_mv3dis/2d_pth_score \
  --yolo-world-root /defaultShare/archive/user_name/private/OpenYOLO3D-main/models/YOLO-World \
  --detector-config /defaultShare/archive/user_name/private/OpenYOLO3D-main/pretrained/configs/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py \
  --detector-checkpoint /defaultShare/archive/user_name/private/checkpoint/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth \
  --clip-model /defaultShare/archive/user_name/private/checkpoint/huggingfaceclip/clip-vit-base-patch32 \
  --sam2-root /defaultShare/archive/user_name/private/Open3DIS-main/segmenter2d/SAM2 \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-checkpoint /defaultShare/archive/user_name/private/checkpoint/sam2.1_hiera_large.pt \
  --yolo-conf-threshold 0.05 \
  --yolo-nms-threshold 0.8 \
  --image-interval 10

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

try:
    from .mask_generator import create_generator
except ImportError:
    from mask_generator import create_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=("yolo", "grounding"), required=True)
    parser.add_argument("--data-root", required=True, help="Root containing scene folders")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classes-file", required=True, help="One class name per line")
    parser.add_argument("--scene-list", help="Optional file with one scene ID per line")
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--image-subdir", default="color")
    parser.add_argument("--image-interval", type=int, default=10)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--detector-config", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--grounding-dino-root")
    parser.add_argument("--yolo-world-root")
    parser.add_argument(
        "--clip-model",
        help="Local Hugging Face CLIP directory used by YOLO-World",
    )
    parser.add_argument("--sam2-root")
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument(
        "--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml"
    )

    # Original GroundingDINO parameters. They are ignored by YOLO-World.
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.3)
    parser.add_argument("--prompt-chunk-size", type=int, default=5)
    parser.add_argument(
        "--yolo-conf-threshold",
        type=float,
        default=0.05,
        help="YOLO-World confidence threshold (default: 0.05)",
    )
    parser.add_argument(
        "--yolo-nms-threshold",
        type=float,
        default=0.8,
        help="YOLO-World box NMS IoU threshold (default: 0.8)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Remove this worker's claim if a scene fails, allowing a retry",
    )
    return parser.parse_args()


def read_nonempty_lines(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def scene_ids(args: argparse.Namespace) -> list[str]:
    if args.scene_list:
        scenes = sorted(read_nonempty_lines(args.scene_list))
    else:
        scenes = sorted(path.name for path in Path(args.data_root).iterdir() if path.is_dir())
    return scenes[args.begin : args.end]


def claim_scene(claim_dir: Path, scene_id: str) -> Path | None:
    """Atomically reserve a scene so multiple nodes do not duplicate work."""
    claim_path = claim_dir / f"{scene_id}.claim"
    try:
        descriptor = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    with os.fdopen(descriptor, "w") as file:
        file.write(f"pid={os.getpid()}\n")
    return claim_path


def main() -> None:
    args = parse_args()
    if args.image_interval <= 0:
        raise ValueError("--image-interval must be positive")
    classes = read_nonempty_lines(args.classes_file)
    if not classes:
        raise ValueError("Classes file is empty")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    claim_dir = output_dir / ".claims"
    claim_dir.mkdir(exist_ok=True)
    scenes = scene_ids(args)
    if not scenes:
        raise RuntimeError("No scenes selected")

    generator = create_generator(args)
    for scene_id in tqdm(scenes, desc="Scenes"):
        output_path = output_dir / f"{scene_id}.pth"
        if output_path.is_file():
            continue
        claim_path = claim_scene(claim_dir, scene_id)
        if claim_path is None:
            continue
        try:
            result = generator.process_scene(
                Path(args.data_root) / scene_id,
                classes,
                image_subdir=args.image_subdir,
                image_interval=args.image_interval,
            )
            temporary = output_dir / f".{scene_id}.{os.getpid()}.tmp"
            torch.save(result, temporary)
            os.replace(temporary, output_path)
            claim_path.unlink(missing_ok=True)
        except Exception:
            if args.retry_failed:
                claim_path.unlink(missing_ok=True)
            raise


if __name__ == "__main__":
    main()

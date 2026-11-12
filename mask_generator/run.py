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

SAM2.1
CUDA_VISIBLE_DEVICES=0 python mask_generator/run.py \
  --scannetpp \
  --scene_id 3f15a9266d \
  --data2d_path /defaultShare/pubdata/Indoordata/scannetppv2/data \
  --sam2_root /defaultShare/archive/zhaoyibo/private/MV3DIS-main/reports/scannetpp_best_ap220245_20261102/mask_source/SAM2 \
  --checkpoint /defaultShare/archive/zhaoyibo/private/MV3DIS-main/reports/scannetpp_best_ap220245_20261102/checkpoints/sam2.1_hiera_large.pt \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --view_freq 2 \
  --chunk_frames 16 \
  --output_root ./outputs/sam2_scannetpp
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import os
from pathlib import Path

import torch
from tqdm import tqdm



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


def main_detectors() -> None:
    try:
        from .mask_generator import create_generator
    except ImportError:
        from mask_generator import create_generator
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


MASK_PARAMS = dict(
    points_per_side=32, points_per_batch=64, pred_iou_thresh=.75,
    stability_score_thresh=.88, stability_score_offset=1., box_nms_thresh=.7,
    crop_n_layers=0, crop_nms_thresh=.7, crop_n_points_downscale_factor=2,
    min_mask_region_area=0, use_m2m=False, multimask_output=True,
    output_mode='coco_rle',
)


def build_parser():
    parser = argparse.ArgumentParser(description="Generate ScanNet++ automatic SAM2 masks with the best settings")
    parser.add_argument('--data2d_path', type=Path, required=True,
                        help='ScanNet++ data directory containing scene/iphone/color')
    parser.add_argument('--scene_id', required=True, help='Comma-separated scene IDs')
    parser.add_argument('--sam2_root', type=Path, required=True,
                        help='SAM2 repository directory containing the sam2 package')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model_cfg', default='configs/sam2.1/sam2.1_hiera_l.yaml')
    parser.add_argument('--output_root', type=Path, required=True)
    parser.add_argument('--view_freq', type=int, default=2)
    parser.add_argument('--sampling_json', type=Path,
                        help='Optional exact scene-to-frame-list manifest; overrides view_freq')
    parser.add_argument('--chunk_frames', type=int, default=16)
    parser.add_argument('--keep_chunks', action='store_true')
    parser.add_argument('--smoke', action='store_true',
                        help='Generate only two frames under output_root/smoke/masks')
    return parser


def select_frames(args, scene):
    if args.sampling_json:
        frames = json.loads(args.sampling_json.read_text())[scene]
    else:
        paths = sorted((args.data2d_path / scene / 'iphone/color').glob('*.jpg'),
                       key=lambda path: int(path.stem))
        frames = [f'{int(path.stem):05d}' for path in paths[::args.view_freq]]
    if not frames or len(set(frames)) != len(frames):
        raise ValueError('Expected a nonempty list of unique frames: ' + scene)
    return frames[:2] if args.smoke else frames


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def generate_scene(args, scene, checkpoint_sha, source_sha):
    import numpy as np
    import torch
    from PIL import Image
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    frames = select_frames(args, scene)
    root = args.output_root / 'smoke' if args.smoke else args.output_root
    for directory in ['masks', 'status', 'chunks/' + scene]:
        (root / directory).mkdir(parents=True, exist_ok=True)
    folder = root / 'chunks' / scene
    output = root / 'masks' / f'{scene}.pth'
    status = root / 'status' / f'{scene}.json'
    settings = dict(params=MASK_PARAMS, frames=frames, checkpoint_sha256=checkpoint_sha,
                    model_cfg=args.model_cfg, sam2_source_sha256=source_sha,
                    generator_sha256=sha256(Path(__file__)),
                    data2d_path=str(args.data2d_path.resolve()),
                    dtype='bfloat16', apply_postprocessing=False,
                    chunk_frames=args.chunk_frames)
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    if status.exists():
        previous = json.loads(status.read_text())
        if previous['signature'] != signature:
            raise ValueError('Output settings changed; use another --output_root: ' + scene)
        if previous['state'] == 'complete' and output.is_file():
            print('Already completed:', scene, flush=True)
            return
    elif output.exists():
        raise FileExistsError('Existing masks have no matching generation status: ' + str(output))

    def save(path, value):
        temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
        torch.save(value, temporary)
        temporary.replace(path)

    torch.set_num_threads(4)
    model = generator = None
    parts = []
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, len(frames), args.chunk_frames):
            batch = frames[start:start + args.chunk_frames]
            path = folder / f'{start:06d}.pth'
            if path.exists():
                chunk = torch.load(path, map_location='cpu', weights_only=False)
                if chunk['signature'] != signature or list(chunk['frames']) != batch:
                    raise ValueError('Incompatible cached chunk: ' + str(path))
            else:
                if generator is None:
                    model = build_sam2(args.model_cfg, str(args.checkpoint), device='cuda',
                                       apply_postprocessing=False)
                    generator = SAM2AutomaticMaskGenerator(model, **MASK_PARAMS)
                data = {}
                for frame in batch:
                    image_path = args.data2d_path / scene / 'iphone/color' / f'{frame}.jpg'
                    with Image.open(image_path) as image_file:
                        image = np.array(image_file.convert('RGB'))
                    masks = generator.generate(image)
                    masks.sort(key=lambda mask: mask['predicted_iou'], reverse=True)
                    if masks:
                        confidence = torch.tensor([mask['predicted_iou'] for mask in masks])
                        if not torch.isfinite(confidence).all():
                            raise ValueError('Non-finite mask confidence: ' + str(image_path))
                        if any(mask['segmentation']['size'] != list(image.shape[:2]) for mask in masks):
                            raise ValueError('Mask/image dimensions differ: ' + str(image_path))
                        data[frame] = dict(
                            masks=[mask['segmentation'] for mask in masks], conf=confidence,
                            score=confidence.clone(),
                            stability_score=torch.tensor([mask['stability_score'] for mask in masks]))
                    else:
                        data[frame] = None
                chunk = dict(signature=signature, frames=data)
                save(path, chunk)
            parts.append(path)
            atomic_json(status, dict(state='generating', signature=signature,
                                     processed_frames=min(start + len(batch), len(frames)),
                                     total_frames=len(frames)))
            print('MASK_PROGRESS', scene, min(start + len(batch), len(frames)), len(frames), flush=True)
            del chunk

    result, empty = {}, []
    for path in parts:
        chunk = torch.load(path, map_location='cpu', weights_only=False)
        for frame, value in chunk['frames'].items():
            if value is None:
                empty.append(frame)
            else:
                result[frame] = value
    if not result:
        raise ValueError('No masks were generated: ' + scene)
    save(output, result)
    atomic_json(status, dict(state='complete', signature=signature, settings=settings,
                             total_frames=len(frames), empty_frames=empty,
                             masks=sum(len(value['masks']) for value in result.values()),
                             output=str(output), seconds=time.monotonic() - started))
    if not args.keep_chunks:
        for path in parts:
            path.unlink()
        folder.rmdir()
    print('MASK_COMPLETE', output, flush=True)


def main_scannetpp(argv=None):
    args = build_parser().parse_args(argv)
    if args.view_freq < 1 or args.chunk_frames < 1:
        raise SystemExit('--view_freq and --chunk_frames must be positive')
    if not (args.sam2_root / 'sam2').is_dir():
        raise SystemExit('--sam2_root must contain the sam2 package')
    sys.path.insert(0, str(args.sam2_root.resolve()))
    checkpoint_sha = sha256(args.checkpoint)
    source = hashlib.sha256()
    files = sorted(path for path in (args.sam2_root / 'sam2').rglob('*')
                   if path.suffix in ('.py', '.yaml') and path.is_file())
    for path in files:
        source.update(str(path.relative_to(args.sam2_root)).encode())
        source.update(path.read_bytes())
    for scene in args.scene_id.split(','):
        generate_scene(args, scene, checkpoint_sha, source.hexdigest())




def main():
    argv = sys.argv[1:]
    if '--scannetpp' in argv:
        main_scannetpp([value for value in argv if value != '--scannetpp'])
    else:
        main_detectors()

if __name__ == "__main__":
    main()

# 2D Mask Generator

This tool detects objects with YOLO-World or GroundingDINO and generates ScanNetV2 2D masks with SAM2. Its output can be used directly by MV3DIS.

## Output Format

Each scene is saved as `<scene_id>.pth`. Every frame contains:

```python
{
    "masks": list_of_coco_rle,
    "conf": detector_confidence,
    "score": sam2_quality_score,
}
```

Store the results under `<sam2d_path>/2d_pth_score`. MV3DIS loads existing `2d_mask_score` PNGs when available. Otherwise, it reconstructs the 2D label maps in memory using the PTH `score` values.

## Installation

Install the basic dependencies from the MV3DIS root directory:

```bash
pip install -r mask_generator/requirements.txt
```

Install SAM2 and the selected detector separately by following their official instructions:

- [SAM2](https://github.com/facebookresearch/sam2)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [YOLO-World](https://github.com/AILab-CVC/YOLO-World)

Use `--sam2-root`, `--grounding-dino-root`, and `--yolo-world-root` when the projects are available as local source trees rather than installed packages.

## Environment and Input

- Recommended environment: `openyolo3d`
- RGB input layout: `<data-root>/<scene_id>/color/*.jpg`
- Scene list: `meta_data/scannetv2_val.txt`
- Class list: `mask_generator/scannet200_classes.txt`
- Default sampling interval: one image every 10 frames

## GroundingDINO + SAM2

Run from the MV3DIS root directory:

```bash
conda activate openyolo3d

CUDA_VISIBLE_DEVICES=0 python mask_generator/run.py \
  --detector grounding \
  --data-root /path/to/scannet/RGB_D/validation \
  --scene-list meta_data/scannetv2_val.txt \
  --classes-file mask_generator/scannet200_classes.txt \
  --output-dir /path/to/grounding_sam2_mv3dis/2d_pth_score \
  --grounding-dino-root /path/to/GroundingDINO \
  --detector-config /path/to/GroundingDINO_SwinT_OGC.py \
  --detector-checkpoint /path/to/groundingdino_swint_ogc.pth \
  --sam2-root /path/to/SAM2 \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-checkpoint /path/to/sam2.1_hiera_large.pt \
  --image-interval 10
```

## YOLO-World + SAM2

Use a local CLIP model on nodes without Hugging Face access:

```bash
conda activate openyolo3d

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 python mask_generator/run.py \
  --detector yolo \
  --data-root /path/to/scannet/RGB_D/validation \
  --scene-list meta_data/scannetv2_val.txt \
  --classes-file mask_generator/scannet200_classes.txt \
  --output-dir /path/to/yoloworld_sam2_mv3dis/2d_pth_score \
  --yolo-world-root /path/to/YOLO-World \
  --detector-config /path/to/yolo_world_config.py \
  --detector-checkpoint /path/to/yolo_world.pth \
  --clip-model /path/to/clip-vit-base-patch32 \
  --sam2-root /path/to/SAM2 \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-checkpoint /path/to/sam2.1_hiera_large.pt \
  --yolo-conf-threshold 0.05 \
  --yolo-nms-threshold 0.8 \
  --image-interval 10
```

## Resuming and Multi-Node Execution

Existing `<scene_id>.pth` files are skipped automatically. Multiple workers may share one output directory; atomic files under `.claims` prevent duplicate processing. If a worker exits unexpectedly, remove the corresponding `.claims/<scene_id>.claim` before retrying that scene.

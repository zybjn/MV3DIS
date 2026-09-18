#!/usr/bin/env python
"""Generate 2D confidence-ordered mask maps from scene-level RLE files."""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask
import torch
from tqdm import tqdm


def filter_masks(masks, scores, overlap_threshold=0.5):
    """Match the production MV3DIS IoU filtering and score tie-breaking."""
    mask_count = masks.shape[0]
    gpu_masks = torch.as_tensor(masks, dtype=torch.float32, device="cuda")
    flat_masks = gpu_masks.reshape(mask_count, -1)
    intersection = torch.mm(flat_masks, flat_masks.t())
    area = flat_masks.sum(dim=1)
    union = area[:, None] + area[None, :] - intersection
    high_overlap_pairs = (
        ((intersection / union) > overlap_threshold)
        .triu(diagonal=1)
        .nonzero(as_tuple=False)
    )

    keep = torch.ones(mask_count, dtype=torch.bool, device="cuda")
    if high_overlap_pairs.shape[0] > 0:
        gpu_scores = torch.as_tensor(scores, dtype=torch.float32, device="cuda")
        first = high_overlap_pairs[:, 0]
        second = high_overlap_pairs[:, 1]
        losers = torch.where(gpu_scores[first] < gpu_scores[second], first, second)
        keep[losers] = False
    return keep.cpu().numpy()


def compose_mask_map(masks, scores, keep):
    """Compose labels so the higher score wins every overlapping pixel."""
    indices = np.flatnonzero(keep)
    height, width = masks.shape[-2:]
    output = np.zeros((height, width), dtype=np.uint16)
    if indices.size == 0:
        return output

    order = np.lexsort((indices, -scores[indices]))
    ordered_indices = indices[order]
    class_ids = np.arange(indices.size, 0, -1, dtype=np.uint16)
    candidate = np.empty_like(output)
    for mask_index, class_id in zip(ordered_indices, class_ids):
        np.multiply(masks[mask_index], class_id, out=candidate, casting="unsafe")
        np.maximum(output, candidate, out=output)
    return output


def atomic_write_png(path, image):
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if not cv2.imwrite(str(temporary), image):
        raise IOError(f"Failed to write {temporary}")
    os.replace(temporary, path)


def atomic_write_npy(path, array):
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "wb") as file:
        np.save(file, array)
    os.replace(temporary, path)


def scene_shape(scene_data):
    for frame_data in scene_data.values():
        masks = frame_data.get("masks", [])
        if masks:
            return tuple(masks[0]["size"])
    raise ValueError("Scene contains no masks, so image size is unknown")


def process_scene(scene_path, output_root, overwrite=False):
    scene_id = scene_path.stem
    output_dir = output_root / scene_id
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_data = torch.load(scene_path, map_location="cpu")
    height, width = scene_shape(scene_data)

    for frame_id, frame_data in tqdm(scene_data.items(), desc=scene_id, leave=False):
        png_path = output_dir / f"maskraw_{frame_id}.png"
        keep_path = output_dir / f"kept_mask_indices_{frame_id}.npy"
        if not overwrite and png_path.is_file() and keep_path.is_file():
            continue

        encoded_masks = frame_data["masks"]
        scores = frame_data["conf"]
        if torch.is_tensor(scores):
            scores = scores.detach().cpu().numpy()
        else:
            scores = np.asarray(scores, dtype=np.float32)

        if encoded_masks:
            masks = np.stack(
                [pycocotools.mask.decode(mask) for mask in encoded_masks],
                axis=0,
            )
            keep = filter_masks(masks, scores)
            mask_map = compose_mask_map(masks, scores, keep)
        else:
            keep = np.zeros(0, dtype=bool)
            mask_map = np.zeros((height, width), dtype=np.uint16)

        atomic_write_png(png_path, mask_map)
        atomic_write_npy(keep_path, keep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--scene", action="append")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker-index must be in [0, num-workers)")

    scene_paths = sorted(args.source_dir.glob("scene*.pth"))
    if args.scene:
        requested = set(args.scene)
        scene_paths = [path for path in scene_paths if path.stem in requested]
        missing = requested - {path.stem for path in scene_paths}
        if missing:
            raise FileNotFoundError(f"Missing scenes: {sorted(missing)}")
    scene_paths = scene_paths[args.worker_index :: args.num_workers]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for scene_path in tqdm(scene_paths, desc="Scenes"):
        process_scene(scene_path, args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()

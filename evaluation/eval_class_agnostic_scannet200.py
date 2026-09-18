#!/usr/bin/env python3
"""Standalone class-agnostic instance evaluation for ScanNet200. From Open3DIS.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from tqdm import tqdm


SCANNET200_INSTANCE_CLASS_COUNT = 198
CLASS_NAME = "class_agnostic"


def load_torch(path: Path) -> Any:
    """Load trusted local prediction/GT files on CPU across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    """Decode Open3DIS 1-indexed start/length RLE into a flat uint8 mask."""
    if "length" not in rle or "counts" not in rle:
        raise KeyError("RLE mask must contain 'length' and 'counts'")
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.split()
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    if counts.size % 2:
        raise ValueError("RLE counts must contain start/length pairs")

    starts = counts[0::2] - 1
    ends = starts + counts[1::2]
    mask = np.zeros(int(rle["length"]), dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    return mask


def decode_mask(mask: Any) -> np.ndarray:
    if isinstance(mask, dict):
        mask = decode_rle(mask)
    elif torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    else:
        mask = np.asarray(mask)
    return np.not_equal(np.asarray(mask).reshape(-1), 0).astype(np.uint8)


def _scores(prediction: dict[str, Any], score_key: str | None, count: int) -> np.ndarray:
    if score_key is None:
        return np.ones(count, dtype=np.float32)
    if score_key not in prediction:
        raise KeyError(f"Prediction does not contain score key {score_key!r}")
    values = prediction[score_key]
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) != count:
        raise ValueError(f"Number of scores ({len(values)}) does not match masks ({count})")
    return values


def _load_scene(
    pred_path: Path,
    gt_dir: Path,
    pred_key: str,
    score_key: str | None,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    scene_id = pred_path.stem
    gt_path = gt_dir / f"{scene_id}_inst_nostuff.pth"
    if not pred_path.is_file():
        raise FileNotFoundError(f"Prediction not found: {pred_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    prediction = load_torch(pred_path)
    if not isinstance(prediction, dict):
        raise TypeError(f"Prediction must be a dict, got {type(prediction).__name__}")
    if pred_key not in prediction:
        raise KeyError(f"Prediction does not contain mask key {pred_key!r}")

    gt = load_torch(gt_path)
    if not isinstance(gt, (tuple, list)) or len(gt) < 4:
        raise TypeError("Ground truth must be a tuple/list with at least four entries")
    sem_gt = np.asarray(gt[2], dtype=np.int64).reshape(-1)
    inst_gt = np.asarray(gt[3], dtype=np.int64).reshape(-1)
    if len(sem_gt) != len(inst_gt):
        raise ValueError("GT semantic and instance arrays have different lengths")

    masks = prediction[pred_key]
    scores = _scores(prediction, score_key, len(masks))
    instances = []
    for index, raw_mask in enumerate(masks):
        mask = decode_mask(raw_mask)
        if len(mask) != len(inst_gt):
            raise ValueError(f"Mask {index} has {len(mask)} points, but GT has {len(inst_gt)}")
        instances.append(
            {
                "scan_id": scene_id,
                "conf": float(scores[index]),
                "pred_mask": mask,
            }
        )
    return instances, sem_gt, inst_gt


class ScanNet200ClassAgnosticEvaluator:
    """Self-contained point-mask evaluator matching Open3DIS ScanNetEval."""

    def __init__(self, min_region_size: int = 100) -> None:
        self.encode_value = 1000
        self.valid_class_ids = np.arange(1, SCANNET200_INSTANCE_CLASS_COUNT + 1)
        self.ious = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_size = int(min_region_size)

    def _gt_instances(self, encoded_gt: np.ndarray) -> list[dict[str, Any]]:
        instances = []
        for instance_id in np.unique(encoded_gt):
            if instance_id == 0:
                continue
            label_id = int(instance_id // self.encode_value)
            if label_id not in self.valid_class_ids:
                continue
            instances.append(
                {
                    "instance_id": int(instance_id),
                    "label_id": label_id,
                    "vert_count": int(np.count_nonzero(encoded_gt == instance_id)),
                    "med_dist": -1.0,
                    "dist_conf": 0.0,
                    "box": np.zeros(6),
                }
            )
        return instances

    def assign_instances_for_scan(
        self,
        preds: list[dict[str, Any]],
        gt_semantic: np.ndarray,
        gt_instance: np.ndarray,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
        # Exact ScanNet200 remapping used by Open3DIS ScanNetEval.
        gt_semantic = np.asarray(gt_semantic, dtype=np.int64).copy() - 1
        gt_semantic[gt_semantic < 0] = 0
        gt_instance = np.asarray(gt_instance, dtype=np.int64).copy() + 1
        ignore = gt_instance < 0
        encoded_gt = gt_semantic * self.encode_value + gt_instance
        encoded_gt[ignore] = 0

        gt_instances = deepcopy(self._gt_instances(encoded_gt))
        for gt in gt_instances:
            gt["matched_pred"] = []
        gt2pred = {CLASS_NAME: gt_instances}
        pred2gt: dict[str, list[dict[str, Any]]] = {CLASS_NAME: []}

        bool_void = np.logical_not(
            np.isin(encoded_gt // self.encode_value, self.valid_class_ids)
        )
        num_pred_instances = 0
        for pred in preds:
            pred_mask = np.not_equal(pred["pred_mask"], 0)
            if pred_mask.shape[0] != encoded_gt.shape[0]:
                raise ValueError("Prediction and GT point counts do not match")
            num_points = int(np.count_nonzero(pred_mask))
            if num_points < self.min_region_size:
                continue

            pred_instance = {
                "filename": f"{pred['scan_id']}_{num_pred_instances}",
                "pred_id": num_pred_instances,
                "label_id": None,
                "vert_count": num_points,
                "confidence": float(pred["conf"]),
                "void_intersection": int(np.count_nonzero(bool_void & pred_mask)),
            }
            matched_gt = []
            for gt_index, gt in enumerate(gt2pred[CLASS_NAME]):
                intersection = int(
                    np.count_nonzero((encoded_gt == gt["instance_id"]) & pred_mask)
                )
                if intersection == 0:
                    continue
                gt_copy = gt.copy()
                pred_copy = pred_instance.copy()
                gt_copy["intersection"] = intersection
                pred_copy["intersection"] = intersection
                iou = intersection / (
                    gt_copy["vert_count"] + pred_copy["vert_count"] - intersection
                )
                gt_copy["iou"] = iou
                pred_copy["iou"] = iou
                matched_gt.append(gt_copy)
                gt2pred[CLASS_NAME][gt_index]["matched_pred"].append(pred_copy)
            pred_instance["matched_gt"] = matched_gt
            pred2gt[CLASS_NAME].append(pred_instance)
            num_pred_instances += 1
        return gt2pred, pred2gt

    def evaluate_matches(self, matches: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        ap = np.zeros((1, 1, len(self.ious)), dtype=float)
        rc = np.zeros((1, 1, len(self.ious)), dtype=float)
        for iou_index, iou_threshold in enumerate(self.ious):
            pred_visited = {
                pred["filename"]: False
                for match in matches.values()
                for pred in match["pred"][CLASS_NAME]
            }
            y_true = np.empty(0)
            y_score = np.empty(0)
            hard_false_negatives = 0
            has_gt = False
            has_pred = False

            for match in matches.values():
                pred_instances = match["pred"][CLASS_NAME]
                gt_instances = [
                    gt
                    for gt in match["gt"][CLASS_NAME]
                    if gt["instance_id"] >= self.encode_value
                    and gt["vert_count"] >= self.min_region_size
                ]
                has_gt |= bool(gt_instances)
                has_pred |= bool(pred_instances)
                cur_true = np.ones(len(gt_instances))
                cur_score = np.full(len(gt_instances), -float("inf"))
                cur_match = np.zeros(len(gt_instances), dtype=bool)

                for gt_index, gt in enumerate(gt_instances):
                    found_match = False
                    for pred in gt["matched_pred"]:
                        if pred_visited[pred["filename"]]:
                            continue
                        if pred["iou"] > iou_threshold:
                            confidence = pred["confidence"]
                            if cur_match[gt_index]:
                                cur_score[gt_index], lower = (
                                    max(cur_score[gt_index], confidence),
                                    min(cur_score[gt_index], confidence),
                                )
                                cur_true = np.append(cur_true, 0)
                                cur_score = np.append(cur_score, lower)
                                cur_match = np.append(cur_match, True)
                            else:
                                found_match = True
                                cur_match[gt_index] = True
                                cur_score[gt_index] = confidence
                                pred_visited[pred["filename"]] = True
                    if not found_match:
                        hard_false_negatives += 1

                y_true = np.append(y_true, cur_true[cur_match])
                y_score = np.append(y_score, cur_score[cur_match])

                for pred in pred_instances:
                    if any(gt["iou"] > iou_threshold for gt in pred["matched_gt"]):
                        continue
                    num_ignore = pred["void_intersection"]
                    for gt in pred["matched_gt"]:
                        if gt["instance_id"] < self.encode_value:
                            num_ignore += gt["intersection"]
                        if gt["vert_count"] < self.min_region_size:
                            num_ignore += gt["intersection"]
                    if float(num_ignore) / pred["vert_count"] <= iou_threshold:
                        y_true = np.append(y_true, 0)
                        y_score = np.append(y_score, pred["confidence"])

            if has_gt and has_pred and len(y_true):
                score_order = np.argsort(y_score)
                y_true_sorted = y_true[score_order]
                y_true_cumsum = np.cumsum(y_true_sorted)
                _, unique_indices = np.unique(y_score[score_order], return_index=True)
                precision = np.zeros(len(unique_indices) + 1)
                recall = np.zeros(len(unique_indices) + 1)
                num_examples = len(y_true_sorted)
                num_true_examples = y_true_cumsum[-1]
                y_true_cumsum = np.append(y_true_cumsum, 0)
                for result_index, score_index in enumerate(unique_indices):
                    cumsum = y_true_cumsum[score_index - 1]
                    tp = num_true_examples - cumsum
                    fp = num_examples - score_index - tp
                    fn = cumsum + hard_false_negatives
                    precision[result_index] = float(tp) / (tp + fp)
                    recall[result_index] = float(tp) / (tp + fn)
                rc_current = recall[0]
                precision[-1] = 1.0
                recall[-1] = 0.0
                recall_for_conv = np.append(recall[0], recall)
                recall_for_conv = np.append(recall_for_conv, 0.0)
                step_widths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], "valid")
                ap_current = np.dot(precision, step_widths)
            elif has_gt:
                ap_current = rc_current = 0.0
            else:
                ap_current = rc_current = float("nan")
            ap[0, 0, iou_index] = ap_current
            rc[0, 0, iou_index] = rc_current
        return ap, rc

    def compute_averages(self, ap: np.ndarray, rc: np.ndarray) -> dict[str, Any]:
        iou50 = np.where(np.isclose(self.ious, 0.5))
        iou25 = np.where(np.isclose(self.ious, 0.25))
        all_but_25 = np.where(~np.isclose(self.ious, 0.25))
        class_metrics = {
            "ap": float(np.average(ap[0, 0, all_but_25])),
            "ap50%": float(np.average(ap[0, 0, iou50])),
            "ap25%": float(np.average(ap[0, 0, iou25])),
            "rc": float(np.average(rc[0, 0, all_but_25])),
            "rc50%": float(np.average(rc[0, 0, iou50])),
            "rc25%": float(np.average(rc[0, 0, iou25])),
        }
        return {
            "all_ap": float(np.nanmean(ap[0, :, all_but_25])),
            "all_ap_50%": float(np.nanmean(ap[0, :, iou50])),
            "all_ap_25%": float(np.nanmean(ap[0, :, iou25])),
            "all_rc": float(np.nanmean(rc[0, :, all_but_25])),
            "all_rc_50%": float(np.nanmean(rc[0, :, iou50])),
            "all_rc_25%": float(np.nanmean(rc[0, :, iou25])),
            "classes": {CLASS_NAME: class_metrics},
        }

    @staticmethod
    def print_results(metrics: dict[str, Any]) -> None:
        values = metrics["classes"][CLASS_NAME]
        print("\n" + "#" * 64)
        print(f"{'what':<15}:{'AP':>8}{'AP_50%':>8}{'AP_25%':>8}{'AR':>8}{'RC_50%':>8}{'RC_25%':>8}")
        print("#" * 64)
        print(
            f"{CLASS_NAME:<15}:{values['ap']:>8.3f}{values['ap50%']:>8.3f}"
            f"{values['ap25%']:>8.3f}{values['rc']:>8.3f}"
            f"{values['rc50%']:>8.3f}{values['rc25%']:>8.3f}"
        )
        print("-" * 64)
        print(
            f"{'average':<15}:{metrics['all_ap']:>8.3f}{metrics['all_ap_50%']:>8.3f}"
            f"{metrics['all_ap_25%']:>8.3f}{metrics['all_rc']:>8.3f}"
            f"{metrics['all_rc_50%']:>8.3f}{metrics['all_rc_25%']:>8.3f}"
        )
        print("#" * 64 + "\n")

    @staticmethod
    def write_result_file(metrics: dict[str, Any], filename: Path) -> None:
        values = metrics["classes"][CLASS_NAME]
        with filename.open("w") as file:
            file.write("class,class id,ap,ap50,ap25\n")
            file.write(
                f"{CLASS_NAME},{values['ap']},{values['ap50%']},{values['ap25%']}\n"
            )

    def evaluate(
        self,
        predictions: Sequence[list[dict[str, Any]]],
        gt_semantic: Sequence[np.ndarray],
        gt_instance: Sequence[np.ndarray],
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        matches = {}
        for index in tqdm(range(len(gt_semantic)), desc="Matching scenes"):
            gt2pred, pred2gt = self.assign_instances_for_scan(
                predictions[index], gt_semantic[index], gt_instance[index]
            )
            matches[f"gt_{index}"] = {"gt": gt2pred, "pred": pred2gt}
        ap, rc = self.evaluate_matches(matches)
        metrics = self.compute_averages(ap, rc)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.write_result_file(metrics, output_dir / "result.txt")
        self.print_results(metrics)
        return metrics


def _scene_files(
    pred_dir: Path,
    scene_list: str | Path | Iterable[str] | None,
) -> list[Path]:
    if scene_list is None:
        return sorted(pred_dir.glob("*.pth"))
    if isinstance(scene_list, (str, Path)):
        scene_ids = Path(scene_list).expanduser().read_text().splitlines()
    else:
        scene_ids = list(scene_list)
    return [
        pred_dir / f"{str(scene_id).strip().removesuffix('.pth')}.pth"
        for scene_id in scene_ids
        if str(scene_id).strip()
    ]


def evaluate_class_agnostic_scannet200(
    pred_dir: str | Path,
    gt_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    pred_key: str = "ins",
    score_key: str | None = None,
    scene_list: str | Path | Iterable[str] | None = None,
    max_scenes: int | None = None,
    skip_errors: bool = False,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Validate ScanNet200 class-agnostic point masks and return AP/AR metrics.

    ``score_key=None`` assigns score 1.0 to every proposal, matching the
    original Open3DIS class-agnostic evaluator.  Set it to a prediction dict
    key (for example ``"score"``) to evaluate using saved confidence scores.
    """
    pred_dir = Path(pred_dir).expanduser().resolve()
    gt_dir = Path(gt_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve() if output_dir else pred_dir
    if not pred_dir.is_dir():
        raise NotADirectoryError(f"Prediction directory not found: {pred_dir}")
    if not gt_dir.is_dir():
        raise NotADirectoryError(f"GT directory not found: {gt_dir}")
    if max_scenes is not None and max_scenes <= 0:
        raise ValueError("max_scenes must be positive")

    files = _scene_files(pred_dir, scene_list)
    if max_scenes is not None:
        files = files[:max_scenes]
    if not files:
        raise RuntimeError(f"No prediction .pth files found in {pred_dir}")

    predictions, gt_semantic, gt_instance = [], [], []
    skipped = []
    for pred_path in tqdm(files, desc="Loading scenes"):
        try:
            scene_pred, sem_gt, inst_gt = _load_scene(
                pred_path, gt_dir, pred_key, score_key
            )
        except Exception as error:
            if not skip_errors:
                raise
            skipped.append((pred_path.stem, str(error)))
            print(f"SKIP {pred_path.stem}: {error}")
            continue
        predictions.append(scene_pred)
        gt_semantic.append(sem_gt)
        gt_instance.append(inst_gt)
    if not predictions:
        raise RuntimeError("No valid scenes remain for evaluation")

    print(f"Evaluating {len(predictions)} scenes")
    print(f"Predictions: {pred_dir}")
    print(f"Ground truth: {gt_dir}")
    if skipped:
        print(f"Skipped scenes: {len(skipped)}")
    metrics = ScanNet200ClassAgnosticEvaluator(min_region_size).evaluate(
        predictions, gt_semantic, gt_instance, output_path
    )
    metrics["evaluated_scenes"] = len(predictions)
    metrics["skipped_scenes"] = skipped
    print(f"Result file: {output_path / 'result.txt'}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone class-agnostic ScanNet200 instance evaluation."
    )
    parser.add_argument("--pred-dir", default="")
    parser.add_argument("--gt-dir", default="Open3DIS-main/data/Scannet200/Scannet200_3D/val/groundtruth")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--pred-key", default="ins")
    parser.add_argument("--score-key", default=None)
    parser.add_argument("--scene-list", default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--min-region-size", type=int, default=100)
    return parser.parse_args()

# python eval_class_agnostic_scannet200.py \
#     --pred-dir /path/to/predictions \
#     --gt-dir /path/to/groundtruth \
#     --output-dir /path/to/output

def main() -> None:
    args = parse_args()
    evaluate_class_agnostic_scannet200(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        pred_key=args.pred_key,
        score_key=args.score_key,
        scene_list=args.scene_list,
        max_scenes=args.max_scenes,
        skip_errors=args.skip_errors,
        min_region_size=args.min_region_size,
    )


if __name__ == "__main__":
    main()

"""Minimal YOLO-World/GroundingDINO -> box NMS -> SAM2 pipelines."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def add_source_root(path: str | Path | None, extra: Sequence[str] = ()) -> None:
    """Make a locally cloned model repository importable without hard-coded paths."""
    if path is None:
        return
    root = Path(path).expanduser().resolve()
    for candidate in (root, *(root / item for item in extra)):
        value = str(candidate)
        if candidate.exists() and value not in sys.path:
            sys.path.insert(0, value)


def masks_to_rle(masks: np.ndarray | torch.Tensor) -> list[dict[str, Any]]:
    """Encode N x 1 x H x W masks as COCO RLE dictionaries."""
    import pycocotools.mask as mask_util

    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks shaped N×H×W or N×1×H×W, got {masks.shape}")

    encoded = []
    for mask in masks:
        rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8, copy=False)))
        rle["counts"] = rle["counts"].decode("utf-8")
        encoded.append(rle)
    return encoded


def _aligned_select(
    indices: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: list[str],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    indices = indices.to(boxes.device, dtype=torch.long)
    return boxes[indices], scores[indices], [labels[i] for i in indices.cpu().tolist()]


def _valid_box_indices(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    left = boxes[:, 0].clamp(0, width)
    top = boxes[:, 1].clamp(0, height)
    right = boxes[:, 2].clamp(0, width)
    bottom = boxes[:, 3].clamp(0, height)
    valid = (
        ((bottom - top) > 1)
        & ((right - left) > 1)
        & (((bottom - top) * (right - left)) / (width * height) < 0.85)
    )
    return torch.nonzero(valid, as_tuple=False).flatten()


class GroundingDinoDetector:
    """GroundingDINO detector with the original ScanNet/Open3DIS thresholds."""

    name = "grounding"

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda",
        box_threshold: float = 0.3,
        text_threshold: float = 0.3,
        prompt_chunk_size: int = 5,
    ) -> None:
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict

        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.prompt_chunk_size = prompt_chunk_size
        args = SLConfig.fromfile(str(config_path))
        args.device = device
        self.model = build_model(args)
        checkpoint = _torch_load(checkpoint_path)
        self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.model.to(device).eval()

    @staticmethod
    def _transform(image: Image.Image) -> torch.Tensor:
        import groundingdino.datasets.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.RandomResize([800], max_size=1333),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )
        tensor, _ = transform(image, None)
        return tensor

    def detect(
        self, image_path: str | Path, class_names: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        from groundingdino.util.utils import get_phrases_from_posmap

        image = Image.open(image_path).convert("RGB")
        image_tensor = self._transform(image)
        all_boxes, all_scores, all_labels = [], [], []
        for start in range(0, len(class_names), self.prompt_chunk_size):
            names = class_names[start : start + self.prompt_chunk_size]
            caption = ".".join(names).lower().strip()
            if not caption.endswith("."):
                caption += "."
            with torch.no_grad():
                outputs = self.model(
                    image_tensor[None].to(self.device), captions=[caption]
                )
            logits = outputs["pred_logits"].sigmoid()[0].cpu()
            boxes = outputs["pred_boxes"][0].cpu()
            keep = logits.max(dim=1).values > self.box_threshold
            logits = logits[keep]
            boxes = boxes[keep]
            scores = logits.max(dim=1).values
            tokenizer = self.model.tokenizer
            tokenized = tokenizer(caption)
            labels = [
                get_phrases_from_posmap(
                    logit > self.text_threshold, tokenized, tokenizer
                ).replace(".", "")
                for logit in logits
            ]
            if len(boxes):
                all_boxes.append(torch.as_tensor(boxes, dtype=torch.float32))
                all_scores.append(torch.as_tensor(scores, dtype=torch.float32))
                all_labels.extend(str(label) for label in labels)

        if not all_boxes:
            return torch.empty((0, 4)), torch.empty(0), []

        # GroundingDINO returns normalized CXCYWH. Convert exactly once to
        # absolute XYXY before filtering, NMS, and SAM2.
        boxes = torch.cat(all_boxes).to(self.device)
        scores = torch.cat(all_scores).to(self.device)
        width, height = image.size
        boxes = boxes * boxes.new_tensor([width, height, width, height])
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        keep = _valid_box_indices(boxes, width, height)
        boxes, scores, labels = _aligned_select(keep, boxes, scores, all_labels)
        if not len(boxes):
            return boxes, scores, labels
        keep = nms(boxes, scores, iou_threshold=0.5)
        return _aligned_select(keep, boxes, scores, labels)


class YoloWorldDetector:
    """YOLO-World detector using all class names in one inference call."""

    name = "yolo"

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda",
        confidence_threshold: float = 0.05,
        nms_threshold: float = 0.8,
        clip_model: str | Path | None = None,
    ) -> None:
        from mmengine.config import Config
        from mmengine.dataset import Compose
        from mmengine.runner import Runner

        config = Config.fromfile(str(config_path))
        if clip_model is not None:
            clip_model = Path(clip_model).expanduser().resolve()
            if not clip_model.is_dir():
                raise NotADirectoryError(f"CLIP model directory not found: {clip_model}")
            config.model.backbone.text_model.model_name = str(clip_model)
        config.work_dir = str(Path(tempfile.gettempdir()) / "mv3dis_yolo_world")
        config.load_from = str(checkpoint_path)
        self.runner = Runner.from_cfg(config)
        self.runner.call_hook("before_run")
        self.runner.load_or_resume()
        self.runner.model.eval()
        self.pipeline = Compose(config.test_dataloader.dataset.pipeline)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    def detect(
        self, image_path: str | Path, class_names: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        from mmengine.runner.amp import autocast

        texts = [[name.strip()] for name in class_names] + [[" "]]
        data_info = self.pipeline(
            {"img_id": 0, "img_path": str(image_path), "texts": texts}
        )
        data_batch = {
            "inputs": data_info["inputs"].unsqueeze(0),
            "data_samples": [data_info["data_samples"]],
        }
        with autocast(enabled=False), torch.no_grad():
            output = self.runner.model.test_step(data_batch)[0]
        instances = output.pred_instances

        # Match the settings used by yoloworld2_sam2_iou0.8_0.99.
        keep = nms(
            instances.bboxes,
            instances.scores,
            iou_threshold=self.nms_threshold,
        )
        instances = instances[keep]
        instances = instances[
            instances.scores.float() > self.confidence_threshold
        ]
        if len(instances.scores) > 100:
            instances = instances[instances.scores.float().topk(100).indices]

        boxes = instances.bboxes.detach().to(self.device, dtype=torch.float32)
        scores = instances.scores.detach().to(self.device, dtype=torch.float32)
        label_ids = instances.labels.detach().cpu().tolist()
        labels = [texts[int(index)][0] for index in label_ids]
        if not len(boxes):
            return boxes, scores, labels

        # The detector stage above already performs NMS and top-100 filtering.
        # Do not add another stricter NMS here.
        width, height = Image.open(image_path).size
        keep = _valid_box_indices(boxes, width, height)
        return _aligned_select(keep, boxes, scores, labels)


class MaskGenerator:
    """Shared SAM2 stage for either supported box detector."""

    def __init__(
        self,
        detector: GroundingDinoDetector | YoloWorldDetector,
        sam2_config: str,
        sam2_checkpoint: str | Path,
        device: str = "cuda",
    ) -> None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.detector = detector
        self.device = device
        sam2 = build_sam2(
            sam2_config,
            str(sam2_checkpoint),
            device=device,
        )
        self.sam_predictor = SAM2ImagePredictor(sam2)

    @torch.inference_mode()
    def process_image(
        self, image_path: str | Path, class_names: Sequence[str]
    ) -> dict[str, Any] | None:
        boxes, detector_scores, labels = self.detector.detect(image_path, class_names)
        if not len(boxes):
            return None

        image = np.asarray(Image.open(image_path).convert("RGB"))
        self.sam_predictor.set_image(image)
        masks, sam2_scores, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
        masks = np.asarray(masks)
        if masks.size == 0:
            return None
        if masks.ndim == 3:
            masks = masks[:, None, :, :]
        if masks.ndim != 4 or masks.shape[0] != len(boxes):
            raise ValueError(
                f"SAM2 returned masks {masks.shape} for {len(boxes)} boxes"
            )

        # Sort by SAM2 mask score, not detector confidence, while preserving
        # one-to-one alignment. No mask fitting or mask-level NMS is applied.
        sam2_scores = np.asarray(sam2_scores, dtype=np.float32).reshape(-1)
        order = np.argsort(-sam2_scores, kind="stable")
        masks = masks[order]
        sam2_scores = sam2_scores[order]
        detector_scores = detector_scores[torch.from_numpy(order.copy()).to(boxes.device)]
        labels = [labels[index] for index in order.tolist()]

        # Keep the established Open3DIS per-frame format. Labels are used only
        # internally to maintain alignment and are intentionally not serialized.
        return {
            "masks": masks_to_rle(masks),
            "conf": detector_scores.detach().cpu(),
            "score": torch.from_numpy(sam2_scores).cpu(),
        }

    def process_scene(
        self,
        scene_dir: str | Path,
        class_names: Sequence[str],
        image_subdir: str = "color",
        image_interval: int = 10,
        extensions: Sequence[str] = (".jpg", ".jpeg", ".png"),
    ) -> dict[str, dict[str, Any]]:
        from tqdm import tqdm

        image_dir = Path(scene_dir) / image_subdir
        if not image_dir.is_dir():
            raise NotADirectoryError(f"Image directory not found: {image_dir}")
        allowed = {extension.lower() for extension in extensions}
        images = [path for path in image_dir.iterdir() if path.suffix.lower() in allowed]

        def sort_key(path: Path) -> tuple[int, int | str]:
            return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)

        images = sorted(images, key=sort_key)[::image_interval]
        result = {}
        for image_path in tqdm(images, desc=Path(scene_dir).name, leave=False):
            frame = self.process_image(image_path, class_names)
            if frame is not None:
                result[image_path.stem] = frame
        return result


def create_generator(args: Any) -> MaskGenerator:
    """Create a pipeline from argparse-like attributes."""
    add_source_root(args.sam2_root)
    if args.detector == "grounding":
        add_source_root(args.grounding_dino_root)
        detector = GroundingDinoDetector(
            args.detector_config,
            args.detector_checkpoint,
            device=args.device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            prompt_chunk_size=args.prompt_chunk_size,
        )
    else:
        add_source_root(args.yolo_world_root, extra=("third_party/mmyolo",))
        detector = YoloWorldDetector(
            args.detector_config,
            args.detector_checkpoint,
            device=args.device,
            confidence_threshold=args.yolo_conf_threshold,
            nms_threshold=args.yolo_nms_threshold,
            clip_model=args.clip_model,
        )
    return MaskGenerator(
        detector,
        args.sam2_config,
        args.sam2_checkpoint,
        device=args.device,
    )

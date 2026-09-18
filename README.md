# MV3DIS

Official implementation of **MV3DIS: Multi-View Mask Matching via 3D Guides for Zero-Shot 3D Instance Segmentation**.

MV3DIS performs class-agnostic 3D instance segmentation by leveraging multi-view 2D instance masks and 3D superpoints.

This repository provides a refactored and optimized implementation of MV3DIS. We further improve the 2D mask generation with **Grounded-SAM2**, leading to substantially better results than the original implementation. We also provide a **YOLO-World + SAM2** pipeline for faster 2D mask generation.

## Results

Class-agnostic 3D instance segmentation results using Grounded-SAM2 masks:

| Dataset    |       AP |     AP50 |     AP25 |
| ---------- | -------: | -------: | -------: |
| ScanNet200 | **38.2** | **57.1** | **71.5** |
| ScanNetV2  | **41.9** | **62.5** | **77.2** |

## Installation

Clone the repository:

```bash
git clone https://github.com/zybjn/MV3DIS.git
cd MV3DIS
```

We use the following environment for MV3DIS:

```bash
conda create -n mv3dis python=3.10
conda activate mv3dis

pip install torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu118
    
pip install open3d natsort matplotlib tqdm opencv-python scipy plyfile \
    Pillow pycocotools scikit-learn
```

Additional dependencies are required for 2D mask generation. Please refer to [`mask_generator/README.md`](mask_generator/README.md) for mask generation instructions.

## Data Preparation

The ScanNet data organization follows [SAI3D].

Please prepare the ScanNet point clouds, RGB-D frames, camera poses, intrinsics, and superpoints following the instructions in SAI3D.

The 2D mask results should be organized as:

```text
SAM2D_PATH/
└── 2d_pth_score/
    ├── scene0000_00.pth
    ├── scene0000_01.pth
    └── ...
```

Each frame in a scene `.pth` file contains at least:

```python
{
    "masks": ...,   # RLE-encoded masks
    "conf": ...,    # detector confidence
    "score": ...    # SAM2 mask score
}
```

## 2D Mask Generation

We provide two 2D mask generation pipelines:

* **Grounded-SAM2**
* **YOLO-World + SAM2**

Grounded-SAM2 is used for the results reported above.

YOLO-World + SAM2 provides significantly faster 2D mask generation and can be used as an efficient alternative.

We also provide the complete Grounded-SAM2 masks used in our experiments:

**[Download Grounded-SAM2 Masks](https://drive.google.com/file/d/1tTfXHPh30oGlf7MHBWVtcYDRWgMuBLJv/view?usp=drive_link)**

Please refer to [`mask_generator/README.md`](mask_generator/README.md) for mask generation instructions.

## Inference

Run MV3DIS on a single scene:

```bash
python mv3dis_main.py \
    --scene_id scene0435_03 \
    --base_dir /path/to/scannet \
    --scans_dir /path/to/scannet \
    --points_path /path/to/points_path \
    --data2d_path /path/to/scannet/RGB_D/validation \
    --sam2d_path /path/to/mask_results \
    --eval_dir /path/to/output
```

Run the full validation set:

```bash
python mv3dis_main.py \
    --base_dir /path/to/scannet \
    --scans_dir /path/to/scannet \
    --points_path /path/to/points_path \
    --data2d_path /path/to/scannet/RGB_D/validation \
    --sam2d_path /path/to/mask_results \
    --eval_dir /path/to/output
```

Common options:

```text
--view_freq 10
--thres_dis 0.05
--thres_merge 200
```

The predictions are saved to:

```text
<eval_dir>/final/<scene_id>.pth
```

Use the predictions in `final/` for evaluation.

## Evaluation

### ScanNetV2

```bash
python evaluation/evaluate_class_agnostic_instancepy3.py \
    --pred_path /path/to/results/final \
    --gt_path /path/to/scannetv2_gt
```

The ScanNetV2 evaluation follows [SAI3D]. The evaluation code is the same as SAI3D, and the processed ground truth can be obtained following the instructions in the SAI3D repository.

### ScanNet200

```bash
python evaluation/eval_class_agnostic_scannet200.py \
    --pred-dir /path/to/results/final \
    --gt-dir /path/to/scannet200_gt \
    --output-dir /path/to/evaluation_output
```

The ScanNet200 class-agnostic evaluation is built upon [Open3DIS]. Please refer to Open3DIS for ScanNet200 ground-truth preparation.

For data preparation on other datasets, please refer to [SAI3D] and [Open3DIS].

## Acknowledgements

We thank [SAI3D] and [Open3DIS] for their excellent work and codebase.

We also thank GroundingDINO, YOLO-World, SAM2, and ScanNet for their open-source contributions.

[SAI3D]: https://github.com/yd-yin/SAI3D
[Open3DIS]: https://github.com/VinAIResearch/Open3DIS

## Citation

If you find MV3DIS useful in your research, please consider citing:

```bibtex
@InProceedings{Zhao_2026_CVPR,
    author    = {Zhao, Yibo and Zhang, Yigong and Xie, Jin},
    title     = {MV3DIS: Multi-View Mask Matching via 3D Guides for Zero-Shot 3D Instance Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {17916--17926}
}
```

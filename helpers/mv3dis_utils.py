"""Numerical, geometry, mask, and dataset helpers used by MV3DIS.

The code has been refactored by CodeX.
"""

import glob
import json
import math
import os
from collections import defaultdict, deque
from os.path import basename, dirname, join

import cv2
import numpy as np
import open3d as o3d
import plyfile
import pycocotools.mask
import torch
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def get_points_from_ply(ply_path, points_path):
    ply = plyfile.PlyData.read(ply_path)
    xs = np.array(ply["vertex"].data["x"])[:, None]
    ys = np.array(ply["vertex"].data["y"])[:, None]
    zs = np.array(ply["vertex"].data["z"])[:, None]
    points = np.concatenate((xs, ys, zs), axis=-1)
    np.savetxt(join(dirname(points_path), "points.pts"), points)


def get_splits(base_dir, scannetpp=False):
    if scannetpp:
        train_split_path = join(base_dir, "splits/nvs_sem_train.txt")
        val_split_path = join(base_dir, "splits/nvs_sem_val.txt")
    else:
        train_split_path = join(base_dir, "meta_data/scannetv2_train.txt")
        val_split_path = join(base_dir, "meta_data/scannetv2_val.txt")

    with open(train_split_path, "r") as f:
        train_split = f.readlines()
    train_split = [s.strip() for s in train_split]

    with open(val_split_path, "r") as f:
        val_split = f.readlines()
    val_split = [s.strip() for s in val_split]

    return train_split, val_split


def construct_saving_name(args):
    save_name = f"points_objness_label_{args.mask_name}_connect({args.thres_connect[0]},{args.thres_connect[-1]},{len(args.thres_connect)}).pts"
    if args.thres_trunc > 0:
        save_name = save_name.replace(".pts", f"trunc{args.thres_trunc}.pts")
    save_name = args.similar_metric + "_" + save_name
    if args.max_neighbor_distance is not None:
        save_name = save_name.replace(".pts", f"_depth{args.max_neighbor_distance}.pts")
    if args.thres_merge > 0:
        save_name = f"merge{args.thres_merge}_" + save_name
    if args.text is not None:
        save_name = f"{args.text}_" + save_name
    if args.test:
        save_name = "test_" + save_name

    return save_name


def num_to_natural(group_ids, void_number=-1):
    """
    code credit: SAM3D
    """
    if void_number == -1:
        # [-1,-1,0,3,4,0,6] -> [-1,-1,0,1,2,0,3]
        if np.all(group_ids == -1):
            return group_ids
        array = group_ids.copy()

        unique_values = np.unique(array[array != -1])
        mapping = np.full(np.max(unique_values) + 2, -1)

        mapping[unique_values + 1] = np.arange(len(unique_values))
        array = mapping[array + 1]

    elif void_number == 0:
        if np.all(group_ids == 0):
            return group_ids
        array = group_ids.copy()

        unique_values = np.unique(array[array != 0])
        mapping = np.full(np.max(unique_values) + 2, 0)
        mapping[unique_values] = np.arange(len(unique_values)) + 1
        array = mapping[array]
    else:
        raise Exception("void_number must be -1 or 0")
    print("array.min()", array.min())
    return array, mapping, unique_values



# -----------------------------------------------------------------------------
# Affinity and similarity
# -----------------------------------------------------------------------------



def get_similar_confidence_matrix_handle(
    seg_neighbors,
    seg_ids,
    seg_seen,
    points_label,
    similar_metric,
    seg_seen_num,
    seg_member_count,
    pred_score,
    thres_trunc,
):
    seg_seen_num = seg_seen_num.astype(float)
    view_num = seg_seen.shape[1]
    seg_num = seg_seen.shape[0]
    total_mask = sum(array.shape[1] for array in points_label)

    ssp_vector = np.zeros((seg_num, total_mask), dtype=np.float32)
    frame_index = np.zeros(total_mask, dtype=np.int32)

    print("total_mask", total_mask)
    beg = 0
    for m in tqdm(range(view_num)):
        score = pred_score[m]
        plabels = points_label[m]
        ssp_in_frame_visnum = seg_seen_num[:, m]
        label_range = plabels.shape[1] + 1
        if label_range < 2:
            continue

        seglabels = np.zeros((seg_num, label_range), dtype=np.float32)

        np.add.at(
            seglabels, (seg_ids[:, None], np.arange(1, plabels.shape[1] + 1)), plabels
        )

        nonzero_seglabels = seglabels[:, 1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            visibility_ratio = nonzero_seglabels / ssp_in_frame_visnum[:, np.newaxis]
        visibility_ratio[np.isnan(visibility_ratio)] = 0
        visibility_ratio *= score
        visibility_ratio = np.divide(
            visibility_ratio,
            np.clip(np.linalg.norm(visibility_ratio, axis=-1), 1e-8, np.inf)[:, None],
        )
        ssp_vector[:, beg : beg + visibility_ratio.shape[1]] = visibility_ratio
        frame_index[beg : beg + visibility_ratio.shape[1]] = m
        beg += visibility_ratio.shape[1]

    similar_sum = np.zeros([seg_num, seg_num], dtype=np.float32)
    confidence_sum = np.zeros([seg_num, seg_num], dtype=np.float32)
    one_view_similar = np.zeros([seg_num, seg_num], dtype=np.float64)
    one_view_confidence = np.zeros([seg_num, seg_num], dtype=np.float64)

    for m in tqdm(range(view_num)):
        one_view_similar.fill(0.0)
        one_view_confidence.fill(0.0)
        nonzero_seglabels = ssp_vector[:, np.where(frame_index == m)[0]]

        for i in range(seg_neighbors.shape[0]):
            if seg_neighbors[i].nonzero()[0].size == 0:
                continue
            neighbors_labels = nonzero_seglabels[seg_neighbors[i]]
            one_view_similar[i, seg_neighbors[i]] = np.sum(
                (nonzero_seglabels[i] * neighbors_labels), axis=1
            )

            one_view_confidence[i, seg_neighbors[i]] = (
                seg_seen[seg_neighbors[i], m] * seg_seen[i, m]
            )

        similar_sum = similar_sum + one_view_similar * one_view_confidence
        confidence_sum += one_view_confidence

    return similar_sum, confidence_sum


def get_similar_confidence_matrix_handle_with_torch(
    ch0,
    cw0,
    mask_list,
    points_seen,
    seg_neighbors,
    seg_ids,
    seg_seen,
    similar_metric,
    seg_seen_num,
    seg_member_count,
    pred_score,
    thres_trunc,
    weight_points,
):
    seg_ids = torch.tensor(seg_ids, device=device, dtype=torch.int32)
    seg_seen_num = torch.tensor(seg_seen_num, dtype=torch.float32)
    seg_seen = torch.tensor(seg_seen, device=device, dtype=torch.float32)
    gpu_weight_points = torch.tensor(weight_points, device="cpu", dtype=torch.float32)

    points_num = seg_ids.shape[0]
    seg_num = seg_seen.shape[0]

    similar_sum = torch.zeros((seg_num, seg_num), dtype=torch.float32, device="cuda")
    confidence_sum = torch.zeros((seg_num, seg_num), dtype=torch.float32, device="cuda")
    one_view_similar = torch.zeros(
        (seg_num, seg_num), dtype=torch.float32, device="cuda"
    )
    one_view_confidence = torch.zeros(
        (seg_num, seg_num), dtype=torch.float32, device="cuda"
    )

    neighbors_list = [
        torch.tensor(row.nonzero()[0], device="cuda") for row in seg_neighbors
    ]

    for m in tqdm(range(len(mask_list))):
        mas = mask_list[m]
        mask_values = mas[:, ch0[:, m], cw0[:, m]].T.astype(int)
        points_label = mask_values * points_seen[:, m, None]

        plabels = torch.tensor(points_label, dtype=torch.float32, device="cuda")
        ssp_in_frame_visnum = seg_seen_num[:, m].to("cuda")
        weights = gpu_weight_points[:, m].to("cuda")
        label_range = plabels.shape[1] + 1

        if label_range < 2:
            continue

        seglabels = torch.zeros(
            (seg_num, label_range), dtype=torch.float32, device="cuda"
        )
        seg_label_weights = torch.zeros(
            (seg_num, label_range), dtype=torch.float32, device="cuda"
        )

        for label_idx in range(plabels.shape[1]):
            label_tensor = torch.zeros(
                (points_num, label_range), dtype=torch.float32, device="cuda"
            ).scatter_(
                1,
                torch.full(
                    (points_num, 1), label_idx + 1, dtype=torch.long, device="cuda"
                ),
                plabels[:, label_idx].unsqueeze(1),
            )
            seglabels.index_add_(0, seg_ids, label_tensor)

            weight_tensor = (plabels[:, label_idx] * weights).unsqueeze(1)
            seg_label_weights.index_add_(
                0,
                seg_ids,
                torch.zeros((points_num, label_range), device="cuda").scatter_(
                    1,
                    torch.full(
                        (points_num, 1), label_idx + 1, dtype=torch.long, device="cuda"
                    ),
                    weight_tensor,
                ),
            )

        nonzero_seglabels = seg_label_weights[:, 1:]
        visibility_ratio = torch.div(nonzero_seglabels, ssp_in_frame_visnum[:, None])
        visibility_ratio[torch.isnan(visibility_ratio)] = 0

        visibility_ratio = visibility_ratio / torch.clamp(
            torch.norm(visibility_ratio, dim=-1, keepdim=True), min=1e-8
        )

        one_view_similar.zero_()
        one_view_confidence.zero_()

        for i in range(seg_num):
            if len(neighbors_list[i]) == 0:
                continue

            neighbors_labels = visibility_ratio[neighbors_list[i]]
            one_view_similar[i, neighbors_list[i]] = torch.sum(
                visibility_ratio[i] * neighbors_labels, dim=1
            )
            one_view_confidence[i, neighbors_list[i]] = (
                seg_seen[neighbors_list[i], m] * seg_seen[i, m]
            )

        similar_sum = similar_sum + one_view_similar * one_view_confidence
        confidence_sum += one_view_confidence

    return similar_sum.cpu().numpy(), confidence_sum.cpu().numpy()


def multiprocess_get_similar_confidence_matrix(
    seg_seen,
    seg_neighbors,
    seg_ids,
    points_label_vector,
    similar_metric,
    seg_seen_num,
    seg_member_count,
    pred_score,
    thres_trunc=0.0,
    process_num=0,
):
    """
    :param seg_seen: (S, M), ratio of seen part of primitives in every view
    :param seg_neighbors: (S, S), 1 if two segs are neighbors
    :param seg_ids: (N, ), seg id of every point
    :param points_label: (N, M), labels of all points in all views

    :return similar: wighted sum of how much the two primitives are similar in every view
    :return confidence: sum of wight of how much we can trust the similar score in every view
    """
    similar_sum, confidence_sum = get_similar_confidence_matrix_handle_with_torch(
        seg_neighbors,
        seg_ids,
        seg_seen,
        points_label_vector,
        similar_metric,
        seg_seen_num,
        seg_member_count,
        pred_score,
        thres_trunc,
    )
    return similar_sum, confidence_sum


@torch.inference_mode()
def old_torch_get_similar_confidence_matrix(
    seg_neighbors,
    seg_ids,
    seg_seen,
    points_label,
    similar_metric,
    thres_trunc=0,
    weight_points=None,
):
    """
    :param seg_seen: (S, M), ratio of seen part of primitives in every view
    :param seg_neighbors: (S, S), 1 if two segs are neighbors
    :param seg_ids: (N, ), seg id of every point
    :param points_label: (N, M), labels of all points in all views

    :return similar: wighted sum of how much the two primitives are similar in every view
    :return confidence: sum of wight of how much we can trust the similar score in every view
    """
    view_num = seg_seen.shape[1]
    seg_num = seg_seen.shape[0]

    print("preparing data on gpu")
    gpu_seg_ids = torch.tensor(seg_ids, device=device, dtype=torch.int32)
    gpu_points_label = torch.tensor(points_label, device="cpu", dtype=torch.float32)
    gpu_seg_neighbors = torch.tensor(seg_neighbors, device=device, dtype=torch.bool)
    gpu_seg_seen = torch.tensor(seg_seen, device=device, dtype=torch.float32)

    similar_sum = torch.zeros([seg_num, seg_num], device=device, dtype=torch.float32)
    confidence_sum = torch.zeros([seg_num, seg_num], device=device, dtype=torch.float32)
    one_view_similar = torch.zeros(
        [seg_num, seg_num], device=device, dtype=torch.float32
    )
    one_view_confidence = torch.zeros(
        [seg_num, seg_num], device=device, dtype=torch.float32
    )

    for m in tqdm(range(view_num)):
        plabels = gpu_points_label[:, m].to(device)
        one_view_similar.fill_(0.0)
        one_view_confidence.fill_(0.0)

        label_range = int(torch.max(plabels) + 1)
        if label_range < 2:
            continue

        seglabels = torch.zeros(
            [seg_num, label_range], device=device, dtype=torch.float32
        )

        p_labels_segids = torch.stack([plabels, gpu_seg_ids], dim=1)

        unique_labels_segids, inverse_indices, unique_counts = torch.unique(
            p_labels_segids, return_counts=True, return_inverse=True, dim=0
        )

        unique_labels_segids = unique_labels_segids.type(torch.long)
        unique_counts = unique_counts.type(torch.float32)

        seglabels[unique_labels_segids[:, 1], unique_labels_segids[:, 0]] = (
            unique_counts
        )

        nonzero_seglabels = seglabels[:, 1:]

        if similar_metric == "2-norm":
            nonzero_seglabels = torch.divide(
                nonzero_seglabels,
                torch.clamp(torch.norm(nonzero_seglabels, dim=-1), 1e-8)[:, None],
            )
        else:
            raise NotImplementedError
        del unique_counts, unique_labels_segids
        del p_labels_segids, plabels

        batch_size = 200
        for start_id in range(0, seg_num, batch_size):
            if seg_neighbors[start_id : start_id + batch_size].nonzero()[0].size == 0:
                continue
            all_neighbors_mask = (
                torch.sum(gpu_seg_neighbors[start_id : start_id + batch_size], dim=0)
                > 0
            )

            neighbors_labels = nonzero_seglabels[all_neighbors_mask]

            one_view_similar[start_id : start_id + batch_size, all_neighbors_mask] = (
                torch_calcu_all_similar(
                    nonzero_seglabels[start_id : start_id + batch_size],
                    neighbors_labels,
                    similar_metric=similar_metric,
                    thres_trunc=thres_trunc,
                )
            )

        one_view_confidence = gpu_seg_seen[:, m][:, None] @ gpu_seg_seen[:, m][None, :]

        del seglabels, nonzero_seglabels

        confidence_sum += one_view_confidence
        similar_sum += one_view_similar * one_view_confidence

    return [similar_sum.cpu().numpy(), confidence_sum.cpu().numpy()]


def _seglabels_from_unique(plabels, gpu_seg_ids, seg_num, label_range):
    """Reference counter for (segment, point-label) pairs."""
    seglabels = torch.zeros(
        [seg_num, label_range], device=plabels.device, dtype=torch.float32
    )
    p_labels_segids = torch.stack([plabels, gpu_seg_ids], dim=1)
    unique_labels_segids, unique_counts = torch.unique(
        p_labels_segids, return_counts=True, dim=0
    )
    unique_labels_segids = unique_labels_segids.type(torch.long)
    unique_counts = unique_counts.type(torch.float32)
    seglabels[unique_labels_segids[:, 1], unique_labels_segids[:, 0]] = unique_counts
    return seglabels


def _seglabels_from_bincount(plabels, gpu_seg_ids_long, seg_num, label_range):
    """Count (segment, point-label) pairs through a dense linear index."""
    labels_long = plabels.to(torch.long)
    linear_ids = gpu_seg_ids_long * label_range + labels_long
    counts = torch.bincount(linear_ids, minlength=seg_num * label_range)
    return counts.reshape(seg_num, label_range).to(torch.float32)


def _label_ranges_from_points_label(points_label):
    """Return the exact per-frame max label plus one on the CPU."""
    return np.max(np.asarray(points_label), axis=0).astype(np.int64) + 1


@torch.inference_mode()
def _torch_get_similar_confidence_matrix_impl(
    seg_neighbors,
    seg_ids,
    seg_seen,
    points_label,
    similar_metric,
    thres_trunc,
    weight_points,
    seglabel_builder,
    upload_points_label,
):
    """Optimized equivalent of ``old_torch_get_similar_confidence_matrix``.

    The pair semantics are unchanged: each 200-row batch is compared against
    the ordered union of graph-neighbor columns for that batch.
    """
    view_num = seg_seen.shape[1]
    seg_num = seg_seen.shape[0]

    print("preparing data on gpu")
    gpu_seg_ids = torch.tensor(seg_ids, device=device, dtype=torch.int32)
    gpu_seg_ids_long = (
        gpu_seg_ids.to(torch.long)
        if seglabel_builder is _seglabels_from_bincount
        else None
    )
    if upload_points_label:
        label_ranges = _label_ranges_from_points_label(points_label)
        gpu_points_label = torch.as_tensor(
            points_label, device=device, dtype=torch.float32
        )
    else:
        label_ranges = None
        gpu_points_label = torch.tensor(points_label, device="cpu", dtype=torch.float32)
    gpu_seg_neighbors = torch.tensor(seg_neighbors, device=device, dtype=torch.bool)
    gpu_seg_seen = torch.tensor(seg_seen, device=device, dtype=torch.float32)

    similar_sum = torch.zeros([seg_num, seg_num], device=device, dtype=torch.float32)
    confidence_sum = torch.zeros([seg_num, seg_num], device=device, dtype=torch.float32)
    one_view_similar = torch.zeros(
        [seg_num, seg_num], device=device, dtype=torch.float32
    )

    batch_size = 200
    batch_neighbors = []
    for batch_start in range(0, seg_num, batch_size):
        batch_end = min(batch_start + batch_size, seg_num)
        if seg_neighbors[batch_start:batch_end].nonzero()[0].size == 0:
            continue
        all_neighbors_mask = (
            torch.sum(gpu_seg_neighbors[batch_start:batch_end], dim=0) > 0
        )
        neighbor_indices = torch.nonzero(all_neighbors_mask, as_tuple=False).flatten()
        batch_neighbors.append(
            (batch_start, batch_end, all_neighbors_mask, neighbor_indices)
        )

    for view_index in tqdm(range(view_num)):
        if upload_points_label:
            plabels = gpu_points_label[:, view_index]
            label_range = int(label_ranges[view_index])
        else:
            plabels = gpu_points_label[:, view_index].to(device)
            label_range = int(torch.max(plabels) + 1)
        one_view_similar.fill_(0.0)

        if label_range < 2:
            continue

        if seglabel_builder is _seglabels_from_bincount:
            seglabels = seglabel_builder(
                plabels, gpu_seg_ids_long, seg_num, label_range
            )
        else:
            seglabels = seglabel_builder(plabels, gpu_seg_ids, seg_num, label_range)

        nonzero_seglabels = seglabels[:, 1:]
        if similar_metric == "2-norm":
            nonzero_seglabels = torch.divide(
                nonzero_seglabels,
                torch.clamp(torch.norm(nonzero_seglabels, dim=-1), 1e-8)[:, None],
            )
        else:
            raise NotImplementedError
        del plabels

        for (
            batch_start,
            batch_end,
            all_neighbors_mask,
            neighbor_indices,
        ) in batch_neighbors:
            neighbors_labels = nonzero_seglabels[neighbor_indices]
            one_view_similar[batch_start:batch_end, all_neighbors_mask] = (
                torch_calcu_all_similar(
                    nonzero_seglabels[batch_start:batch_end],
                    neighbors_labels,
                    similar_metric=similar_metric,
                    thres_trunc=thres_trunc,
                )
            )

        view_visibility = gpu_seg_seen[:, view_index]
        one_view_confidence = view_visibility[:, None] @ view_visibility[None, :]

        del seglabels, nonzero_seglabels

        confidence_sum += one_view_confidence
        similar_sum += one_view_similar * one_view_confidence

    return [similar_sum.cpu().numpy(), confidence_sum.cpu().numpy()]


def old_torch_get_similar_confidence_matrix_unique(
    seg_neighbors,
    seg_ids,
    seg_seen,
    points_label,
    similar_metric,
    thres_trunc=0,
    weight_points=None,
):
    """Priority 1 production implementation with torch.unique counting."""
    return _torch_get_similar_confidence_matrix_impl(
        seg_neighbors,
        seg_ids,
        seg_seen,
        points_label,
        similar_metric,
        thres_trunc,
        weight_points,
        _seglabels_from_unique,
        False,
    )


def old_torch_get_similar_confidence_matrix_per_frame_transfer(
    seg_neighbors,
    seg_ids,
    seg_seen,
    points_label,
    similar_metric,
    thres_trunc=0,
    weight_points=None,
):
    """Priority 6 implementation with one CPU-to-GPU copy per frame."""
    return _torch_get_similar_confidence_matrix_impl(
        seg_neighbors,
        seg_ids,
        seg_seen,
        points_label,
        similar_metric,
        thres_trunc,
        weight_points,
        _seglabels_from_bincount,
        False,
    )


def torch_get_similar_confidence_matrix(
    seg_neighbors,
    seg_ids,
    seg_seen,
    points_label,
    similar_metric,
    thres_trunc=0,
    weight_points=None,
):
    """Production implementation using exact linear-index pair counts.

    Point labels are uploaded once, while the historical 200-row batch and
    neighbor-union pair semantics remain unchanged.
    """
    return _torch_get_similar_confidence_matrix_impl(
        seg_neighbors,
        seg_ids,
        seg_seen,
        points_label,
        similar_metric,
        thres_trunc,
        weight_points,
        _seglabels_from_bincount,
        True,
    )


def calcu_similar(label0, labels1, similar_metric, thres_trunc=0):
    """
    :param label0: (lr-1,), label distribuion of one seg
    :param labels1: (neigh,lr-1), label distribution of neighbors of the seg above
    """

    if similar_metric == "2-norm":

        def similar_func(x):
            return (2 - x**2) / 2.0

        if np.sum(labels1 != 0) and np.sum(label0 != 0):
            dis = np.sum((labels1 * label0), axis=1)

        else:
            dis = np.sum((labels1 * label0), axis=1)

        similar = dis
    elif similar_metric == "1-norm":
        dis = np.linalg.norm((labels1 - label0), ord=1, axis=-1)

        def similar_func(x):
            return (2.0 - x) / 2.0

        similar = similar_func(dis)
    elif similar_metric == "inf-norm":
        dis = np.linalg.norm((labels1 - label0), ord=1, axis=-1)

        def similar_func(x):
            return (2.0 - x) / 2.0

        similar = similar_func(dis)
    elif similar_metric == "Hellinger":
        hellinger_div = np.linalg.norm((labels1**0.5 - label0**0.5), axis=-1) / (2**0.5)
        similar = 1.0 - hellinger_div
    elif similar_metric == "cosine":

        def cosine_similarity(a, b):
            dot_product = np.sum(a * b, axis=-1)
            norm_a = np.linalg.norm(a, axis=-1)
            norm_b = np.linalg.norm(b, axis=-1)

            denom = np.clip(norm_a * norm_b, a_min=1e-10, a_max=None)
            return np.clip(dot_product / denom, 0, 1)

        similar = cosine_similarity(labels1, label0)

    else:
        assert 0, "invalid similarity metric!"

    if thres_trunc > 0:
        similar[similar < thres_trunc] = 0

    return similar



@torch.inference_mode()
def old_torch_calcu_all_similar(
    labels1, labels2, similar_metric="2-norm", thres_trunc=0
):
    """Production pairwise similarity implementation using broadcasting."""
    if similar_metric == "2-norm":
        dis = torch.norm(labels1[:, None, :] - labels2[None, :, :], dim=-1)

        def similar_func(x):
            return (2 - x**2) / 2.0
    else:
        raise NotImplementedError

    similar = similar_func(dis)
    if thres_trunc > 0:
        similar[similar < thres_trunc] = 0

    return similar


@torch.inference_mode()
def torch_calcu_all_similar_gemm(
    labels1, labels2, similar_metric="2-norm", thres_trunc=0
):
    """Experimental algebraically equivalent GEMM implementation."""
    if similar_metric == "2-norm":
        a_norm = (labels1 * labels1).sum(dim=1)
        b_norm = (labels2 * labels2).sum(dim=1)
        similar = (
            1
            - 0.5 * a_norm[:, None]
            - 0.5 * b_norm[None, :]
            + torch.matmul(labels1, labels2.T)
        )
    else:
        raise NotImplementedError

    if thres_trunc > 0:
        similar[similar < thres_trunc] = 0

    return similar


# Broadcasting remains the production implementation; GEMM is experimental.

torch_calcu_all_similar = old_torch_calcu_all_similar


# -----------------------------------------------------------------------------
# Coordinate transforms
# -----------------------------------------------------------------------------


def world2cam_pixel(points_world, color_intrinsic, depth_intrinsic, pose):
    """project N points to M images

    :param points_world: (N, 3)
    :param color_intrinsic, depth_intrinsic: (M, 3, 3) camera intrinsics
    :param pose: (M, 4, 4)
    :return: points_cam(N,M,3), color_points_pixel(N,M,2), depth_points_pixel(N,M,2)
    """
    points_world_homo = np.concatenate(
        (points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)), 1
    )

    points_cam_homo = np.matmul(
        np.linalg.inv(pose)[None], points_world_homo[:, None, :, None]
    )
    points_cam_homo = points_cam_homo[..., 0]

    points_cam = np.divide(points_cam_homo, points_cam_homo[..., [-1]])[..., :-1]

    color_points_pixel_homo = np.matmul(color_intrinsic, points_cam[..., None])
    depth_points_pixel_homo = np.matmul(depth_intrinsic, points_cam[..., None])

    color_points_pixel_homo = color_points_pixel_homo[..., 0]
    depth_points_pixel_homo = depth_points_pixel_homo[..., 0]

    color_points_pixel = (
        np.divide(
            color_points_pixel_homo,
            np.clip(color_points_pixel_homo[..., [-1]], 1e-8, np.inf),
        )[..., :-1]
        .round()
        .astype(int)
    )
    depth_points_pixel = (
        np.divide(
            depth_points_pixel_homo,
            np.clip(depth_points_pixel_homo[..., [-1]], 1e-8, np.inf),
        )[..., :-1]
        .round()
        .astype(int)
    )

    return points_cam, color_points_pixel, depth_points_pixel


@torch.inference_mode()
def torch_world2cam_pixel(
    points_world_all: np.array,
    color_intrinsic: np.array,
    depth_intrinsic: np.array,
    pose: np.array,
):
    """project N (1e6) points to M (1e3) images

    :param points_world: (N, 3)
    :param color_intrinsic, depth_intrinsic: (M, 3, 3) the intrinsics of color and depth camera
    :param pose: (M, 4, 4)
    :return: points_cam(n,M,3), color_points_pixel(n,M,2), depth_points_pixel(n,M,2)
    """
    batch_size = 10000

    color_intrinsic = torch.tensor(color_intrinsic, device=device, dtype=torch.float32)
    depth_intrinsic = torch.tensor(depth_intrinsic, device=device, dtype=torch.float32)
    pose = torch.tensor(pose, device=device, dtype=torch.float32)
    pose_inv = torch.linalg.inv(pose)
    del pose

    N = points_world_all.shape[0]
    M = pose_inv.shape[0]

    final_points_cam = np.zeros((N, M, 3), dtype=np.float32)
    final_color_points_pixel = np.zeros((N, M, 2), dtype=np.int32)
    final_depth_points_pixel = np.zeros((N, M, 2), dtype=np.int32)

    for batch_start in tqdm(range(0, points_world_all.shape[0], batch_size)):
        points_world = torch.tensor(
            points_world_all[batch_start : batch_start + batch_size],
            device=device,
            dtype=torch.float32,
        )
        points_world_homo = torch.cat(
            (
                points_world,
                torch.ones(
                    (points_world.shape[0], 1), dtype=torch.float32, device=device
                ),
            ),
            1,
        )

        points_cam_homo = torch.matmul(
            pose_inv[None], points_world_homo[:, None, :, None]
        )
        points_cam_homo = points_cam_homo[..., 0]

        points_cam = torch.div(points_cam_homo[..., :-1], points_cam_homo[..., [-1]])

        color_points_pixel_homo = torch.matmul(color_intrinsic, points_cam[..., None])
        depth_points_pixel_homo = torch.matmul(depth_intrinsic, points_cam[..., None])

        color_points_pixel_homo = color_points_pixel_homo[..., 0]
        depth_points_pixel_homo = depth_points_pixel_homo[..., 0]

        color_points_pixel = (
            torch.div(
                color_points_pixel_homo[..., :-1],
                torch.clip(color_points_pixel_homo[..., [-1]], min=1e-8),
            )
            .round()
            .to(torch.int32)
        )
        depth_points_pixel = (
            torch.div(
                depth_points_pixel_homo[..., :-1],
                torch.clip(depth_points_pixel_homo[..., [-1]], min=1e-8),
            )
            .round()
            .to(torch.int32)
        )

        final_points_cam[batch_start : batch_start + batch_size] = (
            points_cam.cpu().numpy()
        )
        final_color_points_pixel[batch_start : batch_start + batch_size] = (
            color_points_pixel.cpu().numpy()
        )
        final_depth_points_pixel[batch_start : batch_start + batch_size] = (
            depth_points_pixel.cpu().numpy()
        )

    torch.cuda.empty_cache()
    return (final_points_cam, final_color_points_pixel, final_depth_points_pixel)


def batch_pixel2camera(intrinsics, depths):
    """unproject pixels to camera coordinate

    :param intrinsics: (b, 3, 3)
    :param depths: (b, H, W)
    :return: points_camera (b, H*W, 3)
    """
    cx, cy, fx, fy = (
        intrinsics[:, 0, 2],
        intrinsics[:, 1, 2],
        intrinsics[:, 0, 0],
        intrinsics[:, 1, 1],
    )

    b, H, W = depths.shape
    u_base = np.tile(np.arange(W), (H, 1))[None]
    v_base = np.tile(np.arange(H)[:, np.newaxis], (1, W))[None]
    X = (u_base - cx[:, None, None]) * depths / fx[:, None, None]
    Y = (v_base - cy[:, None, None]) * depths / fy[:, None, None]
    coord_camera = np.stack((X, Y, depths), axis=-1).astype(np.float32)
    points_camera = coord_camera.reshape((b, -1, 3))

    return points_camera


def batch_camera2world(points_camera, pose):
    """transform points from camera coordinate to world coordinate

    :param points_camera: (b, N, 3)
    :param pose: (b, 4, 4)
    :return: points_world (b, N, 3)
    """
    points_local_homo = np.concatenate(
        (points_camera, np.ones(points_camera.shape[:2], dtype=np.float32)[..., None]),
        axis=-1,
    )
    points_world_homo = np.matmul(pose, points_local_homo.transpose(0, 2, 1)).transpose(
        0, 2, 1
    )
    points_world = np.divide(
        points_world_homo, np.clip(points_world_homo[..., [-1]], 1e-8, np.inf)
    )[..., :-1]
    return points_world


"""
================================================================================================
utils for matterport dataset
================================================================================================
"""


def get_matterport_color(color_path):
    color = cv2.imread(color_path).astype(np.float32)
    return color


def get_matterport_depth(color_path):
    depth_path = color_path.replace('color', 'depth').replace('.jpg', '.png')
    oldbase = basename(depth_path).split('_')
    oldbase[1] = oldbase[1].replace('i', 'd')
    newbase = '_'.join(oldbase)
    depth_path = join(dirname(depth_path), newbase)

    depth = cv2.imread(depth_path, -1).astype(np.float32)
    if depth.ndim != 2:
        depth = depth[..., 0]
    # set invalid distance to zero
    depth[depth == 65535] = 0
    # depth in millimeters, transfer to meters
    depth /= 4000
    return depth


def get_matterport_intrinsic(color_path):
    intrinsic = np.loadtxt(color_path.replace('color', 'intrinsic').replace('.jpg', '.txt')).astype(np.float32)
    return intrinsic


def get_matterport_pose(color_path):
    pose = np.loadtxt(color_path.replace('color', 'pose').replace('.jpg', '.txt')).astype(np.float32)
    return pose


def get_matterport_mask(base_dir, scene_id, color_name, mask_name, region_id='region4'):
    mask_dir = join(base_dir, '2D_masks', scene_id)
    mask_path = join(mask_dir, mask_name, f'maskraw_{color_name.replace(".jpg", ".png")}')
    mask_raw = cv2.imread(mask_path, -1).astype(np.float32)
    # mask_color_path = join(mask_dir, mask_name, f'maskcolor_{basename(color_path).replace(".jpg", ".png")}')
    # dest_dir = join(mask_dir, mask_name, region_id)
    # mask_color_dest_path = join(dest_dir, f'maskcolor_{basename(color_path).replace(".jpg", ".png")}')
    # color_dest_path = join(dest_dir, basename(color_path))

    # shutil.copy(mask_color_path,mask_color_dest_path)
    # shutil.copy(color_path,color_dest_path)

    return mask_raw


def get_matterport_semantic_mask(color_path, semantic):
    semantic_dir = join(dirname(dirname(color_path)), 'results', 'everything', 'semantic', semantic)
    semantic_path = join(semantic_dir, f'{basename(color_path).replace(".jpg", ".png")}')
    # from 0 to category_num, 0 means no class
    semantic_mask = cv2.imread(semantic_path, -1).astype(np.float32)
    # print(np.unique(semantic_mask))
    return semantic_mask


def get_points_from_openscene_pth(base_dir, scene_id):

    def read_pth(points_list):
        p_lst, c_lst, v_lst = [], [], []
        for points_path in points_list:
            points, color, vertex_labels = torch.load(points_path)
            p_lst.append(points)
            c_lst.append(color)
            v_lst.append(vertex_labels)

        p_lst = np.vstack(p_lst)
        c_lst = np.vstack(c_lst)
        v_lst = np.concatenate(v_lst)

        return p_lst, c_lst, v_lst

    points_list = sorted(glob.glob(
        join(base_dir, 'matterport_3d', '**', f'{scene_id}*.pth'), recursive=True))
    points = read_pth(points_list)[0]
    np.savetxt(join(base_dir, 'matterport_2d', scene_id, 'points.pts'), points)




# -----------------------------------------------------------------------------
# ScanNet data access
# -----------------------------------------------------------------------------


def get_scannet_depth_mask(color_path, base_dir, scene_id, mask_group_name):
    mask_dir = join(base_dir, '2D_masks', scene_id, 'depth_' + mask_group_name)
    color_name = basename(color_path)
    # mask_name = 'maskraw_' + color_name.replace('jpg','png')
    mask_name = 'maskraw_' + color_name.replace('jpg', 'png')
    mask_path = join(mask_dir, mask_name)
    mask_raw = cv2.imread(mask_path, -1).astype(np.float32)
    # print("mask:",mask_raw.shape)  #(968, 1296)
    return mask_raw


def get_scannet_pose(color_path):
    pose_path = color_path.replace("color", "pose")
    pose = np.loadtxt(pose_path.replace(".jpg", ".txt")).astype(np.float32)
    return pose



def get_scannet_depth(color_path):
    depth_path = color_path.replace("color", "depth")
    depth_path = depth_path.replace(".jpg", ".png")
    depth = cv2.imread(depth_path, -1).astype(np.float32)

    depth /= 1000.0

    return depth


def get_replic_depth(color_path):
    depth_path = color_path.replace("color", "depth")
    depth_path = depth_path.replace(".jpg", ".png")
    depth = cv2.imread(depth_path, -1).astype(np.float32)

    depth /= 6553.5

    return depth


def get_mask_list(sam2d_path, color_path, scene_id, mask_group_name):
    mask_dir = join(sam2d_path, scene_id, mask_group_name)
    color_name = basename(color_path)
    mask_name = "maskdict_" + color_name.replace("jpg", "npy")
    mask_path = join(mask_dir, mask_name)

    npy_file_path = os.path.join(mask_path)

    loaded_data = np.load(npy_file_path, allow_pickle=True).item()

    binary_values_loaded = loaded_data["binary_values"]
    predicted_iou_loaded = loaded_data["predicted_iou"]
    return np.stack(binary_values_loaded, 0), predicted_iou_loaded


def get_mask_list_grounding_sam(groundedsam_data_dict, fram_id):
    if str(fram_id) in groundedsam_data_dict:
        encoded_masks = groundedsam_data_dict[str(fram_id)]["masks"]
    else:
        return None, None
    masks = []
    for mask in encoded_masks:
        masks.append(pycocotools.mask.decode(mask))
    masks = np.stack(masks, axis=0)
    predicted_iou_loaded = np.array(
        [t.numpy() for t in groundedsam_data_dict[str(fram_id)]["conf"]]
    )

    return masks, predicted_iou_loaded


def get_mask_list_grounding_sam_fast(groundedsam_data_dict, fram_id):
    """Decode the same masks while converting all confidences in one step."""
    frame_data = groundedsam_data_dict.get(str(fram_id))
    if frame_data is None:
        return None, None
    masks = np.stack(
        [pycocotools.mask.decode(mask) for mask in frame_data["masks"]],
        axis=0,
    )
    confidence = frame_data["conf"]
    if torch.is_tensor(confidence):
        confidence = confidence.detach().cpu().numpy()
    else:
        confidence = np.array(
            [
                value.detach().cpu().numpy() if torch.is_tensor(value) else value
                for value in confidence
            ]
        )
    return masks, confidence


def get_grounding_sam_frame_scores(groundedsam_data_dict, frame_id):
    """Return the per-mask ``score`` values for one frame in mask order."""
    frame_data = groundedsam_data_dict.get(str(frame_id))
    if frame_data is None:
        return None
    if "score" not in frame_data:
        raise KeyError(f"Frame {frame_id} does not contain a 'score' field")

    scores = frame_data["score"]
    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()
    else:
        scores = np.asarray(scores)
    return scores.reshape(-1)


def get_scannet_color_and_depth_intrinsic(color_path):
    intrinsic_path = color_path.replace("color", "intrinsic")
    color_intrinsic_path = join(dirname(intrinsic_path), "intrinsic_color.txt")
    depth_intrinsic_path = join(dirname(intrinsic_path), "intrinsic_depth.txt")

    color_intrinsic = np.loadtxt(color_intrinsic_path).astype(np.float32)[:3, :3]
    depth_intrinsic = np.loadtxt(depth_intrinsic_path).astype(np.float32)[:3, :3]

    return color_intrinsic, depth_intrinsic


def get_scannet_mask(sam2d_path, color_path, scene_id, mask_group_name):
    mask_path = get_scannet_mask_path(
        sam2d_path, color_path, scene_id, mask_group_name
    )
    mask_raw = cv2.imread(mask_path, -1)
    if mask_raw is None:
        raise FileNotFoundError(f"Unable to read 2D mask image: {mask_path}")
    return mask_raw.astype(np.float32)


def get_scannet_mask_path(sam2d_path, color_path, scene_id, mask_group_name):
    mask_dir = join(sam2d_path, scene_id, mask_group_name)
    color_name = basename(color_path)
    mask_name = "maskraw_" + color_name.replace("jpg", "png")
    return join(mask_dir, mask_name)


def get_scannet_color_image(image_pth):
    """
    apply transformation to the image. crop the image ot 640 short edge by default
    """
    image = Image.open(image_pth).convert("RGB")

    image_ori = np.asarray(image)
    images = torch.from_numpy(image_ori.copy()).permute(2, 0, 1)

    return image_ori, images


def get_scannet_semantic_mask(color_path, base_dir, scene_id, is_scannet200):
    if not is_scannet200:
        mask_dir = join(base_dir, "2D_masks", scene_id, "ovseg")
    else:
        mask_dir = join(base_dir, "2D_masks", scene_id, "ovseg200")
    color_name = basename(color_path)
    mask_name = "semantic_maskraw_" + color_name.replace("jpg", "png")
    mask_path = join(mask_dir, mask_name)

    mask_raw = cv2.imread(mask_path, -1).astype(np.float32)

    return mask_raw


def get_adapted_intrinsic(
    intrinsic, depth, intrinsic_original_resolution, desired_resolution
):
    """Get adjusted camera intrinsics."""
    if intrinsic_original_resolution == desired_resolution:
        return depth, intrinsic
    depth = cv2.resize(
        depth,
        (desired_resolution[1], desired_resolution[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    resize_width = int(
        math.floor(
            desired_resolution[1]
            * float(intrinsic_original_resolution[0])
            / float(intrinsic_original_resolution[1])
        )
    )

    adapted_intrinsic = intrinsic.copy()
    adapted_intrinsic[0, 0] *= float(resize_width) / float(
        intrinsic_original_resolution[0]
    )
    adapted_intrinsic[1, 1] *= float(desired_resolution[1]) / float(
        intrinsic_original_resolution[1]
    )
    adapted_intrinsic[0, 2] *= float(desired_resolution[0] - 1) / float(
        intrinsic_original_resolution[0] - 1
    )
    adapted_intrinsic[1, 2] *= float(desired_resolution[1] - 1) / float(
        intrinsic_original_resolution[1] - 1
    )

    return depth, adapted_intrinsic


# -----------------------------------------------------------------------------
# ScanNet++ data access
# -----------------------------------------------------------------------------


def get_scannetpp_poses_and_intrinsics(base_dir, scene_id, freq):
    pose_dir = join(base_dir, scene_id, "pose")
    intrinsics_dir = join(base_dir, scene_id, "intrinsic")
    frame_num = len([f for f in os.listdir(pose_dir) if f.endswith(".txt")])

    poses = []
    intrinsics = []
    for frame in range(0, frame_num, freq):
        key = "%05d" % frame
        pose = np.loadtxt(os.path.join(pose_dir, key + ".txt")).astype(np.float32)
        intrinsic = np.loadtxt(os.path.join(intrinsics_dir, key + ".txt")).astype(
            np.float32
        )
        poses.append(pose)
        intrinsics.append(intrinsic)

    return poses, intrinsics


def get_scannetpp_depth(color_path, color_shape):
    depth_path = color_path.replace("color", "depth").replace(".jpg", ".png")

    depth = cv2.imread(depth_path, -1).astype(np.float32)
    depth = cv2.resize(depth, [color_shape[1], color_shape[0]], interpolation=1)

    depth /= 1000.0

    return depth


def get_scannetpp_color_and_depth_intrinsic(color_path):
    intrinsic_path = color_path.replace("color", "intrinsic")
    color_intrinsic_path = intrinsic_path.replace("jpg", "txt")
    depth_intrinsic_path = intrinsic_path.replace("jpg", "txt")

    color_intrinsic = np.loadtxt(color_intrinsic_path).astype(np.float32)[:3, :3]
    depth_intrinsic = np.loadtxt(depth_intrinsic_path).astype(np.float32)[:3, :3]

    return color_intrinsic, depth_intrinsic


def get_scannetpp_mask(sam2d_path, color_path, scene_id, mask_group_name):
    mask_dir = join(sam2d_path, scene_id, mask_group_name)
    color_name = basename(color_path)
    mask_name = "maskraw_" + color_name.replace("jpg", "png")
    mask_path = join(mask_dir, mask_name)
    mask_raw = cv2.imread(mask_path, -1).astype(np.float32)

    mask_raw = mask_raw[:, :, 0]

    unique_values = np.unique(mask_raw)
    unique_values = unique_values[unique_values != 0]

    masks = np.zeros((len(unique_values), *mask_raw.shape), dtype=int)

    for idx, value in enumerate(unique_values):
        masks[idx] = (mask_raw == value).astype(int)

    ones_array = np.ones(masks.shape[0], dtype=int)
    return masks, ones_array


def scannetpp_get_semantic_mask(
    color_path, base_dir, scene_id, semantic_text, color_shape
):
    mask_dir = join(base_dir, "2D_masks", scene_id, semantic_text)
    color_name = basename(color_path)

    mask_name = "semantic_maskraw_" + color_name.replace("jpg", "png")
    mask_path = join(mask_dir, mask_name)

    mask_raw = cv2.imread(mask_path, -1).astype(np.float32)
    mask_raw = cv2.resize(
        mask_raw, [color_shape[1], color_shape[0]], interpolation=cv2.INTER_NEAREST
    )

    return mask_raw


# -----------------------------------------------------------------------------
# Post-processing
# -----------------------------------------------------------------------------


def filter_by_population(points, filter_population, save_path):
    """filter out small groups of points"""
    if points is None:
        points = np.loadtxt(save_path)

    if not isinstance(filter_population, list) and not isinstance(
        filter_population, tuple
    ):
        filter_population = [filter_population]

    labels = points[..., -1]
    labels_unique = np.unique(labels)

    for popu in filter_population:
        labels_filter = labels.copy()

        for label in labels_unique:
            idx = labels_filter == label
            if idx.sum() < popu:
                labels_filter[idx] = 0

        rearange_label = np.unique(labels_filter, return_inverse=True)[1]
        points_filter = np.concatenate((points[..., :-1], rearange_label[:, None]), -1)

        points_path = save_path.replace(".pts", f"_amount{popu:.0f}.pts")
        np.savetxt(points_path, points_filter)
        print(f"save to {points_path}")


def get_common_label(labels, max_label=None):
    """
    get the most common label from a group of labels (except for 0)
    assign a new label when there exist three or more major label in a seg
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    if unique_labels.shape[0] == 1:
        common_label = unique_labels[np.argsort(-counts)][0]
    else:
        fg_labels = unique_labels[unique_labels != 0]
        counts = counts[unique_labels != 0]
        prim_label_num = np.sum(counts > np.sum(counts) / counts.shape[0] * 3)
        if max_label is not None and prim_label_num >= 3:
            common_label = max_label + 1
        else:
            common_label = fg_labels[np.argsort(-counts)][0]

    return common_label


def get_seg_label(seg_ids, labels, points, voxel_size, max_label):
    new_labels = np.zeros(labels.shape[0])

    ids = np.unique(seg_ids)
    for id in ids:
        group = seg_ids == id
        group_points = points[group]
        group_labels = labels[group]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(group_points)
        pcd_ds, _, voxel_members = pcd.voxel_down_sample_and_trace(
            voxel_size, pcd.get_min_bound(), pcd.get_max_bound(), False
        )
        voxel_num = len(voxel_members)
        voxel_labels = np.zeros(voxel_num)

        for i in range(voxel_num):
            ids = voxel_members[i]
            voxel_labels[i] = get_common_label(group_labels[ids])

        seg_common_labels = get_common_label(voxel_labels, max_label)
        if seg_common_labels > max_label:
            max_label = seg_common_labels
        new_labels[group] = seg_common_labels
    return new_labels


# -----------------------------------------------------------------------------
# Evaluation and export
# -----------------------------------------------------------------------------


def export_ids(filename, ids):
    if not os.path.exists(dirname(filename)):
        os.mkdir(dirname(filename))
    with open(filename, "w") as f:
        for item_id in ids:
            f.write("%d\n" % item_id)


def export_res_to_class_agnostc_eval(base_dir, scene_id, pol_name, semantic_name=None):
    assert pol_name.endswith(".pts")
    pol_path = join(base_dir, "scans", scene_id, "results", pol_name)
    instance_ids = np.loadtxt(pol_path)[:, 4].astype(int)

    semantic_ids = np.ones_like(instance_ids, dtype=int)

    save_dir = join(base_dir, "results", pol_name[:-4])
    os.makedirs(save_dir, exist_ok=True)

    filename = join(save_dir, f"{scene_id}.txt")
    output_mask_path_relative = f"{scene_id}_pred_mask"
    name = os.path.splitext(os.path.basename(filename))[0]
    output_mask_path = os.path.join(
        os.path.dirname(filename), output_mask_path_relative
    )
    if not os.path.isdir(output_mask_path):
        os.mkdir(output_mask_path)
    insts = np.unique(instance_ids)
    zero_mask = np.zeros(shape=(instance_ids.shape[0]), dtype=np.int32)
    with open(filename, "w") as f:
        for idx, inst_id in tqdm(enumerate(insts)):
            if inst_id == 0:
                continue
            relative_output_mask_file = os.path.join(
                output_mask_path_relative, name + "_" + str(idx) + ".txt"
            )
            output_mask_file = os.path.join(
                output_mask_path, name + "_" + str(idx) + ".txt"
            )
            loc = np.where(instance_ids == inst_id)
            label_id = semantic_ids[loc[0][0]]
            f.write("%s %d %f\n" % (relative_output_mask_file, label_id, 1.0))

            mask = np.copy(zero_mask)
            mask[loc[0]] = 1
            export_ids(output_mask_file, mask)


def new_export_res_to_class_agnostc_eval(
    base_dir, scene_id, pol_name, semantic_name=None
):
    assert pol_name.endswith(".pts")
    pol_path = join(base_dir, "scans", scene_id, "results", pol_name)
    instance_ids = np.loadtxt(pol_path)[:, 4].astype(np.int32)

    print(np.unique(instance_ids), len(np.unique(instance_ids)))
    save_dir = join(base_dir, "results", "new_formation_" + pol_name[:-4])
    os.makedirs(save_dir, exist_ok=True)

    np.savetxt(
        join(save_dir, f"{scene_id}.txt"), instance_ids.astype(np.int32), fmt="%d"
    )
    print("save to ", join(save_dir, f"{scene_id}.txt"))


def export_small_gt_to_class_agnostic_eval(base_dir, scene_id):
    gt_path = join(base_dir, "scans", scene_id, "scans", "segments_anno.json")
    with open(gt_path, "r") as f:
        data = json.load(f)

    seg_path = join(base_dir, "scans", scene_id, "scans", "segments.json")
    with open(seg_path, "r") as f:
        data2 = json.load(f)
    seg_ids = np.array(data2["segIndices"])

    classes_path = join(base_dir, "metadata/semantic/instance_classes.txt")
    with open(classes_path, "r") as f:
        classes = f.readlines()
    instance_valid_label = [valid_class.strip() for valid_class in classes]

    eval_ids = np.zeros_like(seg_ids, dtype=int)
    vert_ins_size = np.ones_like(seg_ids, dtype=int) * 1000000
    seg_groups = data["segGroups"]
    for seg_group in seg_groups:
        segments = seg_group["segments"]
        semantic_label = seg_group["label"]
        if semantic_label not in instance_valid_label:
            continue
        inst_mask = np.isin(seg_ids, segments)
        inst_size = inst_mask.nonzero()[0].shape[0]
        inst_mask = inst_mask & (vert_ins_size > inst_size)

        eval_ids[inst_mask] = 1000 + seg_group["objectId"] + 1
        vert_ins_size[inst_mask] = inst_size

    os.makedirs(join(base_dir, "results", "small_class-agnostic_gt_ids"), exist_ok=True)
    np.savetxt(
        join(base_dir, "results", "small_class-agnostic_gt_ids", f"{scene_id}.txt"),
        eval_ids,
        fmt="%d",
    )






# -----------------------------------------------------------------------------
# 3D-guided visibility and mask matching
# -----------------------------------------------------------------------------


def calculate_visible_points_per_superpoint_numpy(
    color_pixes, points_seen, seg_num, seg_members, weight_points
):
    frame_count = points_seen.shape[1]

    visible_counts = np.zeros((seg_num, frame_count), dtype=float)

    visible_weight_counts = np.zeros((seg_num, frame_count), dtype=float)

    visible_coords = np.empty((seg_num, frame_count), dtype=object)
    visible_weight_coords = np.empty((seg_num, frame_count), dtype=object)
    for seg_idx in range(seg_num):
        for frame_idx in range(frame_count):
            visible_coords[seg_idx, frame_idx] = np.empty((0, 2), dtype=int)
            visible_weight_coords[seg_idx, frame_idx] = np.empty((0, 1), dtype=int)

    for seg_idx in range(seg_num):
        point_indices = np.array(seg_members[seg_idx]["point_indices"])

        visibility = points_seen[point_indices]
        weight = weight_points[point_indices]

        visible_counts[seg_idx] = visibility.sum(axis=0)
        visible_weight_counts[seg_idx] = weight.sum(axis=0)

        visible_points = color_pixes[point_indices]

        for frame_idx in range(frame_count):
            visible = visibility[:, frame_idx].astype(bool)
            coords = visible_points[visible, frame_idx]
            coords_weight = weight[visible, frame_idx]

            if coords.size > 0:
                visible_coords[seg_idx, frame_idx] = coords
                visible_weight_coords[seg_idx, frame_idx] = coords_weight

    return visible_counts, visible_coords, visible_weight_counts, visible_weight_coords


def calculate_top_k_visible_frames(
    visible_counts, visible_coords, seg_count, visible_weight_coords, iou=0.3
):
    """Select frames whose visible-point ratio exceeds ``iou`` per segment."""
    N, M = visible_counts.shape

    top_k_frames_list = []
    top_k_coords_list = []
    top_k_weight_coords_list = []

    for i in range(N):
        counts = visible_counts[i]

        ratio = counts / seg_count[i]
        top_k_indices = np.where(ratio > iou)[0]

        if len(top_k_indices) > 0:
            top_k_coords = [visible_coords[i][frame_idx] for frame_idx in top_k_indices]
            top_k_weight_coords = [
                visible_weight_coords[i][frame_idx] for frame_idx in top_k_indices
            ]
        else:
            top_k_coords = []
            top_k_weight_coords = []

        top_k_frames_list.append(top_k_indices.tolist())
        top_k_coords_list.append(top_k_coords)
        top_k_weight_coords_list.append(top_k_weight_coords)

    return top_k_frames_list, top_k_coords_list, top_k_weight_coords_list




# -----------------------------------------------------------------------------
# Legacy region-growing helpers sai3d
# -----------------------------------------------------------------------------


def judge_connect(
    adj,
    p1_id,
    p2_id,
    thres_connect,
    seg_indirect_neighbors,
    seg_member_count,
    seg_labels,
    region_label,
    group_points_count,
    max_neighbor_distance,
    decay=0.5,
):
    """Apply the legacy hierarchical criterion to a region candidate."""
    weight_sum = 0.0
    adj_sum = 0.0
    weight = 1

    seg_id = p2_id
    for i in range(max_neighbor_distance):
        neighbor_ids = seg_indirect_neighbors[i][seg_id]
        if i > 0:
            neighbor_ids = np.logical_and(
                neighbor_ids, np.logical_not(seg_indirect_neighbors[i - 1][seg_id])
            )
        neighbor_ids = np.logical_and(neighbor_ids, seg_labels == region_label)
        neighbor_ids = neighbor_ids.nonzero()[0]

        adj_sum += (
            weight * adj[seg_id, neighbor_ids] * seg_member_count[neighbor_ids]
        ).sum(0)
        weight_sum += weight * (seg_member_count[neighbor_ids]).sum(0)
        weight *= decay

    score = adj_sum / weight_sum
    return score >= thres_connect


def assign_seg_label(
    seg_num,
    seg_member_count,
    seg_indirect_neighbors,
    seg_direct_neighbors,
    dis_decay,
    adj,
    thres_connect,
    max_neighbor_distance,
    dense_neighbor=False,
):
    assign_id = 1
    seg_labels = np.zeros(seg_num, dtype=np.float32)
    for i in range(seg_num):
        if seg_labels[i] <= 0:
            queue = []
            queue.append(i)
            seg_labels[i] = assign_id
            group_points_count = seg_member_count[i]
            seg_parents = np.full([seg_num], -1, dtype=int)

            while queue:
                v = queue.pop(0)
                if dense_neighbor:
                    js = seg_direct_neighbors[v]
                else:
                    js = seg_direct_neighbors[v].nonzero()[0]

                seg_parents[js] = v
                for j in js:
                    if seg_labels[j] != 0:
                        continue

                    connect = judge_connect(
                        adj,
                        v,
                        j,
                        thres_connect,
                        seg_indirect_neighbors,
                        seg_member_count,
                        seg_labels,
                        assign_id,
                        group_points_count,
                        max_neighbor_distance,
                        decay=dis_decay,
                    )

                    if not connect:
                        continue
                    seg_labels[j] = assign_id
                    group_points_count += seg_member_count[j]
                    queue.append(j)
            assign_id += 1

    return seg_labels


# -----------------------------------------------------------------------------
# Mask filtering
# -----------------------------------------------------------------------------


def old_filter_masks(masks, mask_weight, overlap_threshold=0.5):
    masks = torch.tensor(masks, dtype=torch.float32, device="cuda")
    mask_weight = torch.tensor(mask_weight, dtype=torch.float32, device="cuda")

    n = masks.shape[0]
    keep_mask = torch.ones(n, dtype=torch.bool, device="cuda")

    flat_masks = masks.reshape(n, -1)

    intersection = torch.mm(flat_masks, flat_masks.t())
    area = flat_masks.sum(dim=1)
    union = area.unsqueeze(1) + area.unsqueeze(0) - intersection

    overlap_ratio = intersection / union

    high_overlap = (overlap_ratio > overlap_threshold).triu(diagonal=1)

    high_overlap_indices = high_overlap.nonzero(as_tuple=False)

    for i, j in high_overlap_indices:
        if mask_weight[i] < mask_weight[j]:
            keep_mask[i] = False
        else:
            keep_mask[j] = False
    return keep_mask.cpu().numpy()


def _mask_geometry_identity(masks):
    """Return a scene-local identity key without copying mask pixels."""
    if torch.is_tensor(masks):
        return (
            "torch",
            id(masks),
            int(masks.data_ptr()),
            tuple(masks.shape),
            tuple(masks.stride()),
            str(masks.dtype),
            str(masks.device),
            int(masks._version),
        )
    array = np.asarray(masks)
    return (
        "numpy",
        id(masks),
        int(array.__array_interface__["data"][0]),
        tuple(array.shape),
        tuple(array.strides),
        str(array.dtype),
    )


def filter_masks(
    masks,
    mask_weight,
    overlap_threshold=0.5,
    overlap_cache=None,
    return_overlap_cache=False,
):
    """Filter masks, optionally reusing geometry-only overlap pairs."""
    geometry_identity = _mask_geometry_identity(masks)
    n = masks.shape[0]

    if overlap_cache is None:
        gpu_masks = torch.tensor(masks, dtype=torch.float32, device="cuda")
        flat_masks = gpu_masks.reshape(n, -1)
        intersection = torch.mm(flat_masks, flat_masks.t())
        area = flat_masks.sum(dim=1)
        union = area.unsqueeze(1) + area.unsqueeze(0) - intersection
        overlap_ratio = intersection / union
        high_overlap = (overlap_ratio > overlap_threshold).triu(diagonal=1)
        high_overlap_indices = high_overlap.nonzero(as_tuple=False)
        overlap_cache = {
            "geometry_identity": geometry_identity,
            "overlap_threshold": overlap_threshold,
            "mask_count": n,
            "high_overlap_indices": high_overlap_indices,
        }
    else:
        if (
            overlap_cache["geometry_identity"] != geometry_identity
            or overlap_cache["overlap_threshold"] != overlap_threshold
            or overlap_cache["mask_count"] != n
        ):
            raise ValueError("overlap_cache does not match the input mask geometry")
        high_overlap_indices = overlap_cache["high_overlap_indices"]

    gpu_mask_weight = torch.tensor(mask_weight, dtype=torch.float32, device="cuda")
    keep_mask = torch.ones(n, dtype=torch.bool, device="cuda")
    if high_overlap_indices.shape[0] > 0:
        first = high_overlap_indices[:, 0]
        second = high_overlap_indices[:, 1]
        losers = torch.where(
            gpu_mask_weight[first] < gpu_mask_weight[second],
            first,
            second,
        )
        keep_mask[losers] = False

    keep = keep_mask.cpu().numpy()
    if return_overlap_cache:
        return keep, overlap_cache
    return keep


def compute_mask_weights(
    top_k_frames_list,
    top_k_coords_list,
    top_k_weight_coords_list,
    mask_list,
    seg_counts,
    seg_member_count_union,
    points_seen,
    cw0,
    ch0,
    weight_points,
    threshold,
):
    N = len(top_k_frames_list)
    M = len(mask_list)
    mask_w1 = [np.zeros(len(masks), dtype=np.float32) for masks in mask_list]
    mask_w2 = [np.zeros(len(masks), dtype=np.float32) for masks in mask_list]

    P = points_seen.shape[0]
    num_seg = len(seg_member_count_union)
    point2seg = np.full(P, -1, dtype=np.int32)
    for seg_id, d in seg_member_count_union.items():
        if "point_indices" in d and len(d["point_indices"]) > 0:
            point2seg[np.asarray(d["point_indices"], dtype=np.int64)] = int(seg_id)

    points_seen = points_seen.astype(bool)
    cw0 = cw0.astype(np.int64)
    ch0 = ch0.astype(np.int64)

    super_point_mask_info = {}

    for super_idx in tqdm(range(N)):
        frames = top_k_frames_list[super_idx]
        coords_list = top_k_coords_list[super_idx]

        super_point_mask_info[super_idx] = []
        for idx, frame_idx in enumerate(frames):
            coords = coords_list[idx]
            if coords.size == 0:
                continue

            mask_data = mask_list[frame_idx]
            num_masks, H, W = mask_data.shape
            x_coords, y_coords = coords[:, 0], coords[:, 1]

            num_visible_points = coords.shape[0]

            intersection = (
                mask_data[:, y_coords, x_coords].sum(axis=1).astype(np.float32)
            )

            overlap_ratio = intersection / max(1, num_visible_points)
            valid_mask_idx = np.where(overlap_ratio > threshold)[0]
            if valid_mask_idx.size > 0:
                super_point_mask_info[super_idx].append((frame_idx, valid_mask_idx))

                mask_w1[frame_idx][valid_mask_idx] += intersection[valid_mask_idx]
                mask_w2[frame_idx][valid_mask_idx] += num_visible_points

    pairs_by_frame = defaultdict(list)
    for sp, items in super_point_mask_info.items():
        for f, mask_ids in items:
            for mid in np.atleast_1d(mask_ids):
                pairs_by_frame[f].append(int(mid))

    for f in list(pairs_by_frame.keys()):
        if len(pairs_by_frame[f]) == 0:
            pairs_by_frame.pop(f)
        else:
            pairs_by_frame[f] = sorted(set(pairs_by_frame[f]))

    dist_cache = {}

    for f, mask_ids in pairs_by_frame.items():
        mask_data = mask_list[f].astype(bool)
        Kf, H, W = mask_data.shape

        vis = points_seen[:, f]
        p_idx = np.flatnonzero(vis & (point2seg >= 0))
        if p_idx.size == 0:
            for mid in mask_ids:
                dist_cache[(f, mid)] = np.zeros((num_seg,), dtype=np.float32)
            continue

        x = cw0[p_idx, f]
        y = ch0[p_idx, f]

        inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if not np.any(inb):
            for mid in mask_ids:
                dist_cache[(f, mid)] = np.zeros((num_seg,), dtype=np.float32)
            continue

        p_idx = p_idx[inb]
        weights = weight_points[p_idx, f]
        x = x[inb]
        y = y[inb]
        flat = (y * W + x).astype(np.int64)

        seg_ids = point2seg[p_idx].astype(np.int64)

        order = np.argsort(seg_ids, kind="mergesort")
        seg_sorted = seg_ids[order]
        uniq_seg, start_idx = np.unique(seg_sorted, return_index=True)

        boundaries = np.concatenate([start_idx, [seg_sorted.size]])
        group_sizes = np.diff(boundaries).astype(np.float32)

        sel = np.asarray(mask_ids, dtype=np.int64)
        mflat = mask_data[sel].reshape(sel.size, -1)

        inside_bool = mflat[:, flat]

        inside_bool_sorted = inside_bool[:, order]
        weights_sorted = weights[order]

        counts_inside = np.add.reduceat(
            inside_bool_sorted.astype(np.int32), boundaries[:-1], axis=1
        ).astype(np.float32)

        weighted_inside_sorted = (
            inside_bool_sorted.astype(np.float32) * weights_sorted[None, :]
        )
        sum_w_inside = np.add.reduceat(
            weighted_inside_sorted, boundaries[:-1], axis=1
        ).astype(np.float32)

        mean_w_inside = np.divide(
            sum_w_inside,
            np.maximum(counts_inside, 1.0),
            out=np.zeros_like(sum_w_inside, dtype=np.float32),
            where=counts_inside > 0,
        )

        ratio_inside = counts_inside / group_sizes[None, :]

        dist_local = (ratio_inside * mean_w_inside).astype(np.float32)

        dist_batch = np.zeros((sel.size, num_seg), dtype=np.float32)
        dist_batch[:, uniq_seg] = dist_local

        for i, mid in enumerate(sel):
            dist_cache[(f, int(mid))] = dist_batch[i]

    A_dict = {}
    cos_sim_avg_dict = {}
    pairs_dict = {}

    for sp, items in super_point_mask_info.items():
        pairs = []
        for f, mids in items:
            for mid in np.atleast_1d(mids):
                pairs.append((f, int(mid)))

        if len(pairs) == 0:
            A = np.zeros((0, num_seg), dtype=np.float32)
            A_dict[sp] = A
            cos_sim_avg_dict[sp] = np.zeros((0,), dtype=np.float32)
            pairs_dict[sp] = []
            continue

        A = np.vstack([dist_cache[(f, m)] for (f, m) in pairs]).astype(np.float32)

        if A.shape[0] == 1:
            cos_avg = np.array([1.0], dtype=np.float32)
        else:
            norms = np.linalg.norm(A, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            A_norm = A / norms
            sim = A_norm @ A_norm.T

            np.fill_diagonal(sim, 0.0)
            cos_avg = sim.sum(axis=1) / (A.shape[0] - 1)
            cos_avg = cos_avg.astype(np.float32)

        A_dict[sp] = A
        cos_sim_avg_dict[sp] = cos_avg
        pairs_dict[sp] = pairs

    mask_cos_collect = defaultdict(list)
    for sp in cos_sim_avg_dict:
        items = super_point_mask_info[sp]
        pos = 0
        for f, mids in items:
            mids = np.atleast_1d(mids)
            for mid in mids:
                mask_cos_collect[(f, int(mid))].append(cos_sim_avg_dict[sp][pos])
                pos += 1

    mask_cos_avg = [np.zeros(len(mask_list[f]), dtype=np.float32) for f in range(M)]
    for (f, m), vals in mask_cos_collect.items():
        mask_cos_avg[f][m] = np.mean(vals)

    mask_weights = mask_cos_avg

    return mask_weights


def update_masks(mask_list, keep_idx2, mask_weight):
    """Build frame labels without allocating a K x H x W id layer."""
    frame_count = len(mask_list)
    height, width = mask_list[0].shape[1:]
    masks_2D = np.zeros((frame_count, height, width), dtype=np.uint8)

    for frame_index in range(frame_count):
        masks = mask_list[frame_index]
        keep = keep_idx2[frame_index]
        weights = mask_weight[frame_index]
        indices = np.where(keep)[0]
        if len(indices) == 0:
            continue

        selected_weights = weights[indices]
        order = np.lexsort((indices, -selected_weights))
        ordered_indices = indices[order]
        class_ids = np.arange(len(indices), 0, -1)
        layer_dtype = np.result_type(masks.dtype, class_ids.dtype)
        frame_max = np.zeros((height, width), dtype=layer_dtype)
        candidate = np.empty((height, width), dtype=layer_dtype)

        for mask_index, class_id in zip(ordered_indices, class_ids):
            np.multiply(
                masks[mask_index],
                class_id,
                out=candidate,
                casting="unsafe",
            )
            np.maximum(frame_max, candidate, out=frame_max)

        masks_2D[frame_index] = frame_max
    return masks_2D


def build_mask_map_from_scores(masks, scores, overlap_threshold=0.5):
    """Build one 2D label map using the score-based mask ordering."""
    scores = np.asarray(scores).reshape(-1)
    if masks.shape[0] != scores.shape[0]:
        raise ValueError(
            f"Mask/score count mismatch: {masks.shape[0]} masks and "
            f"{scores.shape[0]} scores"
        )

    keep = filter_masks(masks, scores, overlap_threshold=overlap_threshold)
    mask_map = update_masks([masks], [keep], [scores])[0]
    return mask_map

"""
The code has been refactored by CodeX.
"""

import argparse
import glob
import json
import os
from os.path import basename, dirname, join

import cv2
import numpy as np
import scipy
import torch
from natsort import natsorted
from tqdm import tqdm

import helpers.mv3dis_utils as utils
from linetimer import CodeTimer
from mv3dis_base import MV3DISBase


DEFAULT_SCANNET_ROOT = "/scannet"

DEFAULT_POINTS_PATH = (
    "/points_path"
)

DEFAULT_SAM2D_PATH = "grounding_sam2"

DEFAULT_DATA2D_PATH = "scannet/RGB_D/validation"

DEFAULT_EVAL_DIR = ""

_ORIGINAL_TENSOR_REPR = torch.Tensor.__repr__


def _tensor_repr_with_shape(tensor):
    """Include tensor shape in debug output without wrapping repr twice."""
    return f"{{Tensor:{tuple(tensor.shape)}}} {_ORIGINAL_TENSOR_REPR(tensor)}"


torch.Tensor.__repr__ = _tensor_repr_with_shape


# -----------------------------------------------------------------------------
# Dataset adapter
# -----------------------------------------------------------------------------


def load_ids(filename):
    with open(filename) as file:
        ids = file.read().splitlines()
    ids = np.array(ids, dtype=np.int64)
    return ids


class ScanNetMV3DIS(MV3DISBase):
    """ScanNet/ScanNet++ data adapter for the MV3DIS core."""

    def __init__(self, points, args):
        self.scannetpp = args.scannetpp
        super().__init__(points, args)

    def init_data(
        self,
        scene_id,
        data2d_path,
        mask_name,
        need_semantic=False,
        inference_fast=False,
    ):
        """Load 2D observations and initialize scene image metadata.

        ``inference_fast`` skips fields that the production inference path
        never reads while preserving every array consumed by ``assign_label``.
        """
        (
            self.poses,
            self.color_intrinsics,
            self.depth_intrinsics,
            self.depths,
            self.images,
            self.ori_images,
            self.mask_list,
            self.mask_epoch,
            self.pred_score,
            self.masks_2D,
            self.points_label,
        ) = self.get_mask_data(
            data2d_path,
            scene_id,
            mask_name,
            need_semantic,
            self.scannetpp,
            inference_fast=inference_fast,
        )
        self.M = self.poses.shape[0]
        if self.images is None:
            if self.masks_2D is not None:
                self.CH, self.CW = self.masks_2D.shape[-2:]
            else:
                self.CH, self.CW = self.mask_list[0].shape[-2:]
        else:
            self.CH, self.CW = self.images.shape[-2:]
        self.DH, self.DW = self.depths.shape[-2:]

        self.img_dim = [self.CW, self.CH]
        self.data2d_path = data2d_path
        self.scene_id = scene_id

    def old_get_mask_data(
        self, data2d_path, scene_id, mask_name, need_semantic=False, scannetpp=False
    ):
        if scannetpp:
            rgb_dir = join(data2d_path, scene_id)
            color_list = natsorted(glob.glob(join(rgb_dir, "iphone", "color", "*.jpg")))
            color_shape = np.array(cv2.imread(color_list[0])).shape

            poses = []
            color_intrinsics = []
            depth_intrinsics = []
            depths = []
            ori_color_image = []
            color_image = []
            mask_list = []
            mask_epoch = []
            pred_score = []

            sam_2d_path = ""
            mask_dir = join(sam_2d_path, scene_id + ".pth")
            groundedsam_data_dict = torch.load(mask_dir)

            for i, color_path in enumerate(tqdm(color_list, desc="Read 2D data")):
                color_name = basename(color_path)

                num = int(color_name[-9:-4])
                if i % self.view_freq != 0:
                    continue

                single_mask_list, single_maskscore_list = (
                    utils.get_mask_list_grounding_sam(
                        groundedsam_data_dict, fram_id=f"{num:05d}"
                    )
                )
                if single_mask_list is None:
                    continue
                dep = utils.get_scannetpp_depth(color_path, color_shape=color_shape)
                if dep is None:
                    continue
                depths.append(dep)

                poses.append(utils.get_scannet_pose(color_path))
                color_intrinsic, depth_intrinsic = (
                    utils.get_scannetpp_color_and_depth_intrinsic(color_path)
                )
                color_intrinsics.append(color_intrinsic)
                depth_intrinsics.append(depth_intrinsic)
                ori, im = utils.get_scannet_color_image(color_path)
                ori_color_image.append(ori)
                color_image.append(im)
                mask_list.append(single_mask_list)
                mask_epoch.append(np.zeros(single_mask_list.shape[0]))
                pred_score.append(single_maskscore_list.reshape(-1))
            poses = np.stack(poses, 0)
            color_intrinsics = np.stack(color_intrinsics, 0)
            depth_intrinsics = np.stack(depth_intrinsics, 0)
            depths = np.stack(depths, 0)
            color_image = np.stack(color_image, 0)
            ori_color_image = np.stack(ori_color_image, 0)

            return (
                poses,
                color_intrinsics,
                depth_intrinsics,
                depths,
                color_image,
                ori_color_image,
                mask_list,
                mask_epoch,
                pred_score,
            )

        else:
            data_dir = join(data2d_path, scene_id)
            color_list = natsorted(glob.glob(join(data_dir, "color", "*.jpg")))

            poses = []
            color_intrinsics = []
            depth_intrinsics = []
            depths = []
            ori_color_image = []
            color_image = []
            mask_list = []
            mask_epoch = []
            pred_score = []
            masks_2D = []
            mask_dir = join(
                self.args.sam2d_path, "2d_pth_score", scene_id + ".pth"
            )
            groundedsam_data_dict = torch.load(mask_dir)

            for color_path in tqdm(color_list, desc="Read 2D data"):
                color_name = basename(color_path)
                num = int(color_name[-9:-4])
                if num % self.view_freq != 0:
                    continue
                single_mask_list, single_maskscore_list = (
                    utils.get_mask_list_grounding_sam(
                        groundedsam_data_dict, fram_id=num
                    )
                )
                if single_mask_list is None:
                    continue
                mas = self._load_or_build_score_mask(
                    groundedsam_data_dict,
                    frame_id=num,
                    color_path=color_path,
                    scene_id=scene_id,
                    masks=single_mask_list,
                )
                masks_2D.append(mas)
                poses.append(utils.get_scannet_pose(color_path))
                depths.append(utils.get_scannet_depth(color_path))
                color_intrinsic, depth_intrinsic = (
                    utils.get_scannet_color_and_depth_intrinsic(color_path)
                )
                color_intrinsics.append(color_intrinsic)
                depth_intrinsics.append(depth_intrinsic)
                ori, im = utils.get_scannet_color_image(color_path)
                ori_color_image.append(ori)
                color_image.append(im)
                mask_list.append(single_mask_list)
                mask_epoch.append(np.zeros(single_mask_list.shape[0]))
                pred_score.append(single_maskscore_list.reshape(-1))

            poses = np.stack(poses, 0)
            masks_2D = np.stack(masks_2D, 0)
            color_intrinsics = np.stack(color_intrinsics, 0)
            depth_intrinsics = np.stack(depth_intrinsics, 0)
            depths = np.stack(depths, 0)
            color_image = np.stack(color_image, 0)
            ori_color_image = np.stack(ori_color_image, 0)

            # Preserve the legacy loader's GT-file validation side effect.
            _ = load_ids(
                os.path.join(
                    "",
                    args.scene_id + ".txt",
                )
            )

            loader = torch.load(
                os.path.join(
                    "",
                    args.scene_id + ".pth",
                )
            )
            sem_gt, inst_gt = loader[2], loader[3]
            gts_sem = np.array(sem_gt).astype(np.int32)
            gts_ins = np.array(inst_gt).astype(np.int32)
            gts_sem = gts_sem - 2 + 1
            gts_sem[gts_sem < 0] = 0
            gts_ins = gts_ins + 1
            ignore_inds = gts_ins < 0

            # scannet encoding rule
            gts = gts_sem * 1000 + gts_ins
            gts[ignore_inds] = 0

            return (
                poses,
                color_intrinsics,
                depth_intrinsics,
                depths,
                color_image,
                ori_color_image,
                mask_list,
                mask_epoch,
                pred_score,
                masks_2D,
                gts,
            )

    def _resolve_mask_data_path(self, scene_id):
        """Resolve the required score-based Grounded-SAM scene file."""
        mask_path = join(
            self.args.sam2d_path,
            "2d_pth_score",
            scene_id + ".pth",
        )
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(
                f"No score-based scene mask file found at {mask_path}"
            )
        return mask_path

    def _load_or_build_score_mask(
        self,
        groundedsam_data_dict,
        frame_id,
        color_path,
        scene_id,
        masks,
    ):
        """Read a score mask PNG or recreate it from the PTH ``score`` field."""
        mask_path = utils.get_scannet_mask_path(
            self.args.sam2d_path,
            color_path,
            "2d_mask_score",
            scene_id,
        )
        if os.path.isfile(mask_path):
            mask_map = cv2.imread(mask_path, -1)
            if mask_map is not None:
                return mask_map.astype(np.float32)

        scores = utils.get_grounding_sam_frame_scores(
            groundedsam_data_dict, frame_id
        )
        mask_map = utils.build_mask_map_from_scores(masks, scores)
        return mask_map.astype(np.float32)

    def get_mask_data(
        self,
        data2d_path,
        scene_id,
        mask_name,
        need_semantic=False,
        scannetpp=False,
        inference_fast=False,
    ):
        """Load scene inputs, optionally skipping data unused by inference."""
        if not inference_fast or scannetpp:
            return self.old_get_mask_data(
                data2d_path,
                scene_id,
                mask_name,
                need_semantic,
                scannetpp,
            )

        data_dir = join(data2d_path, scene_id)
        color_list = natsorted(glob.glob(join(data_dir, "color", "*.jpg")))
        mask_dir = self._resolve_mask_data_path(scene_id)
        groundedsam_data_dict = torch.load(mask_dir)

        selected_paths = []
        mask_list = []
        pred_score = []
        masks_2D = []
        poses = []
        depths = []

        for color_path in tqdm(color_list, desc="Read 2D data"):
            color_name = basename(color_path)
            frame_id = int(color_name[-9:-4])
            if frame_id % self.view_freq != 0:
                continue
            single_masks, single_scores = utils.get_mask_list_grounding_sam_fast(
                groundedsam_data_dict, frame_id
            )
            if single_masks is None:
                continue

            selected_paths.append(color_path)
            mask_list.append(single_masks)
            pred_score.append(single_scores.reshape(-1))
            mask_map = self._load_or_build_score_mask(
                groundedsam_data_dict,
                frame_id=frame_id,
                color_path=color_path,
                scene_id=scene_id,
                masks=single_masks,
            )
            masks_2D.append(mask_map)
            poses.append(utils.get_scannet_pose(color_path))
            depths.append(utils.get_scannet_depth(color_path))

        color_intrinsic, depth_intrinsic = utils.get_scannet_color_and_depth_intrinsic(
            selected_paths[0]
        )
        frame_count = len(selected_paths)
        color_intrinsics = np.stack([color_intrinsic] * frame_count, axis=0)
        depth_intrinsics = np.stack([depth_intrinsic] * frame_count, axis=0)

        poses = np.stack(poses, axis=0)
        depths = np.stack(depths, axis=0)
        masks_2D = np.stack(masks_2D, axis=0)
        self.selected_frame_ids = np.array(
            [int(basename(path)[-9:-4]) for path in selected_paths],
            dtype=np.int64,
        )

        # Keep the historical tuple shape; unused inference fields remain None.

        return (
            poses,
            color_intrinsics,
            depth_intrinsics,
            depths,
            None,
            None,
            mask_list,
            None,
            pred_score,
            masks_2D,
            None,
        )

    def get_seg_data(
        self,
        scans_dir,
        scene_id,
        max_neighbor_distance,
        seg_ids=None,
        points_obj_labels_path=None,
        k_graph=8,
        point_level=False,
        points_clipfeature=None,
        ssp_split_sum=None,
        members_only=False,
    ):
        """Build primitive membership and neighborhood data for region growing.

        :param scene_id: id of scene
        :param max_neighbor_distance: max distance for searching seg neighbors.
        :param seg_ids(N,): ids of superpoints which each point belongs to.
        :param points_obj_labels_path: path to the file that contains points and their superpoint ids/
        :param k_graph: parameter for kdtree search
        :param point_level: whether to use points as primitives
        :return: seg_ids(N,): ids of superpoints which each point belongs to.
                 seg_num: number of superpoints
                seg_members: dict, key is superpoint id, value is the ids of points that belong to this superpoint
                seg_neineighbors: (max_neighbor_distance, seg_num, seg_num), binary matrix, "True" indicating the logical distance between two superpoints is leq max_neighbor_distance
        """

        if point_level:
            seg_ids = np.arange(self.N, dtype=int)
            seg_num = self.N
            seg_members = seg_ids
            points_kdtree = scipy.spatial.KDTree(self.points)
            points_neighbors = points_kdtree.query(self.points, k_graph, workers=20)[1]
            self.seg_member_count = np.ones(self.N, dtype=int)

            return seg_ids, seg_num, seg_members, points_neighbors

        if seg_ids is None:
            if points_obj_labels_path is None:
                if not self.scannetpp:
                    scene_seg_path = join(
                        args.scans_dir,
                        "scans",
                        scene_id,
                        f"{scene_id}_vh_clean_2.0.010000.segs.json",
                    )
                    with open(scene_seg_path, "r") as f:
                        seg_data = json.load(f)
                    seg_ids = np.array(seg_data["segIndices"])
                else:
                    scene_seg_path = join(
                        args.scans_dir,
                        scene_id,
                        "scans",
                        "mesh_aligned_0.05.0.010000.segs.json",
                    )
                    with open(scene_seg_path, "r") as f:
                        seg_data = json.load(f)
                    seg_ids = np.array(seg_data["segIndices"])

            else:
                seg_path = join(
                    scans_dir, "scans", scene_id, "results", points_obj_labels_path
                )
                seg_ids = np.loadtxt(seg_path)[:, 4].astype(int)

        seg_ids, mapping, unique_values = utils.num_to_natural(seg_ids)
        unique_seg_ids, counts = np.unique(seg_ids, return_counts=True)
        seg_num = unique_seg_ids.shape[0]

        if ssp_split_sum is not None:
            new_seg_split_sum = np.zeros(len(unique_values))

            valid_old_ids = np.arange(len(ssp_split_sum))
            valid_new_ids = mapping[valid_old_ids + 1]
            mask = valid_new_ids != -1
            np.add.at(new_seg_split_sum, valid_new_ids[mask], ssp_split_sum[mask])

        seg_members = {}
        for id in unique_seg_ids:
            seg_members[id] = np.where(seg_ids == id)[0]

        if ssp_split_sum is not None:
            for seg_id, point_indices in seg_members.items():
                seg_members[seg_id] = {
                    "point_indices": point_indices,
                    "seg_num": new_seg_split_sum[seg_id],
                }
        else:
            for seg_id, point_indices in seg_members.items():
                seg_members[seg_id] = {"point_indices": point_indices}
        if members_only:
            return seg_ids, seg_num, seg_members, None, counts, None

        knn_cache = getattr(self, "_point_knn_cache", None)
        if knn_cache is None or knn_cache[0] != k_graph:
            points_kdtree = scipy.spatial.KDTree(self.points)
            points_neighbors = points_kdtree.query(self.points, k_graph, workers=20)[1]
            self._point_knn_cache = (k_graph, points_neighbors)
        else:
            points_neighbors = knn_cache[1]

        seg_direct_neighbors = np.zeros((seg_num, seg_num), dtype=bool)
        for id, members in seg_members.items():
            neighbors = points_neighbors[members["point_indices"]]
            neighbor_seg_ids = seg_ids[neighbors]
            seg_direct_neighbors[id][neighbor_seg_ids] = 1
        seg_direct_neighbors[np.eye(seg_num, dtype=bool)] = 0  # exclude self
        # make neighboring matrix symmetric
        seg_direct_neighbors[seg_direct_neighbors.T] = 1

        centroids = np.zeros((seg_num, self.points.shape[1]))
        for i, members_i in seg_members.items():
            points_i = self.points[members_i["point_indices"]]
            centroids[i] = np.mean(points_i, axis=0)

        diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        distance_matrix = np.linalg.norm(diff, axis=2)

        seg_neineighbors = np.zeros(
            (max_neighbor_distance, seg_num, seg_num), dtype=bool
        )
        seg_neineighbors[0] = seg_direct_neighbors
        for i in range(1, max_neighbor_distance):
            for seg_id in range(seg_num):
                last_layer_neighbors = seg_neineighbors[i - 1, seg_id]
                this_layer_neighbors = (
                    seg_neineighbors[i - 1, last_layer_neighbors].sum(0) > 0
                )
                seg_neineighbors[i, seg_id] = this_layer_neighbors
            # exclude self
            seg_neineighbors[i, np.eye(seg_num, dtype=bool)] = 0
            # include closer neighbors
            seg_neineighbors[i, seg_neineighbors[i - 1]] = 1

        return seg_ids, seg_num, seg_members, seg_neineighbors, counts, distance_matrix

    def vis_seg_and_neighbor(
        self, query_points, scene_id, save_path, max_neighbor_distance=0
    ):
        """
        visualize the segmentation which the query points belong to and its neighboring segmentations
        """
        kdtree = scipy.spatial.KDTree(self.points)
        point_ids = kdtree.query(query_points, 1, workers=20)[1]
        seg_ids = np.unique(self.seg_ids[point_ids])
        labels = np.zeros(self.points.shape[0])
        assign_id = 1
        print("seg_num: ", self.seg_num)
        for seg_id in seg_ids:
            neighbor_seg_ids = self.seg_indirect_neighbors[max_neighbor_distance][
                seg_id
            ].nonzero()
            neighbor_seg_ids = np.append(neighbor_seg_ids, seg_id)
            print(neighbor_seg_ids)
            for i in tqdm(neighbor_seg_ids):
                labels[self.seg_members[i]] = assign_id
            assign_id += 1

        points_obj_label = np.concatenate(
            [self.points, np.ones([self.N, 1]), labels[:, None]], axis=-1
        )
        print("save to: ", save_path)
        np.savetxt(save_path, points_obj_label)


# -----------------------------------------------------------------------------
# Inference and result export
# -----------------------------------------------------------------------------


def run_mv3dis(args, scene_id):
    time_collection = {}
    with CodeTimer("Load points", dict_collect=time_collection):
        if args.scannetpp:
            ply_path = join(
                args.base_dir, "data", args.scene_id, "scans", "mesh_aligned_0.05.ply"
            )
        else:
            ply_path = join(
                args.base_dir, "scans", args.scene_id, f"{args.scene_id}_vh_clean_2.ply"
            )
        points_path = join(
            args.points_path, args.scene_id, f"{args.scene_id}_vh_clean_2.ply"
        )
        points_path = join(dirname(points_path), "points.pts")

        if not os.path.exists(points_path):
            os.makedirs(dirname(points_path), exist_ok=True)
            print("getting points from ply...")
            utils.get_points_from_ply(ply_path, points_path)

        points = np.loadtxt(points_path).astype(np.float32)
        print("points num:", points.shape[0])
        points_clipfeature = None
        if args.points_clipfeature_path is not None:
            points_clipfeature = utils.get_points_clipfeature(
                args.points_clipfeature_path, args.scene_id
            )

    save_dir = join(args.points_path, args.scene_id, "results")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    agent = ScanNetMV3DIS(points, args)

    with CodeTimer("Load images", dict_collect=time_collection):
        agent.init_data(
            args.scene_id,
            args.data2d_path,
            args.mask_name,
            inference_fast=True,
        )

    with CodeTimer("Assign instance labels", dict_collect=time_collection):
        labels_fine_global = agent.assign_label(
            points,
            thres_connect=args.thres_connect,
            vis_dis=args.thres_dis,
            max_neighbor_distance=args.max_neighbor_distance,
            similar_metric=args.similar_metric,
            points_clipfeature=points_clipfeature,
            scene_id=scene_id,
        )

    with CodeTimer("Save results", dict_collect=time_collection):
        process_points_labels(
            labels_fine_global, os.path.join(args.eval_dir, "final"), args.scene_id
        )

    print("fine labels num:", np.unique(labels_fine_global).shape[0])

    for k, v in time_collection.items():
        print(f"Time {k}: {v:.1f}")
    print(f"Total time: {sum(time_collection.values()):.1f}")


def export_ids(filename, ids):
    if not os.path.exists(dirname(filename)):
        os.mkdir(dirname(filename))
    with open(filename, "w") as f:
        for item_id in ids:
            f.write("%d\n" % item_id)


def rle_encode(mask):
    """Encode a one-dimensional binary mask as RLE."""
    length = mask.shape[0]
    mask = np.concatenate([[0], mask, [0]])
    runs = np.where(mask[1:] != mask[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    counts = " ".join(str(x) for x in runs)
    rle = dict(length=length, counts=counts)
    return rle


def process_points_labels(points_labels, output_dir, scene_name):
    """Encode point-instance labels and save them as a scene ``.pth`` file."""

    unique_labels = np.unique(points_labels)

    rle_data = []

    for label in unique_labels:
        mask = (points_labels == label).astype(int)

        rle = rle_encode(mask)

        rle_data.append(rle)

    os.makedirs(output_dir, exist_ok=True)

    output_pth_file = os.path.join(output_dir, f"{scene_name}.pth")
    torch.save({"ins": rle_data}, output_pth_file)
    print(f"Saved reconstructed .pth file: {output_pth_file}")


def export_merged_ids_for_eval(
    instance_ids, save_dir, args, res_name="None", label_ids_dir=None
):
    """
    code credit: scannet
    Export 3d instance labels for scannet class agnostic instance evaluation
    For semantic instance evaluation if label_ids_dir is not None
    """
    os.makedirs(save_dir, exist_ok=True)

    confidences = np.ones_like(instance_ids)

    if label_ids_dir is None:
        label_ids = np.ones_like(instance_ids, dtype=int)
    else:
        label_ids = np.loadtxt(join(label_ids_dir, f"{args.scene_id}.txt")).astype(int)

    filename = join(save_dir, f"{args.scene_id}.txt")
    print(f"export {res_name} to {filename}")

    output_mask_path_relative = f"{args.scene_id}_pred_mask"
    name = os.path.splitext(os.path.basename(filename))[0]
    output_mask_path = os.path.join(
        os.path.dirname(filename), output_mask_path_relative
    )
    if not os.path.isdir(output_mask_path):
        os.mkdir(output_mask_path)
    insts = np.unique(instance_ids)
    zero_mask = np.zeros(shape=(instance_ids.shape[0]), dtype=np.int32)
    with open(filename, "w") as f:
        for idx, inst_id in enumerate(insts):
            if inst_id == 0:
                continue
            relative_output_mask_file = os.path.join(
                output_mask_path_relative, name + "_" + str(idx) + ".txt"
            )
            output_mask_file = os.path.join(
                output_mask_path, name + "_" + str(idx) + ".txt"
            )
            loc = np.where(instance_ids == inst_id)
            label_id = label_ids[loc[0][0]]
            confidence = confidences[loc[0][0]]
            f.write("%s %d %f\n" % (relative_output_mask_file, label_id, confidence))

            mask = np.copy(zero_mask)
            mask[loc[0]] = 1
            export_ids(output_mask_file, mask)


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------


def build_argument_parser():
    """Build the command-line interface without changing historical defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default=DEFAULT_SCANNET_ROOT,
        help="path to scannet dataset",
    )
    parser.add_argument(
        "--scans_dir",
        type=str,
        default=DEFAULT_SCANNET_ROOT,
        help="path to scannet dataset",
    )
    parser.add_argument(
        "--points_path",
        type=str,
        default=DEFAULT_POINTS_PATH,
        help="path to scannet dataset",
    )
    parser.add_argument(
        "--sam2d_path",
        type=str,
        default=DEFAULT_SAM2D_PATH,
        help="path to scannetv2 dataset",
    )
    parser.add_argument(
        "--data2d_path",
        type=str,
        default=DEFAULT_DATA2D_PATH,
        help="path to scannet dataset",
    )
    parser.add_argument("--scene_id", type=str, default=None)
    parser.add_argument(
        "--mask_name",
        type=str,
        default="semantic-sam10",
        help="which group of mask to use(fast-sam, sam-hq...)",
    )
    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="just a case for tweak parameter, will save file in particular names",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="demo_scannet_10view",
        help="save file with a prefix",
    )
    parser.add_argument(
        "--view_freq",
        type=int,
        default=10,
        help="how many views to select one view from",
    )
    parser.add_argument(
        "--thres_connect",
        type=str,
        default="0.9,0.5,5",
        help='dynamic threshold for progresive region growing, in the format of "start_thres,end_thres,stage_num',
    )
    parser.add_argument(
        "--dis_decay",
        type=float,
        default=0.5,
        help="weight decay for calculating seg-region affinity",
    )
    parser.add_argument(
        "--thres_dis",
        type=float,
        default=0.05,
        help="distance threshold for visibility test",
    )
    parser.add_argument(
        "--thres_merge",
        type=int,
        default=200,
        help="thres to merge small isolated regions in the postprocess",
    )
    parser.add_argument(
        "--max_neighbor_distance",
        type=int,
        default=2,
        help="max logical distance for taking priimtive neighbors into account",
    )
    parser.add_argument(
        "--similar_metric",
        type=str,
        default="2-norm",
        help="metric to compute similarities betweeen primitives, see utils.py/calcu_similar() for detail",
    )
    parser.add_argument(
        "--thres_trunc",
        type=float,
        default=0.0,
        help="trunc similarity that is under thres to 0",
    )
    parser.add_argument(
        "--from_points_thres",
        type=float,
        default=0,
        help="if > 0, use points as primitives for region growing in the first stage",
    )
    parser.add_argument(
        "--use_torch", action="store_true", help="use torch version or numpy version"
    )
    parser.add_argument(
        "--scannetpp",
        default=False,
        action="store_true",
        help="use scannet++ dataset(not debug yet)",
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default=DEFAULT_EVAL_DIR,
        help="where to save eval res",
    )
    parser.add_argument("--points_clipfeature_path", default=None, type=str)

    parser.add_argument(
        "--begin", type=int, default=0, help="how many views to select one view from"
    )
    parser.add_argument(
        "--end", type=int, default=312, help="how many views to select one view from"
    )

    return parser


def main():
    """Run ScanNetV2 inference for the requested validation scenes."""
    global args

    args = build_argument_parser().parse_args()
    _train_split, val_split = utils.get_splits(args.base_dir, args.scannetpp)
    print("eval_dir", args.eval_dir)
    seg_split = val_split
    seg_split = sorted(seg_split)

    if args.scene_id is not None:
        seg_split = args.scene_id.split(",")

    thres_connects = args.thres_connect.split(",")
    assert len(thres_connects) == 3
    args.thres_connect = np.linspace(
        float(thres_connects[0]), float(thres_connects[1]), int(thres_connects[2])
    )

    seg_split = seg_split[args.begin :]
    processed_file = os.path.join(DEFAULT_EVAL_DIR, "processed.txt")
    if not os.path.exists(processed_file):
        print(f"Creating missing file '{processed_file}'...")
        open(processed_file, "w").close()
    for scene_id in tqdm(seg_split):
        args.scene_id = scene_id
        with open(processed_file, "r") as f:
            existing_files = set(f.read().splitlines())
        if scene_id not in existing_files:
            with open(processed_file, "a") as f:
                f.write(scene_id + "\n")
            print(f"Scene '{scene_id}' Processed and added to '{processed_file}'.")
            args.scene_id = scene_id
            run_mv3dis(args, scene_id)
        else:
            print(f"Scene '{scene_id}' already exists in '{processed_file}'.")


if __name__ == "__main__":
    main()

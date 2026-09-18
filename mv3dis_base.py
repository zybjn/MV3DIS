"""Core MV3DIS inference pipeline."""

from collections import deque

import numpy as np
from tqdm import tqdm

import helpers.mv3dis_utils as utils


REGION_GROWTH_THRESHOLD = 0.5


class MV3DISBase:
    def __init__(self, points, args):
        self.points = points
        self.N = points.shape[0]
        self.max_neighbor_distance = args.max_neighbor_distance
        self.similar_metric = args.similar_metric
        self.args = args
        self.view_freq = args.view_freq
        self.dis_decay = args.dis_decay
        self.scans_dir = args.scans_dir

    def assign_label(
        self,
        points,
        thres_connect,
        vis_dis,
        max_neighbor_distance=2,
        similar_metric="2-norm",
        points_clipfeature=None,
        scene_id=None,
    ):
        """Run region growing, mask refinement, then small-segment merging.

        ``thres_connect``, ``max_neighbor_distance`` and ``scene_id`` remain in
        the interface for compatibility. The original implementation uses the
        fixed threshold below and the corresponding values stored on ``self``.
        """
        self._point_visibility_cache = None
        filter_results = [
            utils.filter_masks(masks, scores, return_overlap_cache=True)
            for masks, scores in tqdm(
                zip(self.mask_list, self.pred_score),
                total=len(self.mask_list),
                desc="Processing NMS",
            )
        ]
        keep_indices = [result[0] for result in filter_results]
        overlap_caches = [result[1] for result in filter_results]

        # The direct grounding_sam2 scene files already contain the masks and
        # confidences but do not provide separate 2d_mask_score PNGs. Build
        # the initial per-pixel labels from the same filtered masks here.
        if self.masks_2D is None:
            self.masks_2D = utils.update_masks(
                self.mask_list, keep_indices, self.pred_score
            )

        pts_cam, color_pixels, depth_pixels = self.parallel_world2cam_pixel(
            points, self.color_intrinsics, self.depth_intrinsics, self.poses
        )
        point_masks, points_seen, point_weights = self.get_points_label_seen(
            self.depths,
            pts_cam,
            color_pixels,
            depth_pixels,
            vis_dis=vis_dis,
            M=self.M,
        )
        color_x, color_y = self._split_and_clip_pixels(color_pixels)

        initial_seg_labels, split_sizes, merged_members = self.initial_region_growing(
            color_x,
            color_y,
            points_seen,
            point_masks,
            point_weights,
            keep_indices,
            similar_metric,
            points_clipfeature,
        )

        final_seg_labels, final_adjacency = self.mask_matching_refinement(
            color_pixels,
            color_x,
            color_y,
            points_seen,
            point_weights,
            keep_indices,
            overlap_caches,
            merged_members,
            initial_seg_labels,
            split_sizes,
            similar_metric,
            points_clipfeature,
        )

        # Stage 3: postprocess only after mask-guided refinement is complete.
        if self.args.thres_merge > 0:
            final_seg_labels = self.merge_small_segs(
                final_seg_labels, self.args.thres_merge, final_adjacency
            )
        final_labels = self._segment_labels_to_points(final_seg_labels)
        self._point_visibility_cache = None
        return final_labels

    def initial_region_growing(
        self,
        color_x,
        color_y,
        points_seen,
        point_masks,
        point_weights,
        keep_indices,
        similar_metric,
        points_clipfeature,
    ):
        """Stage 1: ordinary region growing on the original superpoints."""
        self._load_segment_data(points_clipfeature)
        print("Initial superpoint count:", self.seg_num)

        adjacency = self.get_seg_adjacency(
            color_y,
            color_x,
            similar_metric,
            points_seen,
            keep_indices,
            point_masks,
            point_weights,
        )
        seg_labels, split_sizes = self.region_growth_scannet(
            adjacency, REGION_GROWTH_THRESHOLD
        )
        point_labels = self._segment_labels_to_points(seg_labels)

        _, seg_num, merged_members, _, _, _ = self.get_seg_data(
            scans_dir=self.scans_dir,
            scene_id=self.scene_id,
            max_neighbor_distance=self.max_neighbor_distance,
            seg_ids=point_labels,
            points_clipfeature=points_clipfeature,
            ssp_split_sum=split_sizes,
            members_only=True,
        )
        print("Superpoint count after initial merge:", seg_num)
        return seg_labels, split_sizes, merged_members

    def mask_matching_refinement(
        self,
        color_pixels,
        color_x,
        color_y,
        points_seen,
        point_weights,
        keep_indices,
        overlap_caches,
        merged_members,
        initial_seg_labels,
        split_sizes,
        similar_metric,
        points_clipfeature,
    ):
        """Stage 2: match 2D masks to regions and refine the segmentation."""
        ordered_members = sorted(
            merged_members.values(),
            key=lambda member: len(member["point_indices"]),
            reverse=True,
        )
        ordered_members = [
            member for member in ordered_members if member["seg_num"] > -1
        ]
        ordered_members = dict(enumerate(ordered_members))

        (
            _,
            visible_coords,
            visible_weight_counts,
            visible_weight_coords,
        ) = utils.calculate_visible_points_per_superpoint_numpy(
            color_pixels,
            points_seen,
            len(ordered_members),
            ordered_members,
            point_weights,
        )
        member_counts = np.array(
            [len(member["point_indices"]) for member in ordered_members.values()]
        )
        top_frames, top_coords, top_weight_coords = (
            utils.calculate_top_k_visible_frames(
                visible_weight_counts,
                visible_coords,
                member_counts,
                visible_weight_coords,
                iou=0.3,
            )
        )
        mask_weights = utils.compute_mask_weights(
            top_frames,
            top_coords,
            top_weight_coords,
            self.mask_list,
            member_counts,
            merged_members,
            points_seen,
            color_x,
            color_y,
            point_weights,
            threshold=0.9,
        )
        mask_weights = [
            np.where(weights == 0, scores - 1, weights)
            for weights, scores in zip(mask_weights, self.pred_score)
        ]
        refined_keep_indices = [
            utils.filter_masks(masks, weights, overlap_cache=overlap_cache)
            for masks, weights, overlap_cache in tqdm(
                zip(self.mask_list, mask_weights, overlap_caches),
                total=len(self.mask_list),
                desc="Processing NMS",
            )
        ]

        self.masks_2D = utils.update_masks(
            self.mask_list, refined_keep_indices, mask_weights
        )
        refined_point_masks = self.update_all_label(color_pixels, points_seen)

        # Refine the first-stage regions while retaining their initial grouping.
        adjacency = self.get_seg_adjacency(
            color_y,
            color_x,
            similar_metric,
            points_seen,
            refined_keep_indices,
            refined_point_masks,
            point_weights,
        )
        refined_seg_labels, split_sizes = self.region_growth_scannet(
            adjacency,
            REGION_GROWTH_THRESHOLD,
            seg_labels=initial_seg_labels,
        )
        refined_point_labels = self._segment_labels_to_points(refined_seg_labels)

        # Rebuild superpoints from the refined labels, then perform the final
        # mask-guided region growth. Small regions are handled by assign_label.
        self._load_segment_data(
            points_clipfeature,
            seg_ids=refined_point_labels,
            split_sizes=split_sizes,
        )
        final_adjacency = self.get_seg_adjacency(
            color_y,
            color_x,
            similar_metric,
            points_seen,
            refined_keep_indices,
            refined_point_masks,
            point_weights,
        )
        final_seg_labels, _ = self.region_growth_scannet(
            final_adjacency, REGION_GROWTH_THRESHOLD
        )
        return final_seg_labels, final_adjacency

    def _load_segment_data(self, points_clipfeature, seg_ids=None, split_sizes=None):
        (
            self.seg_ids,
            self.seg_num,
            self.seg_members,
            self.seg_indirect_neighbors,
            self.seg_member_count,
            self.distance_matrix,
        ) = self.get_seg_data(
            scans_dir=self.scans_dir,
            scene_id=self.scene_id,
            max_neighbor_distance=self.max_neighbor_distance,
            seg_ids=seg_ids,
            points_clipfeature=points_clipfeature,
            ssp_split_sum=split_sizes,
        )
        self.seg_direct_neighbors = self.seg_indirect_neighbors[0]

    def _segment_labels_to_points(self, seg_labels):
        point_labels = np.zeros(self.N, dtype=int)
        for seg_id, member in self.seg_members.items():
            point_labels[member["point_indices"]] = seg_labels[seg_id]
        return point_labels

    def _split_and_clip_pixels(self, color_pixels):
        color_x, color_y = np.split(color_pixels, 2, axis=-1)
        color_x = color_x[..., 0].clip(0, self.CW - 1)
        color_y = color_y[..., 0].clip(0, self.CH - 1)
        return color_x, color_y

    def parallel_world2cam_pixel(
        self, points, color_intrinsics, depth_intrinsics, poses
    ):
        return utils.torch_world2cam_pixel(
            points, color_intrinsics, depth_intrinsics, poses
        )

    def get_points_label_seen(
        self, depths, pts_cam, color_pixes, depth_pixes, vis_dis, M=0
    ):
        """Project masks and depth consistency weights onto all 3D points."""
        batch_size = 50000
        all_seen = np.zeros((self.N, M), dtype=bool)
        all_labels = np.zeros((self.N, M), dtype=np.float32)
        all_weights = np.zeros((self.N, M), dtype=np.float32)

        for start in tqdm(range(0, self.N, batch_size)):
            stop = start + batch_size
            points_cam = pts_cam[start:stop]
            color_pixels = color_pixes[start:stop]
            depth_pixels = depth_pixes[start:stop]

            color_x, color_y = np.split(color_pixels, 2, axis=-1)
            color_x, color_y = color_x[..., 0], color_y[..., 0]
            in_image = (
                (0 <= color_x)
                * (color_x <= self.CW - 1)
                * (0 <= color_y)
                * (color_y <= self.CH - 1)
            )
            depth_x, depth_y = np.split(depth_pixels, 2, axis=-1)
            depth_x, depth_y = depth_x[..., 0], depth_y[..., 0]
            real_depth = points_cam[..., -1]
            captured_depth = depths[
                np.arange(M),
                depth_y.clip(0, self.DH - 1),
                depth_x.clip(0, self.DW - 1),
            ]
            visible = np.isclose(real_depth, captured_depth, rtol=vis_dis)
            depth_difference = np.abs(real_depth - captured_depth)
            tolerance = vis_dis * np.maximum(np.abs(real_depth), np.abs(captured_depth))
            tolerance = np.maximum(tolerance, 1e-6)
            seen = in_image * visible
            weights = np.where(seen, 1 - depth_difference / tolerance, 0)

            labels = self.masks_2D[
                np.arange(self.M),
                color_y.clip(0, self.CH - 1),
                color_x.clip(0, self.CW - 1),
            ]
            all_seen[start:stop] = seen
            all_labels[start:stop] = labels * seen
            all_weights[start:stop] = weights

        return all_labels, all_seen, all_weights

    def update_all_label(self, color_pixes, all_seen_flag):
        """Resample point labels after ``self.masks_2D`` has been updated."""
        _, frame_count = color_pixes.shape[:2]
        color_x, color_y = np.split(color_pixes, 2, axis=-1)
        color_x = np.clip(color_x[..., 0].astype(int), 0, self.CW - 1)
        color_y = np.clip(color_y[..., 0].astype(int), 0, self.CH - 1)
        labels = self.masks_2D[np.arange(frame_count)[None, :], color_y, color_x]
        return labels * all_seen_flag

    def get_seg_adjacency(
        self,
        color_y,
        color_x,
        similar_metric,
        points_seen,
        keep_indices,
        point_masks,
        point_weights,
    ):
        similar, confidence = self.get_neighbor_seg_similar_confidence_matrix(
            color_y,
            color_x,
            points_seen,
            similar_metric,
            self.args.thres_trunc,
            keep_indices,
            point_masks,
            point_weights,
        )
        return self.get_seg_adjacency_from_similar_confidence(similar, confidence)

    @staticmethod
    def col_nonzero_mean(values):
        sums = np.sum(values, axis=0)
        counts = np.count_nonzero(values, axis=0)
        return np.divide(
            sums,
            counts,
            out=np.zeros_like(sums, dtype=float),
            where=counts != 0,
        )

    def get_neighbor_seg_similar_confidence_matrix(
        self,
        color_y,
        color_x,
        points_seen,
        similar_metric,
        truncation_threshold,
        keep_indices,
        point_masks,
        point_weights,
    ):
        """Accumulate mask similarity and confidence for neighboring segments."""
        neighbors = self.seg_indirect_neighbors[self.max_neighbor_distance - 1]

        cache_key = (
            id(color_y),
            id(color_x),
            id(points_seen),
            id(keep_indices),
            id(point_weights),
            id(self.mask_list),
        )
        cache = self._point_visibility_cache
        if cache is not None and cache[0] == cache_key:
            points_with_mask, visible_with_mask, weighted_visibility = cache[1]
        else:
            points_with_mask = np.zeros((self.N, self.M), dtype=bool)
            for frame_id, masks in enumerate(self.mask_list):
                frame_union = np.any(masks[keep_indices[frame_id]], axis=0)
                points_with_mask[:, frame_id] = frame_union[
                    color_y[:, frame_id], color_x[:, frame_id]
                ]
            visible_with_mask = points_seen & points_with_mask
            weighted_visibility = visible_with_mask * point_weights
            self._point_visibility_cache = (
                cache_key,
                (points_with_mask, visible_with_mask, weighted_visibility),
            )

        segment_visibility = np.zeros((self.seg_num, self.M), dtype=np.float32)
        for seg_id, member in self.seg_members.items():
            point_indices = member["point_indices"]
            visible_ratio = (visible_with_mask[point_indices] > 0).sum(
                axis=0
            ) / point_indices.shape[0]
            mean_weight = self.col_nonzero_mean(weighted_visibility[point_indices])
            segment_visibility[seg_id] = mean_weight * visible_ratio

        return utils.torch_get_similar_confidence_matrix(
            neighbors,
            self.seg_ids,
            segment_visibility,
            point_masks,
            similar_metric,
            truncation_threshold,
            point_weights,
        )

    def get_seg_adjacency_from_similar_confidence(self, similar, confidence):
        assert similar.nonzero()[0].size > 0
        adjacency = np.zeros((self.seg_num, self.seg_num))
        nonzero = confidence.nonzero()
        adjacency[nonzero] = similar[nonzero] / confidence[nonzero]
        rows, cols = adjacency.nonzero()
        adjacency[rows, cols] = adjacency[cols, rows] = np.maximum(
            adjacency[rows, cols], adjacency[cols, rows]
        )
        return adjacency

    def merge_small_segs(self, seg_labels, merge_thres, adjacency):
        """Merge tiny regions into their strongest already-valid neighbor."""
        unique_labels, segment_counts = np.unique(seg_labels, return_counts=True)
        merged_labels = seg_labels.copy()
        mergeable = np.ones_like(seg_labels)

        for label, segment_count in zip(unique_labels, segment_counts):
            if segment_count > 2:
                continue
            seg_ids = np.flatnonzero(seg_labels == label)
            if self.seg_member_count[seg_ids].sum() < merge_thres:
                mergeable[seg_ids] = 0

        merge_count = 0
        while True:
            changed = False
            for label in unique_labels:
                seg_ids = np.flatnonzero(seg_labels == label)
                if mergeable[seg_ids[0]] > 0:
                    continue

                similarities = adjacency[seg_ids].sum(axis=0)
                for target in np.argsort(similarities)[::-1]:
                    if mergeable[target] == 0:
                        continue
                    if similarities[target] == 0:
                        break
                    merged_labels[seg_ids] = merged_labels[target]
                    mergeable[seg_ids] = 1
                    merge_count += 1
                    changed = True
                    break
            if not changed:
                break

        merged_labels[mergeable == 0] = 0
        print("original region number:", segment_counts.shape[0])
        print("mreging count:", merge_count)
        print("remove count:", (mergeable == 0).sum())
        return merged_labels

    def old_region_growth_scannet(self, adjacency, threshold, seg_labels=None):
        """Grow regions using affinity weighted by point count and distance."""
        node_count = len(adjacency)
        union_find = UnionFind(node_count)
        if seg_labels is not None:
            for label in np.unique(seg_labels):
                group = np.flatnonzero(np.asarray(seg_labels) == label)
                for index in range(1, group.shape[0]):
                    union_find.union(group[0], group[index])

        def inverse_distance(region, neighbor):
            distance = self.distance_matrix[region, neighbor]
            distance[distance == 0] = 0.0001
            return 1 / distance

        def affinity(region, neighbor):
            weights = inverse_distance(region, neighbor)
            denominator = np.sum(self.seg_member_count[region] * weights)
            if denominator == 0:
                return 0
            numerator = np.sum(
                adjacency[region, neighbor] * self.seg_member_count[region] * weights
            )
            return numerator / denominator

        neighbors = [row.nonzero()[0] for row in self.seg_indirect_neighbors[0]]
        processed = [False] * node_count
        for node in range(node_count):
            if processed[node]:
                continue
            visited = [False] * node_count
            queue = deque([node])
            visited[node] = True
            processed[node] = True

            while queue:
                current = queue.popleft()
                for neighbor in neighbors[current]:
                    if neighbor == current or visited[neighbor]:
                        continue

                    root_a = union_find.find(current)
                    root_b = union_find.find(neighbor)
                    region_a = [
                        index
                        for index in range(node_count)
                        if union_find.find(index) == root_a
                    ]
                    if root_a == root_b or not (
                        affinity(region_a, neighbor) >= threshold
                    ):
                        continue

                    if union_find.size[root_b] > 1:
                        region_b = [
                            index
                            for index in range(node_count)
                            if union_find.find(index) == root_b and index != neighbor
                        ]
                        if not (
                            affinity(region_a, neighbor) > affinity(region_b, neighbor)
                        ):
                            continue
                        union_find.delete_from_region(neighbor)

                    visited[neighbor] = True
                    processed[neighbor] = True
                    queue.append(neighbor)
                    union_find.union(current, neighbor)

        region_indices = np.array(
            [union_find.find(index) for index in range(node_count)], dtype=int
        )
        split_sizes = np.zeros(np.max(region_indices) + 1)
        for root in region_indices:
            split_sizes[root] = union_find.size[root]
        return region_indices, split_sizes

    def region_growth_scannet(self, adjacency, threshold, seg_labels=None):
        """Grow regions while caching each root's sorted member nodes."""
        node_count = len(adjacency)
        union_find = UnionFind(node_count)
        member_cache = SortedRegionMembers(
            union_find,
            node_count,
            debug=bool(getattr(self, "_debug_validate_region_members", False)),
        )
        if seg_labels is not None:
            for label in np.unique(seg_labels):
                group = np.flatnonzero(np.asarray(seg_labels) == label)
                for index in range(1, group.shape[0]):
                    member_cache.union(group[0], group[index])

        def inverse_distance(region, neighbor):
            distance = self.distance_matrix[region, neighbor]
            distance[distance == 0] = 0.0001
            return 1 / distance

        def affinity(region, neighbor):
            weights = inverse_distance(region, neighbor)
            denominator = np.sum(self.seg_member_count[region] * weights)
            if denominator == 0:
                return 0
            numerator = np.sum(
                adjacency[region, neighbor] * self.seg_member_count[region] * weights
            )
            return numerator / denominator

        neighbors = [row.nonzero()[0] for row in self.seg_indirect_neighbors[0]]
        processed = [False] * node_count
        for node in range(node_count):
            if processed[node]:
                continue
            visited = [False] * node_count
            queue = deque([node])
            visited[node] = True
            processed[node] = True

            while queue:
                current = queue.popleft()
                for neighbor in neighbors[current]:
                    if neighbor == current or visited[neighbor]:
                        continue

                    root_a = union_find.find(current)
                    root_b = union_find.find(neighbor)
                    region_a = member_cache.members(root_a)
                    if root_a == root_b or not (
                        affinity(region_a, neighbor) >= threshold
                    ):
                        continue

                    if union_find.size[root_b] > 1:
                        region_b = member_cache.members_without(root_b, neighbor)
                        if not (
                            affinity(region_a, neighbor) > affinity(region_b, neighbor)
                        ):
                            continue
                        member_cache.delete_from_region(neighbor)

                    visited[neighbor] = True
                    processed[neighbor] = True
                    queue.append(neighbor)
                    member_cache.union(current, neighbor)

        region_indices = np.array(
            [union_find.find(index) for index in range(node_count)], dtype=int
        )
        split_sizes = np.zeros(np.max(region_indices) + 1)
        for root in region_indices:
            split_sizes[root] = union_find.size[root]
        if member_cache.debug:
            self._debug_region_member_events = list(member_cache.debug_events)
        return region_indices, split_sizes

    def old_region_growth_scannetpp(self, adjacency, threshold, seg_labels=None):
        """Reference ScanNet++ region growing from the original implementation."""
        direct_neighbors = self.seg_indirect_neighbors[0]
        second_order_neighbors = self.seg_indirect_neighbors[1]
        second_order_neighbors[direct_neighbors == 1] = 0

        node_count = len(adjacency)
        union_find = UnionFind(node_count)
        if seg_labels is not None:
            groups = [
                np.where(np.asarray(seg_labels) == label)[0]
                for label in np.unique(seg_labels)
            ]
            for group in groups:
                for index in range(1, group.shape[0]):
                    union_find.union(group[0], group[index])

        split_count = np.ones(node_count)
        region_weight = np.zeros(node_count * 10)
        region_point_count = np.zeros(node_count * 10)

        def affinity(region, neighbor):
            region = np.asarray(region)
            direct = region[direct_neighbors[region, neighbor] == 1]
            second_order = region[second_order_neighbors[region, neighbor] == 1]

            weighted_similarity = np.sum(
                adjacency[direct, neighbor] * self.seg_member_count[direct]
            )
            weighted_similarity += 0.5 * np.sum(
                adjacency[second_order, neighbor] * self.seg_member_count[second_order]
            )
            total_weight = np.sum(self.seg_member_count[direct])
            total_weight += 0.5 * np.sum(self.seg_member_count[second_order])
            return weighted_similarity / total_weight if total_weight > 0 else 0

        neighbors = [row.nonzero()[0] for row in direct_neighbors]
        processed = [False] * node_count
        for node in range(node_count):
            if processed[node]:
                continue
            visited = [False] * node_count
            queue = deque([node])
            visited[node] = True
            processed[node] = True

            first_root = union_find.find(node)
            region_weight[first_root] += self.seg_member_count[node] * split_count[node]
            region_point_count[first_root] += self.seg_member_count[node]

            while queue:
                current = queue.popleft()
                for neighbor in neighbors[current]:
                    if (
                        neighbor == current
                        or visited[neighbor]
                        or not direct_neighbors[current, neighbor]
                    ):
                        continue

                    root_a = union_find.find(current)
                    root_b = union_find.find(neighbor)
                    region_a = [
                        index
                        for index in range(node_count)
                        if union_find.find(index) == root_a
                    ]
                    if root_a == root_b or not (
                        affinity(region_a, neighbor) >= threshold
                    ):
                        continue

                    if union_find.size[root_b] > 1:
                        region_b = [
                            index
                            for index in range(node_count)
                            if union_find.find(index) == root_b and index != neighbor
                        ]
                        if not (
                            affinity(region_a, neighbor) > affinity(region_b, neighbor)
                        ):
                            continue

                        region_weight[root_b] += self.seg_member_count[neighbor]
                        split_count[neighbor] += 1
                        region_weight[root_a] += (
                            self.seg_member_count[neighbor] * split_count[neighbor]
                        )
                        region_point_count[root_a] += self.seg_member_count[neighbor]
                        union_find.delete_from_region(neighbor)
                    else:
                        region_weight[root_a] += (
                            self.seg_member_count[neighbor] * split_count[neighbor]
                        )
                        region_point_count[root_a] += self.seg_member_count[neighbor]

                    visited[neighbor] = True
                    processed[neighbor] = True
                    queue.append(neighbor)
                    union_find.union(current, neighbor)

        region_indices = np.array(
            [union_find.find(index) for index in range(node_count)], dtype=int
        )
        split_sizes = np.zeros(np.max(region_indices) + 1)
        for root in region_indices:
            split_sizes[root] = union_find.size[root]
        return region_indices, split_sizes

    def region_growth_scannetpp(self, adjacency, threshold, seg_labels=None):
        """Grow ScanNet++ regions while caching each root's sorted members."""
        direct_neighbors = self.seg_indirect_neighbors[0]
        second_order_neighbors = self.seg_indirect_neighbors[1]
        second_order_neighbors[direct_neighbors == 1] = 0

        node_count = len(adjacency)
        union_find = UnionFind(node_count)
        member_cache = SortedRegionMembers(
            union_find,
            node_count,
            debug=bool(getattr(self, "_debug_validate_region_members", False)),
        )
        if seg_labels is not None:
            for label in np.unique(seg_labels):
                group = np.flatnonzero(np.asarray(seg_labels) == label)
                for index in range(1, group.shape[0]):
                    member_cache.union(group[0], group[index])

        split_count = np.ones(node_count)
        region_weight = np.zeros(node_count * 10)
        region_point_count = np.zeros(node_count * 10)

        def affinity(region, neighbor):
            region = np.asarray(region)
            direct = region[direct_neighbors[region, neighbor] == 1]
            second_order = region[second_order_neighbors[region, neighbor] == 1]

            weighted_similarity = np.sum(
                adjacency[direct, neighbor] * self.seg_member_count[direct]
            )
            weighted_similarity += 0.5 * np.sum(
                adjacency[second_order, neighbor] * self.seg_member_count[second_order]
            )
            total_weight = np.sum(self.seg_member_count[direct])
            total_weight += 0.5 * np.sum(self.seg_member_count[second_order])
            return weighted_similarity / total_weight if total_weight > 0 else 0

        neighbors = [row.nonzero()[0] for row in direct_neighbors]
        processed = [False] * node_count
        for node in range(node_count):
            if processed[node]:
                continue
            visited = [False] * node_count
            queue = deque([node])
            visited[node] = True
            processed[node] = True

            first_root = union_find.find(node)
            region_weight[first_root] += self.seg_member_count[node] * split_count[node]
            region_point_count[first_root] += self.seg_member_count[node]

            while queue:
                current = queue.popleft()
                for neighbor in neighbors[current]:
                    if (
                        neighbor == current
                        or visited[neighbor]
                        or not direct_neighbors[current, neighbor]
                    ):
                        continue

                    root_a = union_find.find(current)
                    root_b = union_find.find(neighbor)
                    region_a = member_cache.members(root_a)
                    if root_a == root_b or not (
                        affinity(region_a, neighbor) >= threshold
                    ):
                        continue

                    if union_find.size[root_b] > 1:
                        region_b = member_cache.members_without(root_b, neighbor)
                        if not (
                            affinity(region_a, neighbor) > affinity(region_b, neighbor)
                        ):
                            continue

                        region_weight[root_b] += self.seg_member_count[neighbor]
                        split_count[neighbor] += 1
                        region_weight[root_a] += (
                            self.seg_member_count[neighbor] * split_count[neighbor]
                        )
                        region_point_count[root_a] += self.seg_member_count[neighbor]
                        member_cache.delete_from_region(neighbor)
                    else:
                        region_weight[root_a] += (
                            self.seg_member_count[neighbor] * split_count[neighbor]
                        )
                        region_point_count[root_a] += self.seg_member_count[neighbor]

                    visited[neighbor] = True
                    processed[neighbor] = True
                    queue.append(neighbor)
                    member_cache.union(current, neighbor)

        region_indices = np.array(
            [union_find.find(index) for index in range(node_count)], dtype=int
        )
        split_sizes = np.zeros(np.max(region_indices) + 1)
        for root in region_indices:
            split_sizes[root] = union_find.size[root]
        if member_cache.debug:
            self._debug_region_member_events = list(member_cache.debug_events)
        return region_indices, split_sizes


class UnionFind:
    """Union-find supporting removal of one node into a fresh virtual root."""

    def __init__(self, node_count):
        self.virtual_node = 2 * node_count
        max_virtual_node = 10 * node_count
        self.parent = list(range(node_count, self.virtual_node))
        self.parent.extend(range(node_count, max_virtual_node))
        self.size = [1] * max_virtual_node

    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, first, second):
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.size[first_root] += self.size[second_root]

    def delete_from_region(self, node):
        root = self.find(node)
        self.size[root] -= 1
        self.parent[node] = self.virtual_node
        self.virtual_node += 1


class SortedRegionMembers:
    """Maintain sorted original-node members for each actual UnionFind root."""

    def __init__(self, union_find, node_count, debug=False):
        self.union_find = union_find
        self.node_count = node_count
        self.debug = debug
        self.root_members = {
            union_find.find(node): [node] for node in range(node_count)
        }
        self.debug_events = []
        self._validate("init")

    @staticmethod
    def _merge_sorted(first, second):
        merged = []
        first_index = 0
        second_index = 0
        while first_index < len(first) and second_index < len(second):
            if first[first_index] < second[second_index]:
                merged.append(first[first_index])
                first_index += 1
            else:
                merged.append(second[second_index])
                second_index += 1
        merged.extend(first[first_index:])
        merged.extend(second[second_index:])
        return merged

    def members(self, root):
        return self.root_members[root]

    def members_without(self, root, node):
        return [member for member in self.root_members[root] if member != node]

    def union(self, first, second):
        first_root = self.union_find.find(first)
        second_root = self.union_find.find(second)
        first_members = self.root_members[first_root]
        second_members = self.root_members[second_root]

        self.union_find.union(first, second)
        if first_root == second_root:
            self._record_and_validate("union_same_root", first, second, first_root)
            return first_root

        new_root = self.union_find.find(first)
        merged_members = self._merge_sorted(first_members, second_members)
        del self.root_members[first_root]
        del self.root_members[second_root]
        self.root_members[new_root] = merged_members
        self._record_and_validate("union", first, second, new_root)
        return new_root

    def delete_from_region(self, node):
        old_root = self.union_find.find(node)
        old_members = self.root_members[old_root]

        self.union_find.delete_from_region(node)
        new_root = self.union_find.find(node)
        self.root_members[old_root] = [
            member for member in old_members if member != node
        ]
        self.root_members[new_root] = [node]
        self._record_and_validate("delete", node, old_root, new_root)
        return new_root

    def _record_and_validate(self, *event):
        if self.debug:
            self.debug_events.append(event)
            self._validate(event[0])

    def _validate(self, mutation):
        if not self.debug:
            return
        actual_members = {}
        for node in range(self.node_count):
            root = self.union_find.find(node)
            actual_members.setdefault(root, []).append(node)
        if self.root_members != actual_members:
            raise AssertionError(
                f"root member cache mismatch after {mutation}: "
                f"cached={self.root_members}, actual={actual_members}"
            )
        for root, members in self.root_members.items():
            if members != sorted(members):
                raise AssertionError(f"unsorted members for root {root}: {members}")
            if self.union_find.size[root] != len(members):
                raise AssertionError(
                    f"size mismatch for root {root}: "
                    f"uf={self.union_find.size[root]}, members={len(members)}"
                )

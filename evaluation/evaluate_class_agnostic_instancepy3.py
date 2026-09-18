import math
import os, sys, argparse
import inspect
from copy import deepcopy
from tqdm import tqdm

try:
    import numpy as np
except ImportError:
    print("Failed to import numpy package.")
    sys.exit(-1)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util
import util_3d

parser = argparse.ArgumentParser()

parser.add_argument('--pred_path',default="", help='path to directory of predicted .txt files')
parser.add_argument('--gt_path',default="", help='path to directory of gt .txt files')
parser.add_argument('--output_file', default='', help='output file')
opt = parser.parse_args()

if opt.output_file == '':
    opt.output_file = os.path.join(opt.pred_path, 'class_agnostic_instance_evaluation.txt')

# ---------- Label info ---------- #
CLASS_LABELS = ['everything']
VALID_CLASS_IDS = np.array([1])
SEMANTIC_VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

ID_TO_LABEL = {}
LABEL_TO_ID = {}
for i in range(len(VALID_CLASS_IDS)):
    LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
opt.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
opt.min_region_sizes = np.array([100])
opt.distance_threshes = np.array([float('inf')])
opt.distance_confs = np.array([-float('inf')])

def evaluate_matches(matches):
    overlaps = opt.overlaps
    min_region_sizes = [opt.min_region_sizes[0]]
    dist_threshes = [opt.distance_threshes[0]]
    dist_confs = [opt.distance_confs[0]]

    ap = np.zeros((len(dist_threshes), len(CLASS_LABELS), len(overlaps)), float)
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]['pred']:
                    for label_name in CLASS_LABELS:
                        for p in matches[m]['pred'][label_name]:
                            if 'filename' in p:
                                pred_visited[p['filename']] = False
            for li, label_name in enumerate(CLASS_LABELS):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    pred_instances = matches[m]['pred'][label_name]
                    gt_instances = matches[m]['gt'][label_name]
                    gt_instances = [gt for gt in gt_instances if gt['instance_id'] >= 1000 and gt['vert_count'] >= min_region_size and gt['med_dist'] <= distance_thresh and gt['dist_conf'] >= distance_conf]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * -float("inf")
                    cur_match = np.zeros(len(gt_instances), dtype=bool)
                    for (gti, gt) in enumerate(gt_instances):
                        found_match = False
                        for pred in gt['matched_pred']:
                            if pred_visited[pred['filename']]:
                                continue
                            overlap = float(pred['intersection']) / (gt['vert_count'] + pred['vert_count'] - pred['intersection'])
                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['filename']] = True
                        if not found_match:
                            hard_false_negatives += 1

                    cur_true = cur_true[cur_match]
                    cur_score = cur_score[cur_match]

                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred['matched_gt']:
                            overlap = float(gt['intersection']) / (gt['vert_count'] + pred['vert_count'] - gt['intersection'])
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred['void_intersection']
                            for gt in pred['matched_gt']:
                                if gt['instance_id'] < 1000 or gt['vert_count'] < min_region_size or gt['med_dist'] > distance_thresh or gt['dist_conf'] < distance_conf:
                                    num_ignore += gt['intersection']
                            proportion_ignore = float(num_ignore) / pred['vert_count']
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                cur_score = np.append(cur_score, pred["confidence"])

                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                if has_gt and has_pred:
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted      = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                    num_prec_recall = len(unique_indices) + 1

                    num_examples = len(y_score)
                    num_true_examples = y_true_sorted_cumsum[-1]
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r

                    precision[-1] = 1.0
                    recall[-1] = 0.0

                    recall_for_conv = np.append([recall[0]], recall)
                    recall_for_conv = np.append(recall_for_conv, 0.0)

                    stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], 'valid')
                    ap_current = np.dot(precision, stepWidths)
                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float('nan')
                ap[di, li, oi] = ap_current
    return ap

def compute_averages(aps):
    d_inf = 0
    o50 = np.where(np.isclose(opt.overlaps, 0.5))
    o25 = np.where(np.isclose(opt.overlaps, 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt.overlaps, 0.25)))
    avg_dict = {}
    avg_dict['all_ap'] = np.nanmean(aps[d_inf, :, oAllBut25])
    avg_dict['all_ap_50%'] = np.nanmean(aps[d_inf, :, o50])
    avg_dict['all_ap_25%'] = np.nanmean(aps[d_inf, :, o25])
    avg_dict["classes"] = {}
    for (li, label_name) in enumerate(CLASS_LABELS):
        avg_dict["classes"][label_name] = {}
        avg_dict["classes"][label_name]["ap"] = np.average(aps[d_inf, li, oAllBut25])
        avg_dict["classes"][label_name]["ap50%"] = np.average(aps[d_inf, li, o50])
        avg_dict["classes"][label_name]["ap25%"] = np.average(aps[d_inf, li, o25])
    return avg_dict

def assign_instances_for_scan(pred_file, gt_file, pred_path, gt_path):
    try:
        pred_info=util_3d.read_masks(pred_file, pred_path)

    except Exception as e:
        util.print_error(f'Unable to load {pred_file}: {str(e)}')
    try:
        new_gt_ids = util_3d.load_ids(gt_file)
    except Exception as e:
        util.print_error(f'Unable to load {gt_file}: {str(e)}')

    gt_instances = util_3d.get_class_agnostic_instances(new_gt_ids, SEMANTIC_VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)
    gt_ids = new_gt_ids

    gt2pred = deepcopy(gt_instances)
    for label in gt2pred:
        for gt in gt2pred[label]:
            gt['matched_pred'] = []
    pred2gt = {}
    for label in CLASS_LABELS:
        pred2gt[label] = []
    num_pred_instances = 0
    bool_void = np.logical_not(np.in1d(gt_ids//1000, SEMANTIC_VALID_CLASS_IDS))
    for idx,pred_mask_file in enumerate(pred_info):
        label_id = VALID_CLASS_IDS[0]
        conf=pred_mask_file['conf']
        if label_id not in ID_TO_LABEL:
            continue
        label_name = ID_TO_LABEL[label_id]
        assert label_name == 'everything'
        pred_mask=pred_mask_file['mask']
        if len(pred_mask) != len(gt_ids):
            util.print_error(f'Wrong number of lines in {pred_mask_file} ({len(pred_mask)}) vs #mesh vertices ({len(gt_ids)}), please double check and/or re-download the mesh')
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < opt.min_region_sizes[0]:
            continue

        pred_instance = {'filename': pred_file+str(idx), 'pred_id': num_pred_instances, 'label_id': label_id, 'vert_count': num, 'confidence': conf, 'void_intersection': np.count_nonzero(np.logical_and(bool_void, pred_mask))}
        matched_gt = []
        for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
            intersection = np.count_nonzero(np.logical_and(gt_ids == gt_inst['instance_id'], pred_mask))
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy['intersection'] = intersection
                pred_copy['intersection'] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)
        pred_instance['matched_gt'] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)

    return gt2pred, pred2gt

def print_results(avgs):
    print("\n" + "#" * 64)
    line = "{:<15} : {:>15} {:>15} {:>15}".format("what", "AP", "AP_50%", "AP_25%")
    print(line)
    print("#" * 64)
    for (li, label_name) in enumerate(CLASS_LABELS):
        ap_avg = avgs["classes"][label_name]["ap"]
        ap_50o = avgs["classes"][label_name]["ap50%"]
        ap_25o = avgs["classes"][label_name]["ap25%"]
        line = "{:<15} : {:>15.3f} {:>15.3f} {:>15.3f}".format(label_name, ap_avg, ap_50o, ap_25o)
        print(line)
    all_ap_avg = avgs["all_ap"]
    all_ap_50o = avgs["all_ap_50%"]
    all_ap_25o = avgs["all_ap_25%"]
    print("-" * 64)
    line = "{:<15} : {:>15.3f} {:>15.3f} {:>15.3f}".format("average", all_ap_avg, all_ap_50o, all_ap_25o)
    print(line)
    print("")

def write_result_file(avgs, filename):
    _SPLITTER = ','
    with open(filename, 'w') as f:
        f.write(_SPLITTER.join(['class', 'class id', 'ap', 'ap50', 'ap25']) + '\n')
        for i in range(len(VALID_CLASS_IDS)):
            class_name = CLASS_LABELS[i]
            class_id = VALID_CLASS_IDS[i]
            ap = avgs["classes"][class_name]["ap"]
            ap50 = avgs["classes"][class_name]["ap50%"]
            ap25 = avgs["classes"][class_name]["ap25%"]
            f.write(_SPLITTER.join([str(x) for x in [class_name, class_id, ap, ap50, ap25]]) + '\n')

def evaluate(pred_files, gt_files, pred_path, output_file):
    print(f'Evaluating {len(pred_files)} scans...')
    matches = {}
    for i in tqdm(range(len(pred_files)), desc='Finding matches'):
        matches_key = os.path.abspath(gt_files[i])
        gt2pred, pred2gt = assign_instances_for_scan(pred_files[i], gt_files[i], pred_path, opt.gt_path)
        matches[matches_key] = {'gt': gt2pred, 'pred': pred2gt}

    ap_scores = evaluate_matches(matches)
    avgs = compute_averages(ap_scores)
    print_results(avgs)
    write_result_file(avgs, output_file)

def main():

    split_path = os.path.join(parentdir, "meta_data", "scannetv2_val.txt")
    with open(split_path, "r") as f:
        val_split = [s.strip() for s in f.readlines()]
    val_split = sorted(val_split)
    pred_files = sorted(val_split)
    pred_files=pred_files[:]
    print(f'Evaluating: {val_split}')
    if len(pred_files) < 5:
        opt.output_file = os.path.join(opt.pred_path, 'test_class_agnostic_instance_evaluation.txt')

    gt_files = []
    for i, scene_id in enumerate(pred_files):
        gt_file = os.path.join(opt.gt_path, scene_id + ".txt")
        if not os.path.isfile(gt_file):
            util.print_error(f'Result file {scene_id} does not match any gt file', user_fault=True)
        gt_files.append(gt_file)
        pred_files[i] = os.path.join(opt.pred_path, scene_id+".pth")

    evaluate(pred_files, gt_files, opt.pred_path, opt.output_file)

if __name__ == '__main__':
    main()

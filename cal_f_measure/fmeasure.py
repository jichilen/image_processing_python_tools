from __future__ import division
import os
from os.path import join, exists, basename, splitext, split
import re
import sys
import shapely
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import glob
import json
from tqdm import tqdm
import pickle
import zipfile
import shutil
import csv
from collections import defaultdict
import operator

IOU_THRESH = 0.5
N_TEST = 2963


def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def polygon_from_str(line):
    """
    Create a shapely polygon object from gt or dt line.
    """
    polygon_points = [float(o) for o in line.split(',')[:8]]
    polygon_points = np.array(polygon_points).reshape(4, 2)
    polygon = Polygon(polygon_points).convex_hull
    return polygon


def polygon_iou(poly1, poly2):
    """
    Intersection over union between two shapely polygons.
    """
    if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            union_area = poly1.area + poly2.area - inter_area
            # union_area = poly1.union(poly2).area
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou


def det_eval(gt_dir, dt_parent,save_parent):
  """
  Evaluation detection by calculating the maximum f-measure across all thresholds.
  ARGS
    gt_dir: the directory of groundtruth files
    dt_dir: the directory of detection results files
    save_parent: the directory for saving evaluation results
  RETURN
    nothing returned
  """
  # load all groundtruths into a dict of {<image-name>: <list-of-polygons>}

  res_flag = False
  n_gt = 0
  all_gt = {}
  gt_files = glob.glob(join(gt_dir, 'image_*.txt'))
  # assert(len(gt_files) == N_TEST)
  print('Number of GT files: %d' % len(gt_files))
  for gt_file in gt_files:
    with open(gt_file, 'r') as f:
      gt_lines = f.readlines()
      polygons = [polygon_from_str(o) for o in gt_lines]
      n_gt += len(polygons)
    #找文件名 basename 取/后面   splitext去掉扩展名
    fname = splitext(basename(gt_file))[0]
    all_gt[fname] = polygons

  # scores and match status of all dts in a single list
  all_dt_match = []
  all_dt_scores = []

  def _recursive_find_sub_dirs(curr_dir):
    for root, subdirs, files in os.walk(curr_dir):
      for s in files:
        if s.endswith(".txt"):
          #dt_files.append(os.path.join(root, s ))
          #sub_id_sub_dir_pairs.append(os.path.join(root, s))
          return root
        else:
          _recursive_find_sub_dirs(s)
    return ''
  dt_dir = _recursive_find_sub_dirs(dt_parent)
  if not dt_dir:
    s = 'file type not consistent'
    return [res_flag, s,0,0,0,0,'']
  # for every detection, calculate its match to groundtruth
  dt_files = glob.glob(join(dt_dir, '*.txt'))
  print('Number of DT files: %d' % len(dt_files))
  p = re.compile(r'.*(image_\d+)\.txt')
  print('Calculating matches')
  try:
    for dt_file in tqdm(dt_files):
      # find corresponding gt file
      fname = basename(dt_file)
      key = p.match(fname).group(1)

      if key not in all_gt:
        print('Result %s not found in groundtruths! This file will be ignored')
        continue

      # calculate matches to groundtruth and append to list
      gt_polygons = all_gt[key]
      with open(dt_file, 'r') as f:
        dt_lines = [o.strip() for o in f.readlines()]
      dt_polygons = [polygon_from_str(o) for o in dt_lines]
      # dt_match = []
      # gt_match = [False] * len(gt_polygons)
      # for dt_poly in dt_polygons:
      #   match = False
      #   for i, gt_poly in enumerate(gt_polygons):
      #     if gt_match[i] == False and polygon_iou(dt_poly, gt_poly) >= IOU_THRESH:
      #       gt_match[i] = True
      #       match = True
      #       break
      #   dt_match.append(match)
      # all_dt_match.extend(dt_match)

      #####################################
      # match scheme by YMK
      dt_match = [False] * len(dt_polygons)
      gt_match = [False] * len(gt_polygons)
      all_ious = defaultdict(tuple)
      for index_gt, gt_poly in enumerate(gt_polygons):
        for index_dt, dt_poly in enumerate(dt_polygons):
          iou = polygon_iou(dt_poly, gt_poly)
          if iou >= IOU_THRESH:
            all_ious[(index_gt, index_dt)] = iou
      sorted_ious = sorted(all_ious.items(), key=operator.itemgetter(1), reverse=True)
      sorted_gt_dt_pairs = [item[0] for item in sorted_ious]
      for gt_dt_pair in sorted_gt_dt_pairs:
        index_gt, index_dt = gt_dt_pair
        if gt_match[index_gt] == False and dt_match[index_dt] == False:
          gt_match[index_gt] = True
          dt_match[index_dt] = True
      all_dt_match.extend(dt_match)
      #####################################

      # calculate scores and append to list
      dt_scores = [float(o.split(',')[8]) for o in dt_lines]
      all_dt_scores.extend(dt_scores)
    # calculate precision, recall and f-measure at all thresholds
    all_dt_match = np.array(all_dt_match, dtype=np.bool).astype(np.int)
    all_dt_scores = np.array(all_dt_scores)

    sort_idx = np.argsort(all_dt_scores)[::-1] # sort in descending order
    all_dt_match = all_dt_match[sort_idx]
    all_dt_scores = all_dt_scores[sort_idx]

    n_pos = np.cumsum(all_dt_match)
    n_dt = np.arange(1, len(all_dt_match)+1)
    precision = n_pos.astype(np.float) / n_dt.astype(np.float)
    recall = n_pos.astype(np.float) / float(n_gt)
    eps = 1e-9
    fmeasure = 2.0 / ((1.0 / (precision + eps)) + (1.0 / (recall + eps)))


    rec = n_pos / float(n_gt)
    prec = n_pos / np.maximum(n_dt, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

  except Exception as e:
    # s = str(e)
    s = repr(e)
    return [res_flag, s,0,0,0,0,'']
  else:
    # find maximum fmeasure
    max_idx = np.argmax(fmeasure)

    eval_results = {
      'fmeasure': fmeasure[max_idx],
      'precision': precision[max_idx],
      'recall': recall[max_idx],
      'ap': ap,
      'threshold': all_dt_scores[max_idx],
      'all_precisions': precision,
      'all_recalls': recall
    }

    # # evaluation summary
    # print('=================================================================')
    # print('Maximum f-measure: %f' % eval_results['fmeasure'])
    # print('  |-- precision:   %f' % eval_results['precision'])
    # print('  |-- recall:      %f' % eval_results['recall'])
    # print('  |-- threshold:   %f' % eval_results['threshold'])
    # print('=================================================================')

    # save evaluation results
    dt_name = os.path.split(dt_dir)[-1]+'_eval'
    save_dir = os.path.join(save_parent, dt_name)
    if not exists(save_dir):
      os.makedirs(save_dir)
    data_save_path = join(save_dir, 'eval_data.pkl')
    with open(data_save_path, 'wb') as f:
      pickle.dump(eval_results, f)
    print('Evaluation results data written to {}'.format(data_save_path))

    # plot precision-recall curve
    vis_save_path = join(save_dir, 'pr_curve.png')
    plt.clf()
    plt.plot(recall, precision)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Precision-Recall Curve')
    plt.grid()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(vis_save_path, dpi=200)
    print('Precision-recall curve written to {}'.format(vis_save_path))
    # save evaluation results
    if not exists(save_dir):
      os.makedirs(save_dir)
    data_save_path = join(save_dir, 'eval_data.pkl')
    with open(data_save_path, 'wb') as f:
      pickle.dump(eval_results, f)
    print('Evaluation results data written to {}'.format(data_save_path))
    dst_dir = save_dir
    src_root = os.path.split(dst_dir)[0]
    src_dir = os.path.split(dst_dir)[1]
    zip_dir = shutil.make_archive(dst_dir, 'zip', src_root, src_dir)
    res_flag = True
    return [res_flag, '', eval_results['fmeasure'],eval_results['precision'],eval_results['recall'],
    eval_results['ap'],zip_dir]



if __name__ == "__main__":

    gt_dir = 'gt/'
    dt_dir = 'train/'

    save_parent = 'save/'



    [res_flag, s, f_measure, precision, recall,ap, zip_dir] = det_eval(gt_dir, dt_dir, save_parent)


    print ([res_flag, f_measure, precision, recall,ap])

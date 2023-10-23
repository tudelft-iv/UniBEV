import os
import random

import json
from typing import Dict, Any

import mmcv
import numpy as np

from matplotlib import pyplot as plt
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, DetectionMetricDataList
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.utils.splits import val
from nuscenes.eval.common.utils import boxes_to_sensor, center_distance
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from nuscenes.eval.detection.data_classes import DetectionConfig

class DetectionEvalVis(DetectionEval):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)


    def main_vis(self,
             plot_examples: int = 0,
             show_GT = True,
             render_curves: bool = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        print("In CustomVis Tool:")
        # Select a random but fixed subset to plot.
        random.seed(42)
        sample_tokens = list(self.sample_tokens)
        random.shuffle(sample_tokens)
        sample_tokens = sample_tokens[:plot_examples]

        # Visualize samples.
        example_dir = os.path.join(self.output_dir, 'examples')
        if not os.path.isdir(example_dir):
            os.mkdir(example_dir)
        for sample_token in sample_tokens:
            visualize_sample_custom(self.nusc,
                                    sample_token,
                                    self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                    # Don't render test GT.
                                    self.pred_boxes,
                                    eval_range=max(self.cfg.class_range.values()),
                                    show_GT = show_GT,
                                    savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

    def main_vis_per_scene(self,
                           conf_th = 0.15,
                           scene_list = None,
                           num_scenes = 0,
                           auto_conf=False,
                           with_GT = False,
                           wo_GT = False):
        print("Get Qualitative Results per scene:")
        val_split = scene_list

        scene_token = {}
        for scene in self.nusc.scene:
            scene_token[scene['name']] = scene['token']
        random.seed(42)
        random.shuffle(val_split)
        val_scene_names = val_split[:num_scenes]
        mmcv.mkdir_or_exist(self.output_dir)

        for scene_name in val_scene_names:
            scene = self.nusc.get('scene', scene_token[scene_name])
            first_sample_token = scene['first_sample_token']
            sample_token = first_sample_token
            while sample_token != '':
                w_GT_path = os.path.join(self.output_dir, 'with_GT', scene_name)
                mmcv.mkdir_or_exist(w_GT_path)
                if with_GT and not os.path.exists(os.path.join(w_GT_path,'{}.png'.format(sample_token))):
                    visualize_sample_custom(self.nusc,
                                            sample_token,
                                            self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                            # Don't render test GT.
                                            self.pred_boxes,
                                            conf_th= conf_th,
                                            auto_conf=auto_conf,
                                            class_names = self.cfg.class_names,
                                            eval_range=max(self.cfg.class_range.values()),
                                            show_GT=True,
                                            savepath=os.path.join(w_GT_path,'{}.png'.format(sample_token)))
                    # visualize_sample_custom_naive(self.nusc,
                    #                         sample_token,
                    #                         self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                    #                         # Don't render test GT.
                    #                         self.pred_boxes,
                    #                         conf_th= conf_th,
                    #                         eval_range=max(self.cfg.class_range.values()),
                    #                         show_GT=True,
                    #                         savepath=os.path.join(w_GT_path,'{}.png'.format(sample_token)))
                if wo_GT:
                    wo_GT_path = os.path.join(self.output_dir, 'without_GT', scene_name)
                    mmcv.mkdir_or_exist(wo_GT_path)
                    visualize_sample_custom(self.nusc,
                                            sample_token,
                                            self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                            # Don't render test GT.
                                            self.pred_boxes,
                                            conf_th= conf_th,
                                            auto_conf = auto_conf,
                                            class_names = self.cfg.class_names,
                                            eval_range=max(self.cfg.class_range.values()),
                                            show_GT=False,
                                            savepath=os.path.join(wo_GT_path,'{}.png'.format(sample_token)))
                sample_token = self.nusc.get('sample', sample_token)['next']

def visualize_sample_custom(nusc: NuScenes,
                            sample_token: str,
                            gt_boxes: EvalBoxes,
                            pred_boxes: EvalBoxes,
                            nsweeps: int = 3,
                            conf_th: float = 0.15,
                            auto_conf = True,
                            class_names=None,
                            eval_range: float = 50,
                            show_GT = True,
                            verbose: bool = True,
                            savepath: str = None) -> None:
    """
        Visualizes a sample from BEV with annotations and detection results.
        :param nusc: NuScenes object.
        :param sample_token: The nuScenes sample token.
        :param gt_boxes: Ground truth boxes grouped by sample.
        :param pred_boxes: Prediction grouped by sample.
        :param nsweeps: Number of sweeps used for lidar visualization.
        :param conf_th: The confidence threshold used to filter negatives.
        :param eval_range: Range in meters beyond which boxes are ignored.
        :param verbose: Whether to print to stdout.
        :param savepath: If given, saves the the rendering here instead of displaying.
        """
    # Retrieve sensor & pose records.
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Get boxes.
    boxes_gt_global = gt_boxes[sample_token]
    boxes_est_global = pred_boxes[sample_token]

    # Map GT boxes to lidar.
    boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)

    if auto_conf:
        predict_boxes = get_boxes_auto_conf_th(boxes_gt_global, boxes_est_global, class_names, center_distance, 2)
        tp_predict_boxes = []
        fp_predict_boxes = []
        for pred_boxes_cls in predict_boxes:
            tp_predict_boxes.extend(pred_boxes_cls['tp_boxes'])
            fp_predict_boxes.extend(pred_boxes_cls['fp_boxes'])
        # Map EST boxes to lidar.
        tp_boxes_est = boxes_to_sensor(tp_predict_boxes, pose_record, cs_record)
        fp_boxes_est = boxes_to_sensor(fp_predict_boxes, pose_record, cs_record)
    else:
        assert conf_th is not None
        predict_boxes = get_boxes_conf_th(boxes_gt_global, boxes_est_global, class_names, center_distance, conf_th=conf_th, dist_th=2)
        tp_predict_boxes = []
        fp_predict_boxes = []
        for pred_boxes_cls in predict_boxes:
            tp_predict_boxes.extend(pred_boxes_cls['tp_boxes'])
            fp_predict_boxes.extend(pred_boxes_cls['fp_boxes'])
        # Map EST boxes to lidar.
        tp_boxes_est = boxes_to_sensor(tp_predict_boxes, pose_record, cs_record)
        fp_boxes_est = boxes_to_sensor(fp_predict_boxes, pose_record, cs_record)
    # Map EST boxes to lidar.
    boxes_est = boxes_to_sensor(boxes_est_global, pose_record, cs_record)

    # Add scores to EST boxes.
    for box_est, box_est_global in zip(boxes_est, boxes_est_global):
        box_est.score = box_est_global.detection_score

    # Get point cloud in lidar frame.
    pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)

    # Init axes.
    _, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.set_facecolor('black')
    ax.tick_params(left = False,
                   right = False ,
                   labelleft = False ,
                   labelbottom = False,
                   bottom = False)
    # Show point cloud.
    points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    # colors = np.minimum(1, dists / eval_range)
    ax.scatter(points[0, :], points[1, :], c='gainsboro', s=0.2)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    # Show GT boxes.
    if show_GT:
        for box in boxes_gt:
            box.render(ax, view=np.eye(4), colors=('lime', 'lime', 'lime'), linewidth=2)


    # # Show EST boxes.
    # for box in boxes_est:
    #     # Show only predictions with a high score.
    #     assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
    #
    #     if box.score >= conf_th:
    #         box.render(ax, view=np.eye(4), colors=('red', 'red', 'red'), linewidth=1)
    for tp_box in tp_boxes_est:
        tp_box.render(ax, view=np.eye(4), colors=('red', 'red', 'red'), linewidth=1)
    for fp_box in fp_boxes_est:
        fp_box.render(ax, view=np.eye(4), colors=('yellow', 'yellow', 'yellow'), linewidth=1)
    # Show EST boxes.
    # for box in boxes_est:
    #     # Show only predictions with a high score.
    #     assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
    #     if box.score >= conf_th:
    #         box.render(ax, view=np.eye(4), colors=('red', 'red', 'red'), linewidth=1)

    # Limit visible range.
    axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

    # Show / save plot.
    if verbose:
        print('Rendering sample token %s' % sample_token)
    # plt.title(sample_token)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        #plt.show()
        plt.close()
    else:
        plt.show()

def get_boxes_auto_conf_th(gt_boxes, pred_boxes, class_names, dist_fcn_callable, dist_th=2):
    conf_th_list = [0.15, 0.2, 0.3, 0.4, 0.5]
    filter_boxes_list = []
    for class_name in class_names:
        f1_score_list_per_class = []
        boxes_list_per_class = []
        for conf_th in conf_th_list:
            npos = len([1 for gt_box in gt_boxes if gt_box.detection_name == class_name])
            pred_boxes_list = [box for box in pred_boxes if box.detection_name == class_name]
            pred_boxes_list = [box for box in pred_boxes_list if box.detection_score > conf_th]
            pred_confs = [box.detection_score for box in pred_boxes_list]
            sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

            tp = []  # Accumulator of true positives.
            fp = []  # Accumulator of false positives.
            conf = []  # Accumulator of confidences.

            taken = set()
            tp_boxes = []
            fp_boxes = []
            for ind in sortind:
                pred_box = pred_boxes_list[ind]
                min_dist = np.inf
                match_gt_idx = None

                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_box.detection_name == class_name and not gt_idx in taken:
                        this_distance = dist_fcn_callable(gt_box, pred_box)
                        if this_distance < min_dist:
                            min_dist = this_distance
                            match_gt_idx = gt_idx

                is_match = min_dist < dist_th

                if is_match:
                    taken.add(match_gt_idx)
                    tp_boxes.append(pred_box)
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)
                    fp_boxes.append(pred_box)

            tp = np.sum(tp).astype(float)
            fp = np.sum(fp).astype(float)

            precision = tp / (fp + tp)
            recall = tp / float(npos)

            F1 = 2*(precision * recall)/(precision + recall)
            f1_score_list_per_class.append(F1)
            boxes_list_per_class.append((tp_boxes, fp_boxes))

        filter_boxes_list.append(dict(class_name= class_name,
                                      tp_boxes = boxes_list_per_class[np.argmax(f1_score_list_per_class)][0],
                                      fp_boxes = boxes_list_per_class[np.argmax(f1_score_list_per_class)][1]))

    return filter_boxes_list


def get_boxes_conf_th(gt_boxes, pred_boxes, class_names, dist_fcn_callable, conf_th=0.15, dist_th=2):
    filter_boxes_list = []
    for class_name in class_names:
        f1_score_list_per_class = []
        boxes_list_per_class = []
        npos = len([1 for gt_box in gt_boxes if gt_box.detection_name == class_name])
        pred_boxes_list = [box for box in pred_boxes if box.detection_name == class_name]
        pred_boxes_list = [box for box in pred_boxes_list if box.detection_score > conf_th]
        pred_confs = [box.detection_score for box in pred_boxes_list]
        sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

        tp = []  # Accumulator of true positives.
        fp = []  # Accumulator of false positives.

        taken = set()
        tp_boxes = []
        fp_boxes = []
        for ind in sortind:
            pred_box = pred_boxes_list[ind]
            min_dist = np.inf
            match_gt_idx = None

            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_box.detection_name == class_name and not gt_idx in taken:
                    this_distance = dist_fcn_callable(gt_box, pred_box)
                    if this_distance < min_dist:
                        min_dist = this_distance
                        match_gt_idx = gt_idx

            is_match = min_dist < dist_th

            if is_match:
                taken.add(match_gt_idx)
                tp_boxes.append(pred_box)
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)
                fp_boxes.append(pred_box)

        filter_boxes_list.append(dict(class_name= class_name,
                                      tp_boxes = tp_boxes,
                                      fp_boxes = fp_boxes))

    return filter_boxes_list

def visualize_sample_custom_naive(nusc: NuScenes,
                            sample_token: str,
                            gt_boxes: EvalBoxes,
                            pred_boxes: EvalBoxes,
                            nsweeps: int = 3,
                            conf_th: float = 0.15,
                            eval_range: float = 50,
                            show_GT = True,
                            verbose: bool = True,
                            savepath: str = None) -> None:
    """
        Visualizes a sample from BEV with annotations and detection results.
        :param nusc: NuScenes object.
        :param sample_token: The nuScenes sample token.
        :param gt_boxes: Ground truth boxes grouped by sample.
        :param pred_boxes: Prediction grouped by sample.
        :param nsweeps: Number of sweeps used for lidar visualization.
        :param conf_th: The confidence threshold used to filter negatives.
        :param eval_range: Range in meters beyond which boxes are ignored.
        :param verbose: Whether to print to stdout.
        :param savepath: If given, saves the the rendering here instead of displaying.
        """
    # Retrieve sensor & pose records.
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Get boxes.
    boxes_gt_global = gt_boxes[sample_token]
    boxes_est_global = pred_boxes[sample_token]

    # Map GT boxes to lidar.
    boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)

    # Map EST boxes to lidar.
    boxes_est = boxes_to_sensor(boxes_est_global, pose_record, cs_record)

    # Add scores to EST boxes.
    for box_est, box_est_global in zip(boxes_est, boxes_est_global):
        box_est.score = box_est_global.detection_score

    # Get point cloud in lidar frame.
    pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)

    # Init axes.
    _, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.set_facecolor('black')
    ax.tick_params(left = False,
                   right = False ,
                   labelleft = False ,
                   labelbottom = False,
                   bottom = False)
    # Show point cloud.
    points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    # colors = np.minimum(1, dists / eval_range)
    ax.scatter(points[0, :], points[1, :], c='gainsboro', s=0.2)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    # Show GT boxes.
    if show_GT:
        for box in boxes_gt:
            box.render(ax, view=np.eye(4), colors=('lime', 'lime', 'lime'), linewidth=2)

    # Show EST boxes.
    for box in boxes_est:
        # Show only predictions with a high score.
        assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
        if box.score >= conf_th:
            box.render(ax, view=np.eye(4), colors=('red', 'red', 'red'), linewidth=1)

    # Limit visible range.
    axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

    # Show / save plot.
    if verbose:
        print('Rendering sample token %s' % sample_token)
    # plt.title(sample_token)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        #plt.show()
        plt.close()
    else:
        plt.show()
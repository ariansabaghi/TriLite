import cv2
import numpy as np
import os

import torch.utils.data as torchdata

from data_loaders import get_image_ids, get_mask_paths
from data_loaders import get_bounding_boxes
from data_loaders import get_image_sizes
from util import check_scoremap_validity
from util import check_box_convention
from collections import defaultdict


_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0
_RESIZE_LENGTH = 224


def calculate_multiple_iou(box_a, box_b):
    """
    Args:
        box_a: numpy.ndarray(dtype=np.int, shape=(num_a, 4))
            x0y0x1y1 convention.
        box_b: numpy.ndarray(dtype=np.int, shape=(num_b, 4))
            x0y0x1y1 convention.
    Returns:
        ious: numpy.ndarray(dtype=np.int, shape(num_a, num_b))
    """
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]

    check_box_convention(box_a, 'x0y0x1y1')
    check_box_convention(box_b, 'x0y0x1y1')

    # num_a x 4 -> num_a x num_b x 4
    box_a = np.tile(box_a, num_b)
    box_a = np.expand_dims(box_a, axis=1).reshape((num_a, num_b, -1))

    # num_b x 4 -> num_b x num_a x 4
    box_b = np.tile(box_b, num_a)
    box_b = np.expand_dims(box_b, axis=1).reshape((num_b, num_a, -1))

    # num_b x num_a x 4 -> num_a x num_b x 4
    box_b = np.transpose(box_b, (1, 0, 2))

    # num_a x num_b
    min_x = np.maximum(box_a[:, :, 0], box_b[:, :, 0])
    min_y = np.maximum(box_a[:, :, 1], box_b[:, :, 1])
    max_x = np.minimum(box_a[:, :, 2], box_b[:, :, 2])
    max_y = np.minimum(box_a[:, :, 3], box_b[:, :, 3])

    # num_a x num_b
    area_intersect = (np.maximum(0, max_x - min_x + 1)
                      * np.maximum(0, max_y - min_y + 1))
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0] + 1) *
              (box_a[:, :, 3] - box_a[:, :, 1] + 1))
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0] + 1) *
              (box_b[:, :, 3] - box_b[:, :, 1] + 1))

    denominator = area_a + area_b - area_intersect
    degenerate_indices = np.where(denominator <= 0)
    denominator[degenerate_indices] = 1

    ious = area_intersect / denominator
    ious[degenerate_indices] = 0
    return ious


def resize_bbox(box, image_size, resize_size):
    """
    Args:
        box: iterable (ints) of length 4 (x0, y0, x1, y1)
        image_size: iterable (ints) of length 2 (width, height)
        resize_size: iterable (ints) of length 2 (width, height)

    Returns:
         new_box: iterable (ints) of length 4 (x0, y0, x1, y1)
    """
    check_box_convention(np.array(box), 'x0y0x1y1')
    box_x0, box_y0, box_x1, box_y1 = map(float, box)
    image_w, image_h = map(float, image_size)
    new_image_w, new_image_h = map(float, resize_size)

    newbox_x0 = box_x0 * new_image_w / image_w
    newbox_y0 = box_y0 * new_image_h / image_h
    newbox_x1 = box_x1 * new_image_w / image_w
    newbox_y1 = box_y1 * new_image_h / image_h
    return int(newbox_x0), int(newbox_y0), int(newbox_x1), int(newbox_y1)


def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list,
                                  multi_contour_eval=False):
    """
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation

    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """
    check_scoremap_validity(scoremap)
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)


        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0, y0, x1, y1])

        return np.asarray(estimated_boxes), len(contours),

    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    for threshold in scoremap_threshold_list:
        boxes, number_of_box = scoremap2bbox(threshold)
        estimated_boxes_at_each_thr.append(boxes)
        number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list


class CamDataset(torchdata.Dataset):
    def __init__(self, scoremap_path, image_ids):
        self.scoremap_path = scoremap_path
        self.image_ids = image_ids

    def _load_cam(self, image_id):
        scoremap_file = os.path.join(self.scoremap_path, image_id + '.npy')
        return np.load(scoremap_file)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        cam = self._load_cam(image_id)
        return cam, image_id

    def __len__(self):
        return len(self.image_ids)



class LocalizationEvaluator(object):
    """ Abstract class for localization evaluation over score maps.

    The class is designed to operate in a for loop (e.g. batch-wise cam
    score map computation). At initialization, __init__ registers paths to
    annotations and data containers for evaluation. At each iteration,
    each score map is passed to the accumulate() method along with its image_id.
    After the for loop is finalized, compute() is called to compute the final
    localization performance.
    """

    def __init__(self, metadata, mask_root, dataset_name, split, cam_threshold_list,
                 iou_threshold_list, multi_contour_eval):
        self.metadata = metadata
        self.cam_threshold_list = cam_threshold_list
        self.iou_threshold_list = iou_threshold_list
        self.dataset_name = dataset_name
        self.split = split
        self.multi_contour_eval = multi_contour_eval
        self.resize_length = _RESIZE_LENGTH

        self.mask_root = mask_root

    def accumulate(self, scoremap, image_id, correct=None):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


class BoxEvaluator(LocalizationEvaluator):
    def __init__(self, **kwargs):
        super(BoxEvaluator, self).__init__(**kwargs)

        self.image_ids = get_image_ids(metadata=self.metadata)
        self.cnt = 0
        self.count_top_k = 0
        self.num_correct = \
            {iou_threshold: np.zeros(len(self.cam_threshold_list))
             for iou_threshold in self.iou_threshold_list}
        self.top_k_correct = {"TOP_1": np.zeros(len(self.cam_threshold_list)),
                              "TOP_5": np.zeros(len(self.cam_threshold_list)),
                              "GT_LOC": np.zeros(len(self.cam_threshold_list))}
        self.original_bboxes = get_bounding_boxes(self.metadata)
        self.image_sizes = get_image_sizes(self.metadata)
        self.gt_bboxes = self._load_resized_boxes(self.original_bboxes)

    def _load_resized_boxes(self, original_bboxes):
        resized_bbox = {image_id: [
            resize_bbox(bbox, self.image_sizes[image_id],
                        (self.resize_length, self.resize_length))
            for bbox in original_bboxes[image_id]]
            for image_id in self.image_ids}
        return resized_bbox

    def accumulate(self, scoremap, image_id, true_class, prediction, correct=None):
        """
        From a score map, a box is inferred (compute_bboxes_from_scoremaps).
        The box is compared against GT boxes. Count a scoremap as a correct
        prediction if the IOU against at least one box is greater than a certain
        threshold (_IOU_THRESHOLD).

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
        """
        if correct is not None and not correct:
            self.cnt += 1
            return
        boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
            scoremap=scoremap,
            scoremap_threshold_list=self.cam_threshold_list,
            multi_contour_eval=self.multi_contour_eval)

        boxes_at_thresholds = np.concatenate(boxes_at_thresholds, axis=0)

        multiple_iou = calculate_multiple_iou(
            np.array(boxes_at_thresholds),
            np.array(self.gt_bboxes[image_id]))

        idx = 0
        sliced_multiple_iou = []
        for nr_box in number_of_box_list:
            sliced_multiple_iou.append(
                max(multiple_iou.max(1)[idx:idx + nr_box]))
            idx += nr_box

        for _THRESHOLD in self.iou_threshold_list:
            correct_threshold_indices = \
                np.where(np.asarray(sliced_multiple_iou) >= (_THRESHOLD / 100))[0]
            self.num_correct[_THRESHOLD][correct_threshold_indices] += 1

        iou_threshold = 0.50
        correct_threshold_indices = \
            np.where(np.asarray(sliced_multiple_iou) >= iou_threshold)[0]

        self.top_k_correct["GT_LOC"][correct_threshold_indices] += 1

        if true_class in prediction[:1]:
            self.top_k_correct["TOP_1"][correct_threshold_indices] += 1
        if true_class in prediction[:5]:
            self.top_k_correct["TOP_5"][correct_threshold_indices] += 1

        self.cnt += 1

    def compute(self, cam_threshold=None):
        """
        Returns:
            max_localization_accuracy: float. The ratio of images where the
                for the final performance.
        """
        max_box_acc = []

        for _THRESHOLD in self.iou_threshold_list:
            localization_accuracies = self.num_correct[_THRESHOLD] * 100. / \
                                      float(self.cnt)
            if cam_threshold is not None:
                idx = self.cam_threshold_list.index(cam_threshold)
                max_box_acc.append(localization_accuracies[idx])
                # max_box_acc_thres.append(cam_threshold)
            else:
                max_index = np.argmax(localization_accuracies)
                max_box_acc.append(localization_accuracies.max())
                # max_box_acc_thres.append(self.cam_threshold_list[localization_accuracies.argmax()])
                print("the best threshold is: ", self.cam_threshold_list[max_index])

        top_k_loc_acc = {}
        for metric, correct_predictions in self.top_k_correct.items():
            localization_accuracies = correct_predictions * 100. / \
                                      float(self.cnt)
            if cam_threshold is not None:
                idx = self.cam_threshold_list.index(cam_threshold)
                top_k_loc_acc[metric] = localization_accuracies[idx]
            else:
                top_k_loc_acc[metric] = localization_accuracies.max()

        return max_box_acc, top_k_loc_acc


def load_mask_image(file_path, resize_size):
    """
    Args:
        file_path: string.
        resize_size: tuple of ints (height, width)
    Returns:
        mask: numpy.ndarray(dtype=numpy.float32, shape=(height, width))
    """
    mask = np.float32(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE))
    mask = cv2.resize(mask, resize_size, interpolation=cv2.INTER_NEAREST)
    return mask


def get_mask(mask_root, mask_paths, ignore_path):
    """
    Ignore mask is set as the ignore box region \setminus the ground truth
    foreground region.

    Args:
        mask_root: string.
        mask_paths: iterable of strings.
        ignore_path: string.

    Returns:
        mask: numpy.ndarray(size=(224, 224), dtype=np.uint8)
    """
    mask_all_instances = []
    for mask_path in mask_paths:
        mask_file = os.path.join(mask_root, mask_path)
        mask = load_mask_image(mask_file, (224, 224))
        mask_all_instances.append(mask > 0.5)
    mask_all_instances = np.stack(mask_all_instances, axis=0).any(axis=0)

    ignore_file = os.path.join(mask_root, ignore_path)
    ignore_box_mask = load_mask_image(ignore_file,
                                      (224, 224))
    ignore_box_mask = ignore_box_mask > 0.5

    ignore_mask = np.logical_and(ignore_box_mask,
                                 np.logical_not(mask_all_instances))

    if np.logical_and(ignore_mask, mask_all_instances).any():
        raise RuntimeError("Ignore and foreground masks intersect.")

    return (mask_all_instances.astype(np.uint8) +
            255 * ignore_mask.astype(np.uint8))


class MaskEvaluator(LocalizationEvaluator):
    def __init__(self, **kwargs):
        super(MaskEvaluator, self).__init__(**kwargs)

        if self.dataset_name != "OpenImages":
            raise ValueError("Mask evaluation must be performed on OpenImages.")

        self.mask_paths, self.ignore_paths = get_mask_paths(self.metadata)

        # cam_threshold_list is given as [0, bw, 2bw, ..., 1-bw]
        # Set bins as [0, bw), [bw, 2bw), ..., [1-bw, 1), [1, 2), [2, 3)
        self.num_bins = len(self.cam_threshold_list) + 2
        self.threshold_list_right_edge = np.append(self.cam_threshold_list,
                                                   [1.0, 2.0, 3.0])

        # Per-class score histograms
        # self.gt_true_score_hist_per_class = defaultdict(lambda: np.zeros(self.num_bins, dtype=np.float32))
        # self.gt_false_score_hist_per_class = defaultdict(lambda: np.zeros(self.num_bins, dtype=np.float32))


        self.gt_true_score_hist = np.zeros(self.num_bins, dtype=np.float32)
        self.gt_false_score_hist = np.zeros(self.num_bins, dtype=np.float32)


    def accumulate(self, scoremap, image_id):
        """
        Score histograms over the score map values at GT positive and negative
        pixels are computed.

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
        """
        check_scoremap_validity(scoremap)
        gt_mask = get_mask(self.mask_root,
                           self.mask_paths[image_id],
                           self.ignore_paths[image_id])

        gt_true_scores = scoremap[gt_mask == 1]
        gt_false_scores = scoremap[gt_mask == 0]

        # histograms in ascending order
        gt_true_hist, _ = np.histogram(gt_true_scores,
                                       bins=self.threshold_list_right_edge)
        self.gt_true_score_hist += gt_true_hist.astype(np.float32)

        gt_false_hist, _ = np.histogram(gt_false_scores,
                                        bins=self.threshold_list_right_edge)
        self.gt_false_score_hist += gt_false_hist.astype(np.float32)

    def accumulate_per_class(self, scoremap, image_id, class_label):
        """
        Accumulate score histograms per class over the score map values at
        GT positive and negative pixels.

        Args:
            scoremap: numpy.ndarray of shape (H, W), dtype=float
            image_id: str, unique image identifier
            class_label: str or int, the class for which this scoremap is evaluated
        """
        check_scoremap_validity(scoremap)

        # Load the GT mask (binary: 1 for object pixels, 0 for background)
        gt_mask = get_mask(self.mask_root,
                           self.mask_paths[image_id],
                           self.ignore_paths[image_id])

        # Separate true (foreground) and false (background) pixels
        gt_true_scores = scoremap[gt_mask == 1]
        gt_false_scores = scoremap[gt_mask == 0]

        # Histogram of values
        gt_true_hist, _ = np.histogram(gt_true_scores, bins=self.threshold_list_right_edge)
        gt_false_hist, _ = np.histogram(gt_false_scores, bins=self.threshold_list_right_edge)

        # Accumulate into per-class histograms
        self.gt_true_score_hist_per_class[class_label.item()] += gt_true_hist.astype(np.float32)
        self.gt_false_score_hist_per_class[class_label.item()] += gt_false_hist.astype(np.float32)

    def compute(self):
        """
        Arrays are arranged in the following convention (bin edges):

        gt_true_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        gt_false_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        tp, fn, tn, fp: >=2.0, >=1.0, ..., >=0.0

        Returns:
            auc: float. The area-under-curve of the precision-recall curve.
               Also known as average precision (AP).
        """
        num_gt_true = self.gt_true_score_hist.sum()
        tp = self.gt_true_score_hist[::-1].cumsum()
        fn = num_gt_true - tp

        num_gt_false = self.gt_false_score_hist.sum()
        fp = self.gt_false_score_hist[::-1].cumsum()
        tn = num_gt_false - fp

        if ((tp + fn) <= 0).all():
            raise RuntimeError("No positive ground truth in the eval set.")
        if ((tp + fp) <= 0).all():
            raise RuntimeError("No positive prediction in the eval set.")

        non_zero_indices = (tp + fp) != 0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        save_auc_curve(precision, recall, non_zero_indices)

        auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
        auc *= 100

        print("Mask AUC on split {}: {}".format(self.split, auc))
        return auc, None

    def compute_per_class(self):
        auc_per_class = {}
        for class_label in self.gt_true_score_hist_per_class.keys():
            gt_true_hist = self.gt_true_score_hist_per_class[class_label]
            gt_false_hist = self.gt_false_score_hist_per_class[class_label]

            num_gt_true = gt_true_hist.sum()
            tp = gt_true_hist[::-1].cumsum()
            fn = num_gt_true - tp

            num_gt_false = gt_false_hist.sum()
            fp = gt_false_hist[::-1].cumsum()
            tn = num_gt_false - fp

            if ((tp + fn) <= 0).all() or ((tp + fp) <= 0).all():
                auc = float('nan')  # or 0.0
            else:
                non_zero_indices = (tp + fp) != 0
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
                auc *= 100

            auc_per_class[class_label] = auc

        for cls, auc in auc_per_class.items():
            print(f"Mask AUC for class {cls} on split {self.split}: {auc:.2f}")

        # Sort by class id (keys)
        classes = sorted(auc_per_class.keys())
        ap_values = [auc_per_class[c] for c in classes]

        # --- Bar chart ---
        plt.figure(figsize=(12, 4))
        plt.bar(classes, ap_values, width=0.8)
        plt.xlabel("Class ID")
        plt.ylabel("Average Precision (AP)")
        plt.title("Per-Class AP Distribution")
        plt.tight_layout()
        plt.savefig("ap_bar_chart_open_images.png", dpi=300)  # save figure
        plt.close()

        # --- Histogram ---
        plt.figure(figsize=(6, 4))
        plt.hist(ap_values, bins=5, edgecolor='black')
        plt.xlabel("Average Precision (AP)")
        plt.ylabel("Number of Classes")
        plt.title("Distribution of Per-Class AP")
        plt.tight_layout()
        plt.savefig("ap_histogram_open_images.png", dpi=300)  # save figure
        plt.close()

        return auc_per_class, None

import matplotlib.pyplot as plt



def save_auc_curve(precision, recall, non_zero_indices,save_path="pr_curve.png"):


    # Filter valid entries
    precision = precision[non_zero_indices]
    recall = recall[non_zero_indices]

    # Plot
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, marker='o', linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)

    # Save the figure to file
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
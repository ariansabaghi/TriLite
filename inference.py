import cv2
import numpy as np
import torch
from tqdm import tqdm

from evaluation import BoxEvaluator, MaskEvaluator
from data_loaders import configure_metadata
from util import t2n

_RESIZE_LENGTH = 224


def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam


class CAMComputer(object):
    def __init__(self, model, loader, metadata_root, mask_root,
                 iou_threshold_list, dataset_name, split,
                 multi_contour_eval, cam_curve_interval=.001):
        self.model = model
        self.model.eval()
        self.loader = loader
        self.split = split
        self.dataset_name = dataset_name

        metadata = configure_metadata(metadata_root)
        cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))

        self.evaluator = {"OpenImages": MaskEvaluator,
                          "CUB": BoxEvaluator,
                          "ILSVRC": BoxEvaluator
                          }[dataset_name](metadata=metadata,
                                          mask_root=mask_root,
                                          dataset_name=dataset_name,
                                          split=split,
                                          cam_threshold_list=cam_threshold_list,
                                          iou_threshold_list=iou_threshold_list,
                                          multi_contour_eval=multi_contour_eval)

    def compute_and_evaluate_cams(self):

        for images, targets, image_ids in tqdm(self.loader):
            images = images.cuda()
            with torch.no_grad():
                logits, *_, localization_maps = self.model(images)

            topk_vals, predictions = torch.topk(logits, 5, dim=1)

            cams = t2n(localization_maps)

            for image, target, prediction, cam, image_id in zip(images.cpu().detach(), targets,
                                                                predictions.cpu().detach(), cams, image_ids):
                cam_resized = cv2.resize(cam, (_RESIZE_LENGTH, _RESIZE_LENGTH),
                                         interpolation=cv2.INTER_CUBIC)
                cam_normalized = normalize_scoremap(cam_resized)

                if self.dataset_name == 'OpenImages':
                    self.evaluator.accumulate(cam_normalized, image_id)
                else:
                    self.evaluator.accumulate(cam_normalized, image_id, target, prediction)

        return self.evaluator.compute()

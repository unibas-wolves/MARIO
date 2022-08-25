from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import (
    non_max_suppression, 
    scale_coords, 
    xyxy2xywh,
)

from strong_sort.strong_sort import StrongSORT


import torchvision.transforms as transforms
import torch

import numpy as np

def normalize(img):
    return img.to(torch.half) / 255

class NormalizeInRange(object):
    def __call__(self, img):
        return normalize(img)

transform = transforms.Compose([
    NormalizeInRange()
])

class ObjectTracker():

    def __init__(self, yolo_weights_path, strong_sort_weights, homography_handler, tracking_src, device):
        self.device = torch.device(device=device)

        self.model = DetectMultiBackend(
            yolo_weights_path, 
            device=self.device, 
            dnn=False, 
            data=None, 
            fp16=True
        )

        self.tracker = StrongSORT(
            strong_sort_weights,
            self.device,
            max_dist=0.2,
            max_iou_distance=0.7,
            max_age=30,
            n_init=3,
            nn_budget=100,
            mc_lambda=0.995,
            ema_alpha=0.9,
        )

        self.homography_handler = homography_handler

        self.dataset = LoadImages(
            tracking_src, 
            img_size=[640, 640], 
            stride=self.model.stride, 
            auto=self.model.pt
        )

    def get_dataset(self):
        return self.dataset

    def track(self, img, img_0):
        """
            Tracks objects positions.

            Params:
                * img (numpy.ndarray): input image.

            Returns:
                * (dict): tracking result containing IDs, classes,
                bounding boxes and perspective transformed bounding 
                boxes. 
        """ 
        preprocessed_img = transform(torch.from_numpy(img))

        input_img = torch.unsqueeze(preprocessed_img, dim=0)
        input_img = input_img.to(self.device)

        detection = self.model(input_img, augment=False, visualize=False)
        detection = non_max_suppression(
            detection, 
            conf_thres=0.5, 
            iou_thres=0.7,
            classes=None,
            agnostic=False, 
            max_det=20
        )
        detection = detection[0]

        res = {
            "ids"     : [],
            "classes" : [],
            "bboxes_o": [],
            "bboxes_t": []
        }
        if detection is not None and len(detection):
            detection[:, :4] = scale_coords(input_img.shape[2:], detection[:, :4], img_0.shape).round()

            xywhs       = xyxy2xywh(detection[:, 0:4])
            confidences = detection[:, 4]
            classes     = detection[:, 5]

            outputs = self.tracker.update(xywhs.cpu(), confidences.cpu(), classes.cpu(), img_0)

            ids      = []
            classes  = []
            bboxes_o = []
            for output in outputs:
                ids.append(int(output[4]))
                classes.append(int(output[5]))
                bboxes_o.append(output[0:4])
            bboxes_t = self._to_another_space(bboxes_o)

            res["ids"]      = ids
            res["classes"]  = classes
            res["bboxes_o"] = bboxes_o
            res["bboxes_t"] = bboxes_t
        return res

    def _to_another_space(self, bboxes_o):
        """
            Apply a perspective transformation.

            Params:
                * bboxes_o (list): bounding boxes.

            Returns:
                * (list): perspective transformed bounding boxes.
        """
        bboxes_t = []
        for bbox in bboxes_o:
            # Bounding box follow the format (x_min, y_min, x_max, y_max).
            x_min = bbox[0]
            x_max = bbox[2]
            y_max = bbox[3]
            # The cartesian origin point correspond to the the median of 
            # the bottom side of the bounding box.
            o = np.array([(x_min + x_max) * 0.5, y_max, 1])
            # (H_0, H_1, H_2)   (o_x)   (H_0 * x + H_1 * y + H_2)   (t_x)   
            # (H_3, H_4, H_5) * (o_y) = (H_3 * x + H_4 * y + H_5) = (t_y) 
            # (H_6, H_7, H_8)   (o_z)   (H_6 * x + H_7 * y + H_8)   (t_z)
            t = np.matmul(self.homography_handler.get_H(), o)
            t_x, t_y, t_z = t[0], t[1], t[2]
            # t is a vector in the homogenous space. We need to transform
            # its components to the cartesian space.
            p_x = int(t_x / t_z)
            p_y = int(t_y / t_z)
            bboxes_t.append(np.array([p_x, p_y]))
        return bboxes_t

if __name__ == "__main__":
    pass
        

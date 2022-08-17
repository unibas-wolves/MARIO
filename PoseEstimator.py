from pose_estimation.modules.keypoints import extract_keypoints, group_keypoints
from pose_estimation.models.with_mobilenet import PoseEstimationWithMobileNet
from pose_estimation.modules.load_state import load_state
from pose_estimation.modules.pose import Pose

import numpy as np

import torch

import cv2

import math

class PoseEstimator:

    def __init__(self, pe_checkpoint):
        if torch.cuda.is_available(): device = "cuda"
        else: device = "cpu"
        self.device = torch.device(device)

        self.net = PoseEstimationWithMobileNet().to(self.device)
        checkpoint = torch.load(pe_checkpoint, map_location=self.device)
        load_state(self.net, checkpoint)

        self.height_size    = 256
        self.stride         = 8
        self.upsample_ratio = 4
        self.num_keypoints  = Pose.num_kpts

    def estimate(self, img):
        self.net = self.net.eval()

        heatmaps, pafs, scale, pad = self._infer(img)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(self.num_keypoints):
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio - pad[0]) / scale

        current_poses, confidences = [], []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0: continue

            pose_keypoints = np.ones((self.num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(self.num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                confidences.append(all_keypoints[int(pose_entries[n][kpt_id]), 2])

            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if not len(current_poses): return None, None

        confidences = np.reshape(confidences[: 18], (18, 1))
        return current_poses[0], confidences

    def _infer(self, img, pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1 / 256)):
        height, _, _ = img.shape
        scale = self.height_size / height

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_img = self._normalize(scaled_img, img_mean, img_scale)
        min_dims = [self.height_size, max(scaled_img.shape[1], self.height_size)]
        padded_img, pad = self._pad_width(scaled_img, self.stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        tensor_img = tensor_img.to(self.device)

        stages_output = self.net(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=self.upsample_ratio, fy=self.upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=self.upsample_ratio, fy=self.upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad

    def _normalize(self, img, img_mean, img_scale):
        img = np.array(img, dtype=np.float32)
        img = (img - img_mean) * img_scale
        return img

    def _pad_width(self, img, stride, pad_value, min_dims):
        h, w, _ = img.shape
        h = min(min_dims[0], h)
        min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
        min_dims[1] = max(min_dims[1], w)
        min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
        pad = []
        pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
        pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
        pad.append(int(min_dims[0] - h - pad[0]))
        pad.append(int(min_dims[1] - w - pad[1]))
        padded_img = cv2.copyMakeBorder(
            img, 
            pad[0], 
            pad[2], 
            pad[1], 
            pad[3],
            cv2.BORDER_CONSTANT, 
            value=pad_value
        )
        return padded_img, pad
    
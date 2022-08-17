from pose_estimation.modules.keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import numpy as np
import torch
import cv2

from collections import Counter
import math
import time

class Game:

    POSE_ESTIMATION = True
    DRAW_POSES      = True

    def __init__(self, resource_manager, homography, tracker, pose_estimator, fall_detector):
        """
            {
                <player_id>: {
                    "team": a value in set {0, 1},
                    "jersey_number": player number,
                    "positions": {
                        "<t_1>": [<pos_x_1>, <pos_y_1>],
                        "<t_2>": [<pos_x_2>, <pos_y_2>],
                        ...                            ,
                        "<t_n>": [<pos_x_N>, <pos_y_N>]
                    },
                    "poses": list containing last 30 poses captured,
                    "last_pose": frame of the last pose captured
                }
            }
            N.B. As well as player's informations, ball's informations 
            are stored in this data structure.
        """
        self.players = {}

        # Modify RGB values accordingly to the colors of the teams.
        self.team_1_rgb = (128, 128, 128)
        self.team_2_rgb = (255,   0,   0)

        self.resource_manager = resource_manager
        self.homography       = homography
        self.tracker          = tracker
        self.pose_estimator   = pose_estimator
        self.fall_detector    = fall_detector

        self.is_initialization_done = False
        
        self.prev_frame_time = 0
        self.d = 0
        self.new_frame_time  = 0

    def loop(self):
        cv2.namedWindow("Tracking")
        cv2.namedWindow("Tracking - Radar 2D")
	
        
        cv2.namedWindow("Plan view 2D")

        dataset = self.tracker.get_dataset()

        for frame, (_, field_resized, field, _, _) in enumerate(dataset):  
            field_copy = field.copy()

            field_2d = self.resource_manager.get_homography_dst()
            plan_view = cv2.warpPerspective(
                field_copy, 
                self.homography.get_H(), 
                (field_2d.shape[1], field_2d.shape[0])
            )
            plan_view = cv2.resize(plan_view,(500,282))
            cv2.moveWindow("Plan view 2D", 1250, 500)
            cv2.imshow("Plan view 2D", plan_view)

            tracker_result = self.tracker.track(field_resized, field)

            # Initially the robots are disposed according to the scheme: 
            # 1     3     5     4     2
            #    2     4     5     3     1
            # |--TEAM #1--|  |--TEAM #2--|
            num_tracked_robots = sum([1 for id_ in tracker_result["ids"] if id_ < 11])
            if frame < 200 and num_tracked_robots == 10 and not self.is_initialization_done:
                bboxes_t_x = [bbox[0] for bbox in tracker_result["bboxes_t"]]

                ids_ordered = [tracker_result["ids"][i] for i in np.argsort(bboxes_t_x)]

                teams          = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
                jersey_numbers = [1, 2, 3, 4, 5, 5, 4, 3, 2, 1]
                for id, team, jersey_number in zip(ids_ordered, teams, jersey_numbers):
                  
                    if id not in self.players.keys():
                        self.players[id] = {
                            "team"         : team,
                            "jersey_number": jersey_number,
                            "positions"    : {},
                            "poses"        : [],
                            "last_pose"    : None
                        }
                    else:
                        self.players[id]["team"]          = team
                        self.players[id]["jersey_number"] = jersey_number

                self.is_initialization_done = True

            for i in range(len(tracker_result["ids"])):
                id     = tracker_result["ids"][i]
                clss   = tracker_result["classes"][i]
                bbox_o = tracker_result["bboxes_o"][i]
                bbox_t = tracker_result["bboxes_t"][i]

                if id not in self.players.keys():
                    self.players[id] = {
                        "team"         : None,
                        "jersey_number": None,
                        "positions"    : {},
                        "poses"        : [],
                        "last_pose"    : None
                    }

                action_name, action_perc, is_fallen = None, None, False

                if self.is_initialization_done:
                    if clss == 0:
                        if self.POSE_ESTIMATION:
                            x_min = int(bbox_o[0])
                            x_max = int(bbox_o[2])
                            y_min = int(bbox_o[1])
                            y_max = int(bbox_o[3])

                            robot = field_copy[y_min: y_max, x_min: x_max, :]

                            pose, confidences = self.pose_estimator.estimate(robot)

                            if pose is not None:
                                keypoints = pose.keypoints

                                for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
                                    kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
                                    global_kpt_a_id = keypoints[kpt_a_id, 0]
                                    if global_kpt_a_id != -1:
                                        x_a, y_a = keypoints[kpt_a_id]
                                        x_a += x_min
                                        y_a += y_min
                                        cv2.circle(
                                            field, 
                                            (int(x_a), int(y_a)), 
                                            3, 
                                            (255, 0, 255), 
                                            -1
                                        )
                                    kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
                                    global_kpt_b_id = keypoints[kpt_b_id, 0]
                                    if global_kpt_b_id != -1:
                                        x_b, y_b = keypoints[kpt_b_id]
                                        x_b += x_min
                                        y_b += y_min
                                        cv2.circle(
                                            field, 
                                            (int(x_b), int(y_b)), 
                                            3, 
                                            (255, 0, 255), 
                                            -1
                                        )
                                    if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                                        cv2.line(
                                            field, 
                                            (int(x_a), int(y_a)), 
                                            (int(x_b), int(y_b)), 
                                            (255, 0, 255), 
                                            1
                                        )

                                pose_with_confidences = np.concatenate([keypoints, confidences], axis=1)
                                # Remove nose, eyes and hears.
                                pose_with_confidences = np.concatenate(
                                    [
                                        pose_with_confidences[:1, :], 
                                        pose_with_confidences[2: -4, :]
                                    ], 
                                    axis=0
                                )

                                num_invalid_kpts = sum([1 for p in pose_with_confidences if p[0] == -1 or p[1] == -1])
                                perc_valid_kpts  = ((pose_with_confidences.shape[0] - num_invalid_kpts) / pose_with_confidences.shape[0]) * 100  
                                if perc_valid_kpts > 80: 
                                    poses = self.players[id]["poses"]

                                    if len(poses) > 29: poses = poses[1:]
                                    poses.append(pose_with_confidences)
                                    self.players[id]["poses"] = poses

                                    if len(poses) == 30:
                                        pts = None
                                        for p in poses:
                                            p = torch.Tensor(p).unsqueeze(dim=0)
                                            if pts is None: pts = p
                                            else:           pts = torch.cat((pts, p), dim=0)

                                        action_name, action_perc = self.fall_detector.detect(pts, robot.shape)
                                        if action_perc < 35: action_name, action_perc = None, None

                            if (action_name == "fallen" and action_perc > 50) or self.fall_detector.detect_with_bbox(bbox_o): is_fallen = True

                        if self.players[id]["team"] == None:
                            self._assign_team(field_copy, id, bbox_o)

                        if self.players[id]["team"] == 0:
                            color = self._rgb_to_bgr(self.team_1_rgb)
                        else:
                            color = self._rgb_to_bgr(self.team_2_rgb)
                    else: 
                        color = (0, 0, 0)
                else:
                    color = (0, 0, 0)

                cv2.rectangle(
                    field, 
                    (int(bbox_o[0]), int(bbox_o[1])), 
                    (int(bbox_o[2]), int(bbox_o[3])), 
                    color, 
                    2
                )
                if is_fallen: s = f"ID: {str(id)} - Fallen"
                else:         s = f"ID: {str(id)}"
                (w, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                half_w = int(w * 0.5)
                half_x = int((bbox_o[0] + bbox_o[2]) * 0.5)
                cv2.rectangle(
                    field, 
                    (half_x - half_w, int(bbox_o[1] - 20)), 
                    (half_x + half_w, int(bbox_o[1]     )), 
                    color, 
                    -1
                )
                text_color = (255, 255, 255)
                cv2.putText(
                    field, 
                    s,
                    (half_x - half_w, int(bbox_o[1] - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6,
                    text_color,
                    2
                )

                # Store player position at current frame.
                self.players[id]["positions"][frame] = [
                    int(bbox_t[0]), 
                    int(bbox_t[1])
                ]

                cv2.circle(
                    field_2d,
                    (bbox_t[0], bbox_t[1]),
                    5,
                    color,
                    -1
                )
                (w, _), _ = cv2.getTextSize(f"ID: {str(id)}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                half_w = int(w * 0.5)
                cv2.putText(
                    field_2d, 
                    f"ID: {str(id)}",
                    (bbox_t[0] - half_w, bbox_t[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5,
                    color,
                    2
                )
            field_2d = cv2.resize(field_2d,(500,282))
            cv2.moveWindow("Tracking - Radar 2D", 1250, 0)
            cv2.imshow("Tracking - Radar 2D", field_2d)
            self.d += 1
            cv2.imwrite("./track/"+str(self.d)+".jpg",field_2d)

            self.new_frame_time = time.time()

            fps = 1 / (self.new_frame_time - self.prev_frame_time)
            self.prev_frame_time = self.new_frame_time

            cv2.putText(
                field, 
                f"{fps:.2f}", 
                (7, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                2, 
                (100, 255, 0), 
                2, 
                cv2.LINE_AA
            )
            field = cv2.resize(field,(600,338))
            cv2.imshow("Tracking", field)  
        
            if cv2.waitKey(1) & 0xFF == ord("q"): break
        cv2.destroyAllWindows()

        if len(self.players.keys()) != 0: self._save_game_data()

    def _assign_team(self, img, id, bbox):
        """
            Assign team to the player.

            Args:
                img (nd.array): field image;
                id (int): player id;
                bbox (nd.array): bounding box of the player.
        """
        x_min = int(bbox[0])
        x_max = int(bbox[2])
        y_min = int(bbox[1])
        y_max = int(bbox[3])
        y_mid = int((y_min + y_max) * 0.5)

        upper_body_robot = img[y_min: y_mid, x_min: x_max, :]

        resized_img  = cv2.resize(upper_body_robot, (300, 300))
        rgb_img      = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        reshaped_img = rgb_img.reshape(-1, 3)

        clusterer = MiniBatchKMeans(n_clusters=5)

        labels = clusterer.fit_predict(reshaped_img)
        counts = Counter(labels)

        center_colors = clusterer.cluster_centers_

        yuv_colors = [self._rgb_to_yuv(center_colors[k]) for k in counts.keys()]

        min_distance_from_team_1 = min(
            [self._euclidean_distance(yuv, self._rgb_to_yuv(self.team_1_rgb)) for yuv in yuv_colors]
        )
        min_distance_from_team_2 = min(
            [self._euclidean_distance(yuv, self._rgb_to_yuv(self.team_2_rgb)) for yuv in yuv_colors]
        )

        if min_distance_from_team_1 < min_distance_from_team_2: team = 0
        else: team = 1

        self.players[id]["team"] = team

    def _euclidean_distance(self, vec_a, vec_b):
        """
            Computes the euclidean distance between two vectors.

            Args:
                vec_a (list, tuple or nd.array): vector A;
                vec_b (list, tuple or nd.array): vector B;

            Returns:
                (float): computed euclidean distance between vector A and vector B.

        """
        return math.sqrt(
              ((vec_b[0] - vec_a[0]) ** 2) 
            + ((vec_b[1] - vec_a[1]) ** 2)
            + ((vec_b[2] - vec_a[2]) ** 2)
        )

    def _rgb_to_bgr(self, rgb):
        """
            Converts RGB triplet to BGR triplet.

            Args:
                rgb (list, tuple or nd.array) RGB triplet.

            Returns:
                (tuple): BGR triplet.
        """
        return (rgb[2], rgb[1], rgb[0])

    def _rgb_to_yuv(self, rgb):
        """
            Converts RGB triplet to YUV triplet.

            Args:
                rgb (list, tuple or nd.array) RGB triplet.

            Returns:
                (tuple): YUV triplet.
        """
        y =  0.257 * rgb[0] + 0.504 * rgb[1] + 0.098 * rgb[2] +  16
        u = -0.148 * rgb[0] - 0.291 * rgb[1] + 0.439 * rgb[2] + 128
        v =  0.439 * rgb[0] - 0.368 * rgb[1] - 0.071 * rgb[2] + 128
        return (y, u, v)

    def _save_game_data(self):
        """
            Save game data.
        """
        data = []
        for player_id in self.players.keys():
            player_data = self.players[player_id]

            if player_data["team"] == None:
                player_team = -1
            else:
                player_team = player_data["team"]

            player_jersey_number = player_data["jersey_number"]

            positions = player_data["positions"]
            for t in positions.keys():
                player_position_x = positions[t][0]
                player_position_y = positions[t][1]

                data.append(
                    [player_id, player_team, player_jersey_number, t, player_position_x, player_position_y]
                )

        df = pd.DataFrame(data, columns=["id", "team", "jersey_number", "time", "x_pos", "y_pos"])
        df.to_csv("./game_data.csv", index=False)

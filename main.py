from ResourceManager import ResourceManager
from PoseEstimator import PoseEstimator
from ObjectTracker import ObjectTracker
from GameAnalyzer import GameAnalyzer
from Preparation import Preparation
from Homography import Homography
from Game import Game

import subprocess

if __name__ == "__main__":
    
    video_path = "./video/dataset1.mp4"

    rm_config = {
        "homography_src_path": "./data/image_BG/background.jpg",
        "homography_dst_path": None,
        "tracker_src_path"   : video_path
    }
    
    prep = Preparation(video_path)

    new_field = prep.is_new_field()
    if new_field: 
        rm_config["homography_dst_path"] = "./field_new.png"
    else:
        rm_config["homography_dst_path"] = "./field_old.png"

    rm = ResourceManager(rm_config)
    src = rm.get_homography_src()
    dst = rm.get_homography_dst()

    h = Homography(src, dst, Homography.H_FROM_DETECTION)

    ot = ObjectTracker(
        yolo_weights_path="./data/detectionCalciatori.pt", 
        strong_sort_weights="./data/osnet_x1_0_msmt17.pth", 
        homography_handler=h
    )

    pe = PoseEstimator("./data/pose_estimation.pth")

    game = Game(
        resource_manager=rm, 
        homography=h, 
        tracker=ot, 
        pose_estimator=pe
    )
    game.loop()

    game_analyzer = GameAnalyzer("./game_data.csv")
    game_analyzer.heatmap()

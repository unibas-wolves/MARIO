import cv2

class MissingOrUnsupportedException(BaseException):
    pass

class ResourceManager:

    def __init__(self, config):
        self.homography_src_path = config["homography_src_path"]
        self.homography_dst_path = config["homography_dst_path"]
        self.tracker_src_path    = config["tracker_src_path"]

    def get_homography_src(self):
        return cv2.imread(self.homography_src_path, -1).copy()

    def get_homography_dst(self):
        return cv2.imread(self.homography_dst_path, -1).copy()

    def get_tracker_src(self):
        return cv2.VideoCapture(self.tracker_src_path)

    

from fall_detector.ActionsEstLoader import TSSTG

class FallDetector:

    def __init__(self, weight_path, device):
        self.action_estimator = TSSTG(weight_path, device)

    def detect(self, pts, image_size):
        res = self.action_estimator.predict(pts, image_size)
        action_name = self.action_estimator.class_names[res[0].argmax()]
        action_perc = res[0].max() * 100
        return action_name, action_perc

    def detect_with_bbox(self, bbox):
        x_min = int(bbox[0])
        x_max = int(bbox[2])
        y_min = int(bbox[1])
        y_max = int(bbox[3])

        w, h = x_max - x_min, y_max - y_min
        
        if w > h * 1.25: return True
        return False
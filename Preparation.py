from ezprogress.progressbar import ProgressBar
import numpy as np
import cv2

from Singleton import Singleton
import subprocess
import shutil
import os

class Preparation:

    def __init__(self, gui=False):
        self.gui       = gui
        self.new_field = False
        
    def is_new_field(self):
        return self.new_field

    def image_subtraction(self):
        if not os.path.exists(os.path.join(os.getcwd(), "imbs-mt/images")):
            os.mkdir(os.path.join(os.getcwd(), "imbs-mt/images"))
        
        command = " ".join(
            [
                os.path.join(os.getcwd(), "imbs-mt/bin/imbs-mt"), 
                "-img", 
                os.path.join(os.getcwd(), "imbs-mt/images/30.jpg")
            ]
        )
        subprocess.run(command, shell=True)

        shutil.rmtree(os.path.join(os.getcwd(), "imbs-mt/images"))
        os.mkdir(os.path.join(os.getcwd(), "imbs-mt/images"))
        subprocess.run("python detectionT/detect.py", shell=True)

        source      = os.path.join(os.getcwd(), "data/image_BG/exp/labels/background.txt")
        destination = os.path.join(os.getcwd(), "data/image_BG/background.txt")
        shutil.copyfile(source, destination)

        source      = os.path.join(os.getcwd(), "data/image_BG/exp/background.jpg")
        destination = os.path.join(os.getcwd(), "data/image_BG/background_detection.jpg")
        shutil.copyfile(source, destination)

        shutil.rmtree(os.path.join(os.getcwd(), "data/image_BG/exp"))

        with open(os.path.join(os.getcwd(), "data/image_BG/background.txt"), "r") as f:
            lines = f.readlines()

        lines_count = sum([1 for l in lines if l[0] == "3"])
        if lines_count > 7: 
            self.new_field = True
        else:
            self.new_field = False
        return self.new_field

    def calibrate_video(self, is_already_calibrated):

        cap = cv2.VideoCapture(Singleton().getVideoPath())
        if not is_already_calibrated:

            npz_calib_file = np.load(Singleton().getCalibPath())

            mtx  = npz_calib_file["intrinsic_matrix"]
            dist = npz_calib_file["distCoeff"]
            v    = npz_calib_file["vect"]
            pb = ProgressBar(50, bar_length=50)
            pb.start()

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
             
            out = cv2.VideoWriter("./output.avi", fourcc, 20.0, (1280,  720))


        if not os.path.exists("./imbs-mt/images"):
            os.mkdir("./imbs-mt/images")

        idx = 0
        while cap.isOpened():
            return_value, frame = cap.read()
            if not return_value: break

            idx += 1

            if not is_already_calibrated:
                print("Video calibration ongoing...")
                frame = cv2.resize(frame, (1280, 720))
                h, w = frame.shape[:2]
                perc = (idx * 50) / frame_count
                pb.update(perc)

                newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

                mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
                dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

                dst_cropped = dst[v[0]: h - v[1], v[2]: w - v[3]]
                dst_cropped = cv2.resize(dst_cropped, (1280, 720))
                
                frame_to_save = dst_cropped
                out.write(frame_to_save)

                os.system("clear")
            else:
                frame_to_save = frame
        
            _, r = divmod(idx, 30)
            if r == 0 and idx < 20000: 
                cv2.imwrite(
                    os.path.join(os.getcwd(), 'imbs-mt/images', str(idx)+ ".jpg"),
                    frame_to_save
                )

            if cv2.waitKey(1) & 0xFF == ord("q"): break
        cap.release()
        
        if not is_already_calibrated:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pass

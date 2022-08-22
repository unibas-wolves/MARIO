import glob
import cv2 as cv
import numpy as np
import os
from ezprogress.progressbar import ProgressBar
import time

f = open("gui/calibration_path.txt","r")
npz_calib_file = np.load(f.readlines()[0])
f.close()
mtx = npz_calib_file['intrinsic_matrix']
dist = npz_calib_file['distCoeff']
v = npz_calib_file['vect']

f = open("gui/video_path.txt","r")
video_path = f.readlines()[0]
f.close()
cap = cv.VideoCapture(video_path)
totali =  int(cap.get(cv.CAP_PROP_FRAME_COUNT))

frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (1280,  720))
i = 1
pb = ProgressBar(50, bar_length=50)
pb.start()
if not os.path.exists("imbs-mt/images"):
        os.mkdir("imbs-mt/images")
while cap.isOpened():
        i += 1
        percentuale = (i*50)/totali
        print("Calibrazione video in corso ...: ")
        pb.update(percentuale)
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv.resize(frame,(1280,720))
        h,  w = frame.shape[:2]
        image_calibrated = cv.undistort(frame, mtx , dist, None)
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
        dst = cv.remap(frame, mapx, mapy, cv.INTER_LINEAR)
        dst_cropped = dst[v[0]:h-v[1], v[2]:w-v[3]]
        dst_cropped = cv.resize(dst_cropped,(1280,720))
        save_path = os.path.join(os.getcwd(),"imbs-mt/images","".join([str(i),".jpg"]))
        print(save_path)
        _, r = divmod(i, 30)
        if r == 0 and (percentuale < 20):
            cv.imwrite(os.getcwd() + "/imbs-mt/images/"+str(i)+".jpg",dst_cropped)
        out.write(dst_cropped)
        cv.waitKey(1)
        os.system('clear')
        if cv.waitKey(1) == ord('q'):
            break

cap.release()
out.release()
cv.destroyAllWindows()

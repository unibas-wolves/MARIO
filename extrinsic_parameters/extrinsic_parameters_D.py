import cv2
import numpy as np

img = cv2.imread("background.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


points=np.float32([[[4.5,0,0],[4.5,3,0],[4.5,6,0],[4.5,3.75,0],[4.5,2.25,0],[0.6,4.1,0],[0.6,1.9,0],[0,1.9,0],[0,4.1,0],[9,0,0],[9,1.9,0],[9,4.1,0],[0,5.76,0],[9,5.9,0]]])
objpoints = []
objpoints.append(points)
	
imgp = np.float32([[[614,70],[619,144],[635,407],[621,182],[617,118],[142,246],[269,142],[225,150],[87,253],[906,79],[985,125],[1127,216],[0,334],[1277,335]]])
imgpoints = []
imgpoints.append(imgp)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None,flags= cv2.CALIB_FIX_ASPECT_RATIO +cv2.CALIB_FIX_K1 + cv2.CALIB_CB_MARKER + cv2.CALIB_FIX_TAUX_TAUY +  cv2.CALIB_SAME_FOCAL_LENGTH  )

np.savez('calibration_old', distCoeff=dist, intrinsic_matrix=mtx)

# undistort
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# undistort
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
# crop the image# crop the image
dst_cropped = dst[70:h-300, 120:w-140]
cv2.imwrite("undistort.jpg",dst)
cv2.imwrite("undistort_cropped.jpg",dst_cropped)
cv2.imwrite("img.jpg",img)

while True:

	cv2.imwrite("undistort.jpg",dst)
	cv2.imwrite("undistort_cropped.jpg",dst_cropped)
	cv2.imwrite("img.jpg",img)
	cv2.imshow("undistort_cropped ",dst_cropped)
	cv2.imshow("img",img)
	
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
		
	

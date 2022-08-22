import cv2
import numpy as np

img = cv2.imread("background.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#[0.6,4.1,0],[0.6,1.9,0],[0,1.9,0],[0,4.1,0],[9,0,0],[9,1.9,0],[9,4.1,0],[0,6,0]]]
points=np.float32([[[4.5,0,0],[4.5,3,0],[4.5,6,0],[4.5,3.75,0],[4.5,2.25,0],[0,0,0],[0,6,0],[9,0,0],[9,1.9,0],[9,4.1,0],[9,5,0],[8.5,6,0]]])
objpoints = []
objpoints.append(points)
	
imgp = np.float32([[[694,169],[693,254],[684,523],[693,296],[694,222],[419,170],[39,441],[989,190],[1066,240],[1208,350],[1279,415],[1278,521]]])
imgpoints = []
imgpoints.append(imgp)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None,flags= cv2.CALIB_FIX_PRINCIPAL_POINT+cv2.CALIB_FIX_ASPECT_RATIO+cv2.CALIB_CB_NORMALIZE_IMAGE +cv2.CALIB_FIX_K1   )

np.savez('calibration_oldA', distCoeff=dist, intrinsic_matrix=mtx)

# undistort
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# undistort
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)

# crop the image# crop the image
dst_cropped = dst[150:h-180, 360:w-215]
cv2.imwrite("undistort.jpg",dst)
cv2.imwrite("undistort_cropped.jpg",dst_cropped)
cv2.imwrite("img.jpg",img)

while True:
	cv2.imshow("undistort_cropped ",dst_cropped)
	#cv2.imshow("img",img)
	
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
		
	

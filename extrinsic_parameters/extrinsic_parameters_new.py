import cv2
import numpy as np

img = cv2.imread("background_new.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#points=np.float32([[[4.5,0,0],[4.5,3,0],[4.5,6,0],[4.5,3.75,0],[4.5,2.25,0],[0.6,4.1,0],[0.6,1.9,0],[0,1.9,0],[0,4.1,0],[9,0,0],[9,1.9,0],[9,4.1,0],[9,5.05,0],[0,4.8,0]]])
points=np.float32([[[4.5,0,0],[4.5,3,0],[4.5,6,0],[4.5,3.75,0],[4.5,2.25,0],[9,0,0],[0,0,0],[0,1,0],[1.65,1,0],[4.5,1.3,0],[0,4.1,0],[9,1,0],[9,4.1,0],[1.65,5,0],[0.65,5.1,0],[0,0.2,0],[7.35,5,0]]])
objpoints = []
objpoints.append(points)
	
#imgp = np.float32([[[641,211],[665,288],[760,560],[679,331],[656,259],[142,356],[299,255],[253,257],[77,357],[923,227],[985,125],[1009,268],[1270,411],[0,407]]])
imgp = np.float32([[[641,211],[666,289],[762,605],[679,331],[656,259],[923,227],[335,217],[297,236],[423,233], [310,291], [79,357],[964,247],[117,356],[188,434],[39,431],[0,430],[1130,421]]])
imgpoints = []
imgpoints.append(imgp)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None,flags= cv2.CALIB_FIX_PRINCIPAL_POINT+cv2.CALIB_FIX_ASPECT_RATIO+cv2.CALIB_CB_NORMALIZE_IMAGE  )

np.savez('calibration_new', distCoeff=dist, intrinsic_matrix=mtx)

# undistort
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# undistort
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
# crop the image# crop the image
# crop the image# crop the image
dst_cropped = dst[195:h-90, 200:w-140]
cv2.imwrite("undistort_new.jpg",dst)
cv2.imwrite("undistort_cropped_new.jpg",dst_cropped)
cv2.imwrite("img_new.jpg",img)

cv2.imwrite("img.jpg",img)

while True:

	cv2.imshow("undistort.jpg",dst_cropped)
	cv2.imwrite("img.jpg",img)
	cv2.imshow("img",img)
	
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
		
	

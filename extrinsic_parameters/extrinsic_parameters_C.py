import cv2
import numpy as np

img = cv2.imread("backgroundC.jpg")
cv2.imwrite("backgroundC.jpg",cv2.resize(img,(1280,720)))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#points=np.float32([[[4.5,0,0],[4.5,3,0],[4.5,6,0],[4.5,3.75,0],[4.5,2.25,0],[0.6,4.1,0],[0.6,1.9,0],[0,1.9,0],[0,4.1,0],[9,0,0],[9,1.9,0],[9,4.1,0],[9,5.05,0],[0,4.8,0]]])
points=np.float32([[[4.5,0,0],[4.5,3,0],[4.5,6,0],[4.5,3.75,0],[4.5,2.25,0],[9,0,0],[0,0,0],[0,4.1,0],[9,4.1,0],[0.8,6,0]]])
objpoints = []
objpoints.append(points)
	
#imgp = np.float32([[[641,211],[665,288],[760,560],[679,331],[656,259],[142,356],[299,255],[253,257],[77,357],[923,227],[985,125],[1009,268],[1270,411],[0,407]]])
imgp = np.float32([[[643,156],[634,254],[608,610],[630,307],[636,219],[976,163],[327,176],[75,342],[1243,328],[3,531]]])
imgpoints = []
imgpoints.append(imgp)



ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None,flags= cv2.CALIB_FIX_PRINCIPAL_POINT+cv2.CALIB_FIX_ASPECT_RATIO+cv2.CALIB_CB_NORMALIZE_IMAGE)

np.savez('calibration_oldC', distCoeff=dist, intrinsic_matrix=mtx)

# undistort
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# undistort
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
# crop the image# crop the image
dst_cropped = dst[120:h-120, 100:w-70]
cv2.imwrite("undistortC.jpg",dst)
cv2.imwrite("undistort_croppedC.jpg",dst_cropped)
cv2.imwrite("imgC.jpg",img)

while True:

	cv2.imshow("undistort_cropped ",dst_cropped)

	
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
		
	


import pandas as pd
import numpy as np
import cv2

from sklearn.cluster import KMeans
import math
import os

class UnknowCalculationMethod(BaseException):
    pass

class Homography():

    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

        self.src_copy = self.src.copy()
        self.dst_copy = self.dst.copy()

        self.src_x, self.src_y = -1, -1
        self.dst_x, self.dst_y = -1, -1

        self.src_list = []
        self.dst_list = []

        self.drawing = False

        self.calibration_data = {}

        self.H = []
            

    def get_mode(self):
        return self.mode

    def get_calibration_data(self):
        return self.calibration_data

    def get_H(self):
        return self.H

    def set_H(self, H):
        self.H = H

    def normalize_points(self,points_virtual_pitch, points_real_pitch):

        def get_normalization_matrix(pts, name="A"):
            pts = np.array(pts).astype(np.float64)
            x_mean, y_mean = np.mean(pts, axis=0)
            var_x, var_y = np.var(pts, axis=0)

            s_x, s_y = np.sqrt(2 / var_x), np.sqrt(2 / var_y)

            print("Matrix: {4} : meanx {0}, meany {1}, varx {2}, vary {3}, sx {5}, sy {6} ".format(x_mean, y_mean, var_x,
                                                                                                   var_y, name, s_x, s_y))

            n = np.array([[s_x, 0, -s_x * x_mean], [0, s_y, -s_y * y_mean], [0, 0, 1]])
            # print(n)

            n_inv = np.array([[1. / s_x, 0, x_mean], [0, 1. / s_y, y_mean], [0, 0, 1]])
            return n.astype(np.float64), n_inv.astype(np.float64)

        ret_correspondences = []
        imp, objp = points_virtual_pitch, points_real_pitch
        N_x, N_x_inv = get_normalization_matrix(objp, "A")
        N_u, N_u_inv = get_normalization_matrix(imp, "B")
        # print(N_x)
        # print(N_u)
        # convert imp, objp to homogeneous
        # hom_imp = np.array([np.array([[each[0]], [each[1]], [1.0]]) for each in imp])
        # hom_objp = np.array([np.array([[each[0]], [each[1]], [1.0]]) for each in objp])
        hom_imp = np.array([[[each[0]], [each[1]], [1.0]] for each in imp])
        hom_objp = np.array([[[each[0]], [each[1]], [1.0]] for each in objp])

        normalized_hom_imp = hom_imp
        normalized_hom_objp = hom_objp

        for i in range(normalized_hom_objp.shape[0]):
            # 54 points. iterate one by onea
            # all points are homogeneous
            n_o = np.matmul(N_x, normalized_hom_objp[i])
            normalized_hom_objp[i] = n_o / n_o[-1]

            n_u = np.matmul(N_u, normalized_hom_imp[i])
            normalized_hom_imp[i] = n_u / n_u[-1]

        normalized_objp = normalized_hom_objp.reshape(normalized_hom_objp.shape[0], normalized_hom_objp.shape[1])
        normalized_imp = normalized_hom_imp.reshape(normalized_hom_imp.shape[0], normalized_hom_imp.shape[1])

        normalized_objp = normalized_objp[:, :-1]
        normalized_imp = normalized_imp[:, :-1]

        # print(normalized_imp)

        ret_correspondences = (imp, objp, normalized_imp, normalized_objp, N_u, N_x, N_u_inv, N_x_inv)

        return ret_correspondences

    def _from_detection(self):
        """
            Semi-automatic detection of the pairs of points to be used to 
            compute the homography matrix. Homography source points are 
            automatically detected, destination points has to be manually 
            selected by the user.
        """
        cv2.namedWindow("Homography - SRC")
        cv2.moveWindow("Homography - SRC",  80, 80)

        cv2.namedWindow("Homography - DST")
        cv2.moveWindow("Homography - DST", 780, 80)
        cv2.setMouseCallback("Homography - DST", self._select_points_dst)

        background = cv2.imread(os.path.join(os.getcwd() ,'data/image_BG/background.jpg'))
        dh, dw, _ = background.shape

        with open(os.path.join(os.getcwd() ,'data/image_BG/background.jpg'), "r") as f:
            data = f.readlines()
        
        for row in data:
            label, x, y, w, h = map(float, row.split(" "))
            if label == 3 or label == 2:
                l = int((x - w * 0.5) * dw)
                r = int((x + w * 0.5) * dw)
                t = int((y - h * 0.5) * dh)
                b = int((y + h * 0.5) * dh)

                cor_x = int((r + l) * 0.5)
                cor_y = int((b + t) * 0.5)

                self.src_list.append([cor_x, cor_y])
                cv2.circle(self.src_copy, (cor_x, cor_y), 0, (0, 0, 255), 10)

                while True:
                    cv2.imshow("Homography - SRC", self.src_copy)
                    cv2.imshow("Homography - DST", self.dst_copy)

                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("s"):
                        cv2.circle(self.src_copy, (cor_x     , cor_y     ), 5, (0, 255, 0), -1)
                        cv2.circle(self.dst_copy, (self.dst_x, self.dst_y), 5, (0, 255, 0), -1)

                        self.dst_list.append([self.dst_x, self.dst_y])

                        break
                    elif k == ord("q"):
                        os._exit(0)
        cv2.destroyAllWindows()

        corr = self.normalize_points(self.dst_list, self.src_list)
        self._compute_view_based_homography(corr)
     

    def _from_detection_automatically(self):
        """
            Automatic detection of the pairs of points to be used to 
            compute the homography matrix. 
        """
     

        background = cv2.imread(os.path.join(os.getcwd() ,'data/image_BG/background.jpg'))
        dh, dw, _ = background.shape

        with open(os.path.join(os.getcwd() ,'data/image_BG/background.jpg'), "r") as f:
            data = f.readlines()

        for row in data:
            label, x, y, w, h = map(float, row.split(" "))
            if label == 3 or label == 2:
                l = int((x - w * 0.5) * dw)
                r = int((x + w * 0.5) * dw)
                t = int((y - h * 0.5) * dh)
                b = int((y + h * 0.5) * dh)

                cor_x = int((r + l) * 0.5)
                cor_y = int((b + t) * 0.5)
                #print([cor_x, cor_y])
                self.src_list.append([cor_x, cor_y])
        self.src_list = sorted(self.src_list, key = lambda a: (a[0], a[1]))
        kmeans = KMeans(n_clusters=3, random_state=0).fit(self.src_list)
        #print(self.src_list)
        #print(kmeans.labels_)

        df_reali = pd.DataFrame();
        df_reali["Group"] = kmeans.labels_
        df_reali["Points_x"] = [self.src_list[index][0] for index in range(len(self.src_list))]
        df_reali["Points_y"] = [self.src_list[index][1] for index in range(len(self.src_list))]
        punti3d = []
        gruppi = df_reali.drop_duplicates(["Group"])["Group"]
        for gruppo in range(len(gruppi)):
                #print("CLUSTER N. " , gruppo)
                df_grupporeale= df_reali[df_reali.Group == gruppo].sort_values(by=["Points_y"])
                #print(df_grupporeale)
                x = (df_grupporeale.Points_x.tolist())
                y = (df_grupporeale.Points_y.tolist())
                for i in range(len(x)):
                    punti3d.append([x[i],y[i]])
         


        self.dst_list = [[4, 160], [4, 340],[511, 3], [511, 497],[1019,159],[1019,341],[4,3],[72,159],[71,339],[950,159],[950,341]] #campo vecchio T  and L intersection
        self.dst_list  = sorted(self.dst_list, key = lambda a: (a[0], a[1]))
        kmeans2 = KMeans(n_clusters=3, random_state=0).fit(self.dst_list)

        #print(kmeans2.labels_)

        df_ideale = pd.DataFrame();
        df_ideale["Group"] = kmeans2.labels_
        df_ideale["Points_x"] = [self.dst_list[index][0] for index in range(len(self.dst_list))]
        df_ideale["Points_y"] = [self.dst_list[index][1] for index in range(len(self.dst_list))]
        gruppi = df_ideale.drop_duplicates(["Group"])["Group"]
        x = []
        y = []
        punti2d = []
        for gruppo in range(len(gruppi)):
                #print("CLUSTER N. " , gruppo)
                df_gruppoideale = df_ideale[df_ideale.Group == gruppo].sort_values(by=["Points_y"])

                #print(df_gruppoideale)

                x_id = (df_gruppoideale.Points_x.tolist())
                y_id = (df_gruppoideale.Points_y.tolist())
                for i in range(len(x_id)):
                    punti2d.append([x_id[i],y_id[i]])
        print(punti3d)  
        print(punti2d)  
        """
        for index in range(len(punti3d)):
            print(punti3d[index])
            while True:
                cv2.imshow("Homography - SRC", self.src_copy)
                cv2.imshow("Homography - DST", self.dst_copy)
		    
                k = cv2.waitKey(1) & 0xFF
                if k == ord("s"):
                                cv2.circle(self.src_copy, (punti3d[index][0]     , punti3d[index][1] ), 5, (0, 255, 0), -1)
                                cv2.circle(self.dst_copy,  (punti2d[index][0]     , punti2d[index][1] ), 5, (0, 255, 0), -1)
                                break
                elif k == ord("q"):
                                os._exit(0)
                  


        cv2.destroyAllWindows()"""

        corr = self.normalize_points(punti2d,punti3d)
        self._compute_view_based_homography(corr)
            
    def _compute_view_based_homography(self, correspondence, reproj=True):
        image_points = correspondence[0]
        object_points = correspondence[1]
        normalized_image_points = correspondence[2]
        normalized_object_points = correspondence[3]
        N_u = correspondence[4]
        N_x = correspondence[5]
        N_u_inv = correspondence[6]
        N_x_inv = correspondence[7]

        N = len(image_points)
        print("Number of points in current view : ", N)

        M = np.zeros((2 * N, 9), dtype=np.float64)
        print("Shape of Matrix M : ", M.shape)

        print("N_model\n", N_x)
        print("N_observed\n", N_u)

        # create row wise allotment for each 0-2i rows
        # that means 2 rows..
        for i in range(N):
            X, Y = normalized_object_points[i]  # A
            u, v = normalized_image_points[i]  # B

            row_1 = np.array([-X, -Y, -1, 0, 0, 0, X * u, Y * u, u])
            row_2 = np.array([0, 0, 0, -X, -Y, -1, X * v, Y * v, v])
            M[2 * i] = row_1
            M[(2 * i) + 1] = row_2

            print("p_model {0} \t p_obs {1}".format((X, Y), (u, v)))

        # M.h  = 0 . solve system of linear equations using SVD
        u, s, vh = np.linalg.svd(M)
        print("Computing SVD of M")
        # print("U : Shape {0} : {1}".format(u.shape, u))
        # print("S : Shape {0} : {1}".format(s.shape, s))
        # print("V_t : Shape {0} : {1}".format(vh.shape, vh))
        # print(s, np.argmin(s))

        h_norm = vh[np.argmin(s)]
        h_norm = h_norm.reshape(3, 3)
        # print("Normalized Homography Matrix : \n" , h_norm)
        print(N_u_inv)
        print(N_x)
        # h = h_norm
        h = np.matmul(np.matmul(N_u_inv, h_norm), N_x)

        # if abs(h[2, 2]) > 10e-8:
        h = h[:, :] / h[2, 2]

        print("Homography for View : \n", h)

        if reproj:
            reproj_error = 0
            for i in range(len(image_points)):
                t1 = np.array([[object_points[i][0]], [object_points[i][1]], [1.0]])
                t = np.matmul(h, t1).reshape(1, 3)
                t = t / t[0][-1]
                formatstring = "Imp {0} | ObjP {1} | Tx {2}".format(image_points[i], object_points[i], t)
                print(formatstring)
                reproj_error += np.sum(np.abs(image_points[i] - t[0][:-1]))
            reproj_error = np.sqrt(reproj_error / N) / 100.0
            print("Reprojection error : ", reproj_error)

        self.H = h


    def _select_points_src(self, event, x, y, flags, params):
        """
            Callback function called when the user select a point
            on the homography source window.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.src_x, self.src_y = x, y
            cv2.circle(self.src_copy, (x, y), 5, (0, 0, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def _select_points_dst(self, event, x, y, flags, params):
        """
            Callback function called when the user select a point
            on the homography destination window.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.dst_x, self.dst_y = x, y
            cv2.circle(self.dst_copy, (x, y), 5, (0, 0, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False   

    def __str__(self):
        return f"H = {self.H}"

import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import math
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import openpyxl
np.set_printoptions(threshold=np.inf)   # 讓矩陣在terminal裡完全顯示
np.set_printoptions(linewidth=400)   # terminal 中可顯示文字的寬度
from numba import njit
from numba.typed import List

##### 匹配計算參數 #####
M = 10# 子集尺寸 2M+1 的M
C = 2  # 下一幀搜尋範圍的倍率
c = 0  # 判斷幀數
num = 0  # 計算迭代次數
gap = 8  # 用於還原三維形貌的解析度(每隔"gap-1"格就計算三維座標)

##### 子集匹配參數 #####
reference_subset = np.full(((2*M)+1, (2*M)+1), float(0), dtype=np.float64)    # 儲存第一幀左影像找到的參考子集
reference_subset_x = np.full(((2*M)+1, (2*M)+1), float(0), dtype=np.float64)    # 儲存每一幀參考子集中，各個點的x座標
reference_subset_y = np.full(((2*M)+1, (2*M)+1), float(0), dtype=np.float64)    # 儲存每一幀參考子集中，各個點的y座標
pointx, pointy = 0, 0    # 儲存每一參考子集的中心點
origin_pointx, origin_pointy = 0, 0    # 用來儲存第一幀我所選取的子集中心點
plate_center_x, plate_center_y = 0, 0    # 用來儲存圓盤中心點位置，目的是要找出圓盤中心點的三維座標

##### 位移量計算參數 #####
disp = 0
displacement = 0
displacement_round = 0
real_displacement = 0
reference_X, reference_Y, reference_Z = 0, 0, 0
deformed_X, deformed_Y, deformed_Z = 0, 0, 0
roi_x1, roi_y1, roi_x2, roi_y2 = 0,0,0,0
roi_h, roi_w = 0,0

##### ZNSSD計算參數 #####
fm_v = float(0)
gm_v = float(0)
delta_f_v = float(0)
delta_g_v = float(0)

##### 位移映射參數 #####
U = 0
V = 0
Ux = 0
Uy = 0
Vx = 0
Vy = 0

##### 一階梯度專用list ######
nabla_C_term_2 = np.full((6,1),float(0), dtype=np.float64)     # 一階梯度，U, V, Ux, Uy, Vx, Vy， 這會是Jacobian Matrix

##### 二階梯度專用0矩陣 ######
double_nabla_C_term_2 = np.full((6,6), float(0), dtype=np.float64)    # 二階梯度，這會是一個6x6的Hessian matrix

#####  迭代專用new_P  ######
new_P = np.full((6,1), float(0), dtype=np.float64)     # Newton-Raphson method迭代出來的值會放這

##### 校正棋盤格數 #####
chessboard_size = (4,6)

##### 用來記錄X軸每幀位移數據 #####
list_X = []
########################################################################################################
###################################           對相機進行校正            ###################################
########################################################################################################

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints_l = []
imgpoints_r = []

img_left = glob.glob(os.path.join("chessboard_stereo/left_1", "*.jpg"))
img_right = glob.glob(os.path.join("chessboard_stereo/right_1", "*.jpg"))

for i, fname in enumerate(img_left):
    img_l = cv2.imread(img_left[i])
    img_r = cv2.imread(img_right[i])

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size, None)

    objpoints.append(objp)

    if ret_l == True:
        rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        imgpoints_l.append(corners_l)
        # ret_1 = cv2.drawChessboardCorners(img_l, chessboard_size, corners_l, ret_l)
        # cv2.imshow(img_left[i], img_l)
        # cv2.waitKey(500)
    if ret_r == True:
        rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        imgpoints_r.append(corners_r)
        # ret_r = cv2.drawChessboardCorners(img_r, chessboard_size, corners_r, ret_r)
        # cv2.imshow(img_right[i], img_r)
        # cv2.waitKey(500)

rt_l, cameramatrix_l, dist_l, rvec_l, tvec_l = cv2.calibrateCamera(objpoints, imgpoints_l, gray_l.shape[::-1], None, None)
rt_r, cameramatrix_r, dist_r, rvec_r, tvec_r = cv2.calibrateCamera(objpoints, imgpoints_r, gray_l.shape[::-1], None, None)

mean_error = 0
for i in range(len(objpoints)):
    imgpointsl2, _ = cv2.projectPoints(objpoints[i], rvec_l[i], tvec_l[i], cameramatrix_l, dist_l)
    error = cv2.norm(imgpoints_l[i], imgpointsl2, cv2.NORM_L2)/len(imgpointsl2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )
mean_error_1 = 0
for i in range(len(objpoints)):
    imgpointsr2, _ = cv2.projectPoints(objpoints[i], rvec_r[i], tvec_r[i], cameramatrix_r, dist_r)
    error = cv2.norm(imgpoints_r[i], imgpointsr2, cv2.NORM_L2)/len(imgpointsr2)
    mean_error_1 += error
print( "total error: {}".format(mean_error/len(objpoints)) )

#flags = 0
#flags |= cv2.CALIB_FIX_INTRINSIC
# flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
##flags |= cv2.CALIB_USE_INTRINSIC_GUESS
##flags |= cv2.CALIB_FIX_FOCAL_LENGTH
# flags |= cv2.CALIB_FIX_ASPECT_RATIO
##flags |= cv2.CALIB_ZERO_TANGENT_DIST
# flags |= cv2.CALIB_RATIONAL_MODEL
# flags |= cv2.CALIB_SAME_FOCAL_LENGTH
# flags |= cv2.CALIB_FIX_K3
# flags |= cv2.CALIB_FIX_K4
# flags |= cv2.CALIB_FIX_K5

stereocalib_critetia = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6)
ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, cameramatrix_l, dist_l, cameramatrix_r, dist_r, gray_l.shape[::-1], criteria=stereocalib_critetia, flags=cv2.CALIB_FIX_INTRINSIC)

######## 顯示平移向量 ########
print(f"平移向量 = {T}")

######## 將3x3旋轉矩陣轉換為3x1旋轉向量(Rodrigue's vector)
RM = R
# 旋转矩阵到欧拉角(弧度值)
def rotateMatrixToEulerAngles(RM):
    theta_z = np.arctan2(RM[1, 0], RM[0, 0])
    theta_y = np.arctan2(-1 * RM[2, 0], np.sqrt(RM[2, 1] * RM[2, 1] + RM[2, 2] * RM[2, 2]))
    theta_x = np.arctan2(RM[2, 1], RM[2, 2])
    print(f"Euler angles(弧度值):\ntheta_x: {theta_x}\ntheta_y: {theta_y}\ntheta_z: {theta_z}")
    return theta_x, theta_y, theta_z

# 旋转矩阵到欧拉角(角度制)
def rotateMatrixToEulerAngles2(RM):
    theta_z = np.arctan2(RM[1, 0], RM[0, 0]) / np.pi * 180
    theta_y = np.arctan2(-1 * RM[2, 0], np.sqrt(RM[2, 1] * RM[2, 1] + RM[2, 2] * RM[2, 2])) / np.pi * 180
    theta_x = np.arctan2(RM[2, 1], RM[2, 2]) / np.pi * 180
    print(f"Euler angles(角度值):\ntheta_x: {theta_x}\ntheta_y: {theta_y}\ntheta_z: {theta_z}")
    return theta_x, theta_y, theta_z

rotateMatrixToEulerAngles(RM)
rotateMatrixToEulerAngles2(RM)

# 左邊相機的"相機矩陣內部參數"
fx = M1[0][0]
fy = M1[1][1]
ox = M1[0][2]
oy = M1[1][2]
print(f"cameramatrix_l = {cameramatrix_l}")
print(f"[左相機]fx, fy, ox, oy, k = {fx, fy, ox, oy, d1}")
print(f"[右相機]fx, fy, ox, oy, k = {M2[0][0], M2[1][1], M2[0][2], M2[1][2], d2}")
# 儲存baseline資訊
baseline = float(abs(T[0]))
print(f"基線距離= {baseline}")


#####################################################################################################################
######################################            選擇感興趣區域            ###########################################
#####################################################################################################################

####################### 在第一幀影像上選擇一量測點，並設定參考子集，再尋找左右影像相同點 #######################
def first_frame(left_rectified):
    global reference_subset
    global pointx
    global pointy
    origin_point = []  # 具有完整數值的座標
    point = []  # 整數座標
    img1_cvt = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2RGB)
    plt.imshow(img1_cvt)
    plt.title("Select measure point!")
    position = plt.ginput(1)
    for i in position:
        A = list(i)
        for j in A:
            B = round(j)  # 將座標四捨五入
            C = j
            point.append(B)
            origin_point.append(C)
    point1 = np.array(origin_point)
    print(f"The point you choose : {point1}")
    pointx = point1[0]  # 該點的x座標         # 參考子集的原始座標
    pointy = point1[1]  # 該點的y座標
    return pointx, pointy

def selectroi(right_rectified):     # 用來選擇要搜尋的感興趣區域(ROI)
    points_list = []
    cvt = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2RGB)
    plt.imshow(cvt)
    plt.title("Select ROI")
    points = plt.ginput(2)
    for i in points:
        A = list(i)
        for j in A:
            B = round(j)
            points_list.append(B)
    roi_x1 = int(points_list[0])
    roi_y1 = int(points_list[1])
    roi_x2 = int(points_list[2])
    roi_y2 = int(points_list[3])
    return roi_x1, roi_y1, roi_x2, roi_y2

#####################################################################################################################
#################################           每個像素的reference subset            #####################################
#####################################################################################################################
@njit
def reference_subset_(img1, reference_subset, x, y, M):
    point_x = x - M
    point_y = y - M
    for i in range((2*M)+1):
        for j in range((2*M)+1):
            reference_subset[i][j] = img1[point_y+i][point_x+j]
    return reference_subset

#####################################################################################################################
############################                       ZNSSD計算函式                      #################################
#####################################################################################################################
@njit
def fm(img, m):
    # global fm_v
    sum = 0
    cons = 1 / pow(((2 * m) - 1), 2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            sum += int(img[i][j])
    fm_v = cons * sum
    return fm_v


# gm : defomed subset image
@njit
def gm(img, m):
    # global gm_v
    sum = 0
    cons = 1 / pow(((2 * m) - 1), 2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            sum += int(img[i][j])
    gm_v = cons * sum
    return gm_v

@njit
def delta_f(img, fm_v):
    # global fm_v
    # global delta_f_v
    sum = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            sum += pow((img[i][j] - fm_v), 2)
    out = np.sqrt(sum)
    delta_f_v = out
    return delta_f_v

@njit
def delta_g(img, gm_v):
    # global gm_v
    # global delta_g_v
    sum = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            sum += pow((img[i][j] - gm_v), 2)
    out = np.sqrt(sum)
    delta_g_v = out
    return delta_g_v

@njit
def ZNSSD(img1, img2, fm_v, gm_v, delta_f_v, delta_g_v):
    znssd = 0
    # global fm_v
    # global gm_v
    # global delta_f_v
    # global delta_g_v
    for i in range(len(img1)):
        for j in range(len(img2)):
            znssd += pow(((img1[i][j] - fm_v) / delta_f_v) - ((img2[i][j] - gm_v) / delta_g_v), 2)
    # ZNSSD_data.append(znssd)
    return znssd

ZNSSD_data = List()
ZNSSD_data.append(1.0)
ZNSSD_data.pop(0)
@njit
def ZNSSD_correlation(right_image, reference_subset, X, Y, roi_w, M, ZNSSD_data):      # 用來做ZNSSD matching
    fm_v = fm(reference_subset, M)
    delta_f_v = delta_f(reference_subset, fm_v)
    for i in range(roi_w):
        if i <= roi_w - ((2 * M) + 1):
            target_subset = right_image[Y - M:Y + M + 1, (X + i):(X + i) + (2 * M) + 1]
            gm_v = gm(target_subset, M)
            delta_g_v = delta_g(target_subset, gm_v)
            znssd = ZNSSD(reference_subset, target_subset, fm_v, gm_v, delta_f_v, delta_g_v)
            ZNSSD_data.append(znssd)
    new_ZNSSD_data = [x for x in ZNSSD_data if np.isnan(x) == False]  # 其中計算出的值有nan值，因此使用 np.isnan() 將nan消除
    corresponding_x = ZNSSD_data.index(min(new_ZNSSD_data)) + M  # 顯示最小值在list中的位置 ; +M是為了算出中心點 ; 子集中心座標: (corresponding_x, pointy)
    ZNSSD_data.clear()

    right_x = X + corresponding_x  # roi左上角的x座標 + 找到的相同點x座標 = 右影像中相同點的x座標
    return right_x

@njit
def ZNSSD_correlation_for_left(search_size, reference_subset, search_size_h, search_size_w, M, ZNSSD_data):
    fm_v = fm(reference_subset, M)
    delta_f_v = delta_f(reference_subset, fm_v)
    for i in range(search_size_h):
        if i <= search_size_h - ((2 * M) + 1):
            for j in range(search_size_w):
                if j <= search_size_w - ((2 * M) + 1):
                    target_subset1 = search_size[i:i + (2 * M) + 1, j:j + (2 * M) + 1]
                    gm_v = gm(target_subset1, M)
                    delta_g_v = delta_g(target_subset1, gm_v)
                    znssd = ZNSSD(reference_subset, target_subset1, fm_v, gm_v, delta_f_v, delta_g_v)
                    ZNSSD_data.append(znssd)

    new_ZNSSD_data = [x for x in ZNSSD_data if np.isnan(x) == False]  # 其中計算出的值有nan值，因此使用 np.isnan() 將nan消除
    size = len(ZNSSD_data)
    position_1 = ZNSSD_data.index(min(new_ZNSSD_data))  # 顯示最小值在list中的位置
    ZNSSD_data.clear()
    return size, position_1

#####################################################################################################################
######################################              計算gray_level              ############################################      07/19 修改為正確aij 詳情:practice/aij_v3.py
#####################################################################################################################
@njit
def bicubic_spline_coefficient(aij, x, y):  # 用來計算某一亞像素點的灰階值  x : aij_delta_x    y : aij_delta_y
    gray_level = (aij[0] * (pow(x, 0)) * (pow(y, 0))) + (aij[1] * (pow(x, 1)) * (pow(y, 0))) + (
            aij[2] * (pow(x, 2)) * (pow(y, 0))) + (aij[3] * (pow(x, 3)) * (pow(y, 0))) + \
                 (aij[4] * (pow(x, 0)) * (pow(y, 1))) + (aij[5] * (pow(x, 1)) * (pow(y, 1))) + (
                         aij[6] * (pow(x, 2)) * (pow(y, 1))) + (aij[7] * (pow(x, 3)) * (pow(y, 1))) + \
                 (aij[8] * (pow(x, 0)) * (pow(y, 2))) + (aij[9] * (pow(x, 1)) * (pow(y, 2))) + (
                         aij[10] * (pow(x, 2)) * (pow(y, 2))) + (aij[11] * (pow(x, 3)) * (pow(y, 2))) + \
                 (aij[12] * (pow(x, 0)) * (pow(y, 3))) + (aij[13] * (pow(x, 1)) * (pow(y, 3))) + (
                         aij[14] * (pow(x, 2)) * (pow(y, 3))) + (aij[15] * (pow(x, 3)) * (pow(y, 3)))
    return gray_level

@njit
def fxi(image, x, y):
    value = (float(image[y][x + 1]) - float(image[y][x - 1])) / 2
    return value

@njit
def fyi(image, x, y):
    value = (float(image[y + 1][x]) - float(image[y - 1][x])) / 2
    return value

@njit
def fxyi(image, x, y):
    value = (float(image[y + 1][x + 1]) - float(image[y][x - 1]) - float(image[y - 1][x]) - float(image[y][x])) / 4
    return value

@njit
def calculate_f(image, point, x_list):
    num = 0
    for i in range(2):
        for j in range(2):
            x, y = point[0] + j, point[1] + i
            value = image[y][x]
            # x_list.append(value)
            x_list[num] = value
            num+=1

@njit
def calculate_fx(image, point, x_list):
    num = 4
    for i in range(2):
        for j in range(2):
            x, y = point[0] + j, point[1] + i
            value = fxi(image, x, y)
            # x_list.append(value)
            x_list[num] = value
            num+=1

@njit
def calculate_fy(image, point, x_list):
    num = 8
    for i in range(2):
        for j in range(2):
            x, y = point[0] + j, point[1] + i
            value = fyi(image, x, y)
            # x_list.append(value)
            x_list[num] = value
            num+=1

@njit
def calculate_fxy(image, point, x_list):
    num = 12
    for i in range(2):
        for j in range(2):
            x, y = point[0] + j, point[1] + i
            value = fxyi(image, x, y)
            # x_list.append(value)
            x_list[num] = value
            num+=1

nag_A = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0],
                  [-3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0],
                  [9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1],
                  [-6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1],
                  [2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                  [-6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1],
                  [4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1]], dtype=np.float64)
@njit
def aij_gray_level(image, point):
    x_list = np.zeros((16,), dtype=np.float64)
    floor_x, floor_y = math.floor(point[0]), math.floor(point[1])
    delta_x, delta_y = abs(floor_x-point[0]), abs(floor_y-point[1])
    calculate_f(image, (floor_x,floor_y), x_list)
    calculate_fx(image, (floor_x,floor_y), x_list)
    calculate_fy(image, (floor_x,floor_y), x_list)
    calculate_fxy(image, (floor_x,floor_y), x_list)
    # matrix_x = np.array(x_list)
    aij = np.dot(nag_A, x_list)
    gray_level = bicubic_spline_coefficient(aij, delta_x, delta_y)
    # x_list.clear()

    if gray_level <= 0:
        return 0, aij, delta_x, delta_y
    elif gray_level >= 255:
        return 255, aij, delta_x, delta_y
    else:
        return gray_level, aij, delta_x, delta_y

 #####################################################################################################################
#########################################          shape function          #########################################     會放在def aij matrix裡面，也會獨立使用
#####################################################################################################################
@njit
def shape_function(reference_subset_point, reference_subset_center, U, V, Ux, Uy, Vx, Vy):
    delta_x = reference_subset_point[0] - reference_subset_center[0]
    delta_y = reference_subset_point[1] - reference_subset_center[1]
    deformed_subset_point_x = reference_subset_center[0] + delta_x + U + (Ux*delta_x) + (Uy*delta_y)
    deformed_subset_point_y = reference_subset_center[1] + delta_y + V + (Vx*delta_x) + (Vy*delta_y)

    return deformed_subset_point_x, deformed_subset_point_y


#####################################################################################################################
#######################################          計算aij matrix         #############################################     會放在def newton raphson裡面
#####################################################################################################################
@njit
def aij_matrix(right_rectified_gray, reference_subset_center, M, U, V, Ux, Uy, Vx, Vy):
    # 建立四個零矩陣，分別是 graylevel, aij, pointx, pointy
    num = 0
    deformed_subset_graylevel_matrix = np.full(((2 * M) + 1, (2 * M) + 1), float(0), dtype=np.float64)    #  07/17加上float
    deformed_subset_aij_list = np.full(( ((2*M)+1)*((2*M)+1), 16 ), float(0), dtype=np.float64)
    deformed_subset_delta_x_matrix = np.full(((2 * M) + 1, (2 * M) + 1), float(0), dtype=np.float64)
    deformed_subset_delta_y_matrix = np.full(((2 * M) + 1, (2 * M) + 1), float(0), dtype=np.float64)
    original_point_x = reference_subset_center[0] - M
    original_point_y = reference_subset_center[1] - M
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            reference_subset_point_y = original_point_y + i
            # print(f'reference_subset_point_x, reference_subset_point_y = {reference_subset_point_x, reference_subset_point_y}')
            # 利用參考子集的座標點P，去推算參考子集的座標點P'，接著利用P'的周圍16個點計算aij，最後再將aij和座標點P'代入Bicubic spline interpolation，來計算P'的灰階值。
            deformed_subset_point_x, deformed_subset_point_y = shape_function(reference_subset_point=(reference_subset_point_x, reference_subset_point_y), reference_subset_center=(reference_subset_center[0], reference_subset_center[1]), U=U, V=V, Ux=Ux, Uy=Uy, Vx=Vx, Vy=Vy)
            # print(f'deformed_subset_point_x, deformed_subset_point_y = {deformed_subset_point_x, deformed_subset_point_y}')
            deformed_subset_point_graylevel, aij_matrix, delta_x, delta_y = aij_gray_level(right_rectified_gray, (deformed_subset_point_x,deformed_subset_point_y))
            deformed_subset_graylevel_matrix[i][j] = deformed_subset_point_graylevel
            # deformed_subset_aij_list.append(aij_matrix)  # list裡總共會有((2M)+1))*((2M)+1)組aij
            for k in range(len(aij_matrix)):
                deformed_subset_aij_list[num][k] = aij_matrix[k]
            num+=1
            deformed_subset_delta_x_matrix[i][j] = delta_x
            deformed_subset_delta_y_matrix[i][j] = delta_y
    return deformed_subset_delta_x_matrix, deformed_subset_delta_y_matrix, deformed_subset_aij_list, deformed_subset_graylevel_matrix


#####################################################################################################################
######################################     Bicubic spline coefficient 偏微分    ######################################
####################################################################################################################
@njit
def partial_bicubic_spline(aij, x, y, partial_UorV):  # 將 aij, 亞像素點x,y座標 代入以下經過偏微分的bicubic spline coefficient中，之後會放進nabla_C中
    if partial_UorV == "partial_U":  # 將bicubic spline coefficient對"x"偏微分
        coefficient_partial_U = aij[1] + (2*aij[2]*x) + (3*aij[3]*(pow(x,2))) + (aij[5]*y) + (2*aij[6]*x*y) + (3*aij[7]*(pow(x,2))*y) + \
                                (aij[9]*(pow(y,2))) + (2*aij[10]*x*(pow(y,2))) + (3*aij[11]*(pow(x,2))*(pow(y,2))) + (aij[13]*(pow(y,3))) + (2*aij[14]*x*(pow(y,3))) + (3*aij[15]*(pow(x,2))*(pow(y,3)))
        return coefficient_partial_U
    if partial_UorV == "partial_V":  # 將bicubic spline coefficient對"y"偏微分
        coefficient_partial_y = aij[4] + (aij[5]*x) + (aij[6]*(pow(x,2))) + (aij[7]*(pow(x,3))) + (2*aij[8]*y) + (2*aij[9]*x*y) + (2*aij[10]*(pow(x,2))*y) + (2*aij[11]*(pow(x,3))*y) + \
                                (3*aij[12]*(pow(y,2))) + (3*aij[13]*x*pow(y,2)) + (3*aij[14]*(pow(x,2))*(pow(y,2))) + (3*aij[15]*(pow(x,3))*(pow(y,2)))
        return coefficient_partial_y


#####################################################################################################################
#############################     利用Bicubic Spline Interpolation計算參考子集的所有像素點    #############################      這用在計算第一幀左影像的參考子集
#####################################################################################################################
@njit
def bicubic_reference_subset(left_rectified_gray, reference_subset_center, M, reference_subset):
    subset_origin_x = reference_subset_center[0] - M
    subset_origin_y = reference_subset_center[1] - M
    for i in range((2*M)+1):
        for j in range((2*M)+1):
            pointx = subset_origin_x + j
            pointy = subset_origin_y + i
            gray_level, aij_matrix, delta_x, delta_y = aij_gray_level(left_rectified_gray, (pointx,pointy))
            reference_subset[i][j] = gray_level
    return reference_subset


#####################################################################################################################
#############################     利用Bicubic Spline Interpolation計算參考子集的所有像素點    #############################      這用在計算第二幀的左影像的參考子集
#####################################################################################################################
# def bicubic_deformed_subset(left_rectified_gray,  M, reference_subset_center, reference_subset_x, reference_subset_y, U, V, Ux, Uy, Vx, Vy):
#     subset = np.full(((2*M)+1, (2*M)+1), 0)
#     for i in range((2*M)+1):
#         for j in range((2*M)+1):
#             x, y = shape_function((reference_subset_x[i][j],reference_subset_y[i][j]), reference_subset_center, U, V, Ux, Uy, Vx, Vy)
#             aij1, aij_delta_x, aij_delta_y = aij(deformed_subset_point=(x,y), right_rectified_gray=left_rectified_gray)
#             subset[i][j] = bicubic_spline_coefficient(aij1, aij_delta_x, aij_delta_y)
#             reference_subset_x[i][j], reference_subset_y[i][j] = x, y
#     return subset, reference_subset_x, reference_subset_y


#####################################################################################################################
###############################     計算 nabla_C (一階梯度 correlation coefficient)    ################################      會放在def newton raphson裡面
#####################################################################################################################
@njit
def nabla_C_first_term(reference_subset, M):     # 先計算nabla_C的第一個項式 : -2 / sigma(g^2(Sp))
    reference_subset_sum = 0
    for i in range((2*M)+1):
        for j in range((2*M)+1):
           reference_subset_sum += pow(float(reference_subset[i][j]),2)
    term_1 = -2 / reference_subset_sum

    return term_1
@njit
def nabla_C_second_term_U(reference_subset, reference_subset_center, M, deformed_subset_aij_list, deformed_subset_graylevel_zero_matrix, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, nabla_C_term_2):
    nabla_C_term_2_U = 0
    a = 0
    for i in range((2*M)+1):
        for j in range((2*M)+1):
            Gsp = reference_subset[i][j]
            Hsp = deformed_subset_graylevel_zero_matrix[i][j]
            Hsp_U = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            nabla_C_term_2_U += (Gsp-Hsp)*Hsp_U
            a += 1
    nabla_C_term_2[0] = nabla_C_term_2_U  # 此為一階梯度correlation coefficient算出的Jacobian matrix內的第一項 (partial_S / partial_U)
@njit
def nabla_C_second_term_V(reference_subset, reference_subset_center, M, deformed_subset_aij_list, deformed_subset_graylevel_zero_matrix, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, nabla_C_term_2):     # 一階梯度中，對V偏微分的函式
    nabla_C_term_2_V = 0
    a = 0
    for i in range((2*M)+1):
        for j in range((2 * M) + 1):
            # Gsp:參考子集中的每一個點 ， Hsp:目標子集中的每一個點
            Gsp = reference_subset[i][j]
            Hsp = deformed_subset_graylevel_zero_matrix[i][j]
            Hsp_V = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            nabla_C_term_2_V += (Gsp-Hsp)*Hsp_V
            a += 1
    nabla_C_term_2[1] = nabla_C_term_2_V
@njit
def nabla_C_second_term_Ux(reference_subset, reference_subset_center, M, deformed_subset_aij_list, deformed_subset_graylevel_zero_matrix, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, nabla_C_term_2):     # 一階梯度中，對Ux偏微分的函式
    nabla_C_term_2_Ux = 0
    original_x = reference_subset_center[0] - M
    a = 0
    for i in range((2*M)+1):
        for j in range((2*M)+1):
            reference_subset_point_x = original_x + j
            # Gsp:參考子集中的每一個點 ， Hsp:目標子集中的每一個點
            Gsp = reference_subset[i][j]
            Hsp = deformed_subset_graylevel_zero_matrix[i][j]
            Hsp_U = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            delta_x = reference_subset_point_x - reference_subset_center[0]
            nabla_C_term_2_Ux += (Gsp-Hsp)*(Hsp_U*delta_x)
            a += 1
    nabla_C_term_2[2] = nabla_C_term_2_Ux
@njit
def nabla_C_second_term_Uy(reference_subset, reference_subset_center, M, deformed_subset_aij_list, deformed_subset_graylevel_zero_matrix, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, nabla_C_term_2):     # 一階梯度中，對Uy偏微分的函式
    nabla_C_term_2_Uy = 0
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2*M)+1):
        for j in range((2*M)+1):
            reference_subset_point_y = original_point_y + i
            # Gsp:參考子集中的每一個點 ， Hsp:目標子集中的每一個點
            Gsp = reference_subset[i][j]
            Hsp = deformed_subset_graylevel_zero_matrix[i][j]
            Hsp_U = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            delta_y = reference_subset_point_y - reference_subset_center[1]
            nabla_C_term_2_Uy += (Gsp-Hsp)*(Hsp_U*delta_y)
            a += 1
    nabla_C_term_2[3] = nabla_C_term_2_Uy
@njit
def nabla_C_second_term_Vx(reference_subset, reference_subset_center, M, deformed_subset_aij_list, deformed_subset_graylevel_zero_matrix, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, nabla_C_term_2):     # 一階梯度中，對Vx偏微分的函式
    nabla_C_term_2_Vx = 0
    original_point_x = reference_subset_center[0] - M
    a = 0
    for i in range((2*M)+1):
        for j in range((2*M)+1):
            reference_subset_point_x = original_point_x + j
            # Gsp:參考子集中的每一個點 ， Hsp:目標子集中的每一個點
            Gsp = reference_subset[i][j]
            Hsp = deformed_subset_graylevel_zero_matrix[i][j]
            Hsp_V = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_x = reference_subset_point_x - reference_subset_center[0]
            nabla_C_term_2_Vx += (Gsp-Hsp)*(Hsp_V*delta_x)
            a += 1
    nabla_C_term_2[4] = nabla_C_term_2_Vx
@njit
def nabla_C_second_term_Vy(reference_subset, reference_subset_center, M, deformed_subset_aij_list, deformed_subset_graylevel_zero_matrix, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, nabla_C_term_2):     # 一階梯度中，對Vy偏微分的函式
    nabla_C_term_2_Vy = 0
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2*M)+1):
        for j in range((2*M)+1):
            reference_subset_point_y = original_point_y + i
            # Gsp:參考子集中的每一個點 ， Hsp:目標子集中的每一個點
            Gsp = reference_subset[i][j]
            Hsp = deformed_subset_graylevel_zero_matrix[i][j]
            Hsp_V = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_y = reference_subset_point_y - reference_subset_center[1]
            nabla_C_term_2_Vy += (Gsp-Hsp)*(Hsp_V*delta_y)
            a += 1
    nabla_C_term_2[5] = nabla_C_term_2_Vy


#####################################################################################################################
##############################   計算 double_nabla_C (二階梯度 correlation coefficient)  ##############################  會放在def newton raphson裡面
#####################################################################################################################
@njit
def double_nabla_C_first_term(reference_subset, M):
    reference_subset_sum = 0
    for i in range((2*M)+1):
        for j in range((2*M)+1):
            reference_subset_sum += pow(reference_subset[i][j], 2)
    term_1 = 2 / reference_subset_sum

    return term_1
@njit
def double_nabla_C_second_term_U_U(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_U_U = 0
    a = 0
    for i in range((2*M)+1):
        for j in range((2*M)+1):
            # 計算 Pi = U , Pj = U 偏微分
            Hsp_PiU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            double_nabla_C_term_2_U_U += (Hsp_PiU*Hsp_PiU)
            a += 1
    double_nabla_C_term_2[0][0] = double_nabla_C_term_2_U_U
@njit
def double_nabla_C_second_term_U_V(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_U_V = 0
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            # 計算 Pi = U , Pj = V 偏微分
            Hsp_PiU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            Hsp_PjV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            double_nabla_C_term_2_U_V += (Hsp_PiU * Hsp_PjV)
            a += 1
    double_nabla_C_term_2[0][1] = double_nabla_C_term_2_U_V
@njit
def double_nabla_C_second_term_U_Ux(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_U_Ux = 0
    original_point_x = reference_subset_center[0] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            # 計算 Pi = U , Pj = Ux 偏微分
            Hsp_PiU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            delta_x = reference_subset_point_x - reference_subset_center[0]
            double_nabla_C_term_2_U_Ux += (Hsp_PiU * (Hsp_PiU*delta_x))
            a += 1
    double_nabla_C_term_2[0][2] = double_nabla_C_term_2_U_Ux
@njit
def double_nabla_C_second_term_U_Uy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_U_Uy = 0
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_y = original_point_y + i
            # 計算 Pi = U , Pj = Uy 偏微分
            Hsp_PiU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            delta_y = reference_subset_point_y - reference_subset_center[1]
            double_nabla_C_term_2_U_Uy += (Hsp_PiU * (Hsp_PiU*delta_y))
            a += 1
    double_nabla_C_term_2[0][3] = double_nabla_C_term_2_U_Uy
@njit
def double_nabla_C_second_term_U_Vx(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_U_Vx = 0
    original_point_x = reference_subset_center[0] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            # 計算 Pi = U , Pj = Vx 偏微分
            Hsp_PiU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            Hsp_PjV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_x = reference_subset_point_x - reference_subset_center[0]
            double_nabla_C_term_2_U_Vx += (Hsp_PiU * (Hsp_PjV*delta_x))
            a += 1
    double_nabla_C_term_2[0][4] = double_nabla_C_term_2_U_Vx
@njit
def double_nabla_C_second_term_U_Vy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_U_Vy = 0
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_y = original_point_y + i
            # 計算 Pi = U , Pj = Vy 偏微分
            Hsp_PiU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            Hsp_PjV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_y = reference_subset_point_y - reference_subset_center[1]
            double_nabla_C_term_2_U_Vy += (Hsp_PiU * (Hsp_PjV * delta_y))
            a += 1
    double_nabla_C_term_2[0][5] = double_nabla_C_term_2_U_Vy
@njit
def double_nabla_C_second_term_V_U(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_V_U = 0
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            # 計算 Pi = V , Pj = U 偏微分
            Hsp_PiV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            Hsp_PjU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            double_nabla_C_term_2_V_U += (Hsp_PiV * Hsp_PjU)
            a += 1
    double_nabla_C_term_2[1][0] = double_nabla_C_term_2_V_U
@njit
def double_nabla_C_second_term_V_V(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_V_V = 0
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            # 計算 Pi = V , Pj = V 偏微分
            Hsp_PiV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            double_nabla_C_term_2_V_V += (Hsp_PiV * Hsp_PiV)
            a += 1
    double_nabla_C_term_2[1][1] = double_nabla_C_term_2_V_V
@njit
def double_nabla_C_second_term_V_Ux(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_V_Ux = 0
    original_point_x = reference_subset_center[0] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            # 計算 Pi = V , Pj = Ux 偏微分
            Hsp_PiV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            Hsp_PjU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            delta_x = reference_subset_point_x - reference_subset_center[0]
            double_nabla_C_term_2_V_Ux += (Hsp_PiV * (Hsp_PjU*delta_x))
            a += 1
    double_nabla_C_term_2[1][2] = double_nabla_C_term_2_V_Ux
@njit
def double_nabla_C_second_term_V_Uy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_V_Uy = 0
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_y = original_point_y + i
            # 計算 Pi = V , Pj = Uy 偏微分
            Hsp_PiV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            Hsp_PjU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            delta_y = reference_subset_point_y - reference_subset_center[1]
            double_nabla_C_term_2_V_Uy += (Hsp_PiV * (Hsp_PjU * delta_y))
            a += 1
    double_nabla_C_term_2[1][3] = double_nabla_C_term_2_V_Uy
@njit
def double_nabla_C_second_term_V_Vx(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_V_Vx = 0
    original_point_x = reference_subset_center[0] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            # 計算 Pi = V , Pj = Vx 偏微分
            Hsp_PiV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_x = reference_subset_point_x - reference_subset_center[0]
            double_nabla_C_term_2_V_Vx += (Hsp_PiV * (Hsp_PiV * delta_x))
            a += 1
    double_nabla_C_term_2[1][4] = double_nabla_C_term_2_V_Vx
@njit
def double_nabla_C_second_term_V_Vy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_V_Vy = 0
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_y = original_point_y + i
            # 計算 Pi = V , Pj = Vy 偏微分
            Hsp_PiV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_y = reference_subset_point_y - reference_subset_center[1]
            double_nabla_C_term_2_V_Vy += (Hsp_PiV * (Hsp_PiV * delta_y))
            a += 1
    double_nabla_C_term_2[1][5] = double_nabla_C_term_2_V_Vy
@njit
def double_nabla_C_second_term_Ux_U(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Ux_U = 0
    original_point_x = reference_subset_center[0] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            # 計算 Pi = Ux , Pj = U 偏微分
            Hsp_PiU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            delta_x = reference_subset_point_x - reference_subset_center[0]
            double_nabla_C_term_2_Ux_U += ((Hsp_PiU*delta_x) * Hsp_PiU)
            a += 1
    double_nabla_C_term_2[2][0] = double_nabla_C_term_2_Ux_U
@njit
def double_nabla_C_second_term_Ux_V(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Ux_V = 0
    original_point_x = reference_subset_center[0] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            # 計算 Pi = Ux , Pj = V 偏微分
            Hsp_PiU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            delta_x = reference_subset_point_x - reference_subset_center[0]
            Hsp_PjU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            double_nabla_C_term_2_Ux_V += ((Hsp_PiU * delta_x) * Hsp_PjU)
            a += 1
    double_nabla_C_term_2[2][1] = double_nabla_C_term_2_Ux_V
@njit
def double_nabla_C_second_term_Ux_Ux(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Ux_Ux = 0
    original_point_x = reference_subset_center[0] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            # 計算 Pi = Ux , Pj = Ux 偏微分
            Hsp_PiU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            delta_x = reference_subset_point_x - reference_subset_center[0]
            double_nabla_C_term_2_Ux_Ux += ((Hsp_PiU * delta_x) * (Hsp_PiU * delta_x))
            a += 1
    double_nabla_C_term_2[2][2] = double_nabla_C_term_2_Ux_Ux
@njit
def double_nabla_C_second_term_Ux_Uy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Ux_Uy = 0
    original_point_x = reference_subset_center[0] - M
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            reference_subset_point_y = original_point_y + i
            # 計算 Pi = Ux , Pj = Uy 偏微分
            Hsp_PiU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            delta_x = reference_subset_point_x - reference_subset_center[0]
            delta_y = reference_subset_point_y - reference_subset_center[1]
            double_nabla_C_term_2_Ux_Uy += ((Hsp_PiU * delta_x) * (Hsp_PiU * delta_y))
            a += 1
    double_nabla_C_term_2[2][3] = double_nabla_C_term_2_Ux_Uy
@njit
def double_nabla_C_second_term_Ux_Vx(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Ux_Vx = 0
    original_point_x = reference_subset_center[0] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            # 計算 Pi = Ux , Pj = Vx 偏微分
            Hsp_PiU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            Hsp_PjV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_x = reference_subset_point_x - reference_subset_center[0]
            double_nabla_C_term_2_Ux_Vx += ((Hsp_PiU * delta_x) * (Hsp_PjV * delta_x))
            a += 1
    double_nabla_C_term_2[2][4] = double_nabla_C_term_2_Ux_Vx
@njit
def double_nabla_C_second_term_Ux_Vy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Ux_Vy = 0
    original_point_x = reference_subset_center[0] - M
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            reference_subset_point_y = original_point_y + i
            # 計算 Pi = Ux , Pj = Vy 偏微分
            Hsp_PiU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            Hsp_PjV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_x = reference_subset_point_x - reference_subset_center[0]
            delta_y = reference_subset_point_y - reference_subset_center[1]
            double_nabla_C_term_2_Ux_Vy += ((Hsp_PiU * delta_x) * (Hsp_PjV * delta_y))
            a += 1
    double_nabla_C_term_2[2][5] = double_nabla_C_term_2_Ux_Vy
@njit
def double_nabla_C_second_term_Uy_U(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Uy_U = 0
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_y = original_point_y + i
            # 計算 Pi = Uy , Pj = U 偏微分
            Hsp_PiU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            delta_y = reference_subset_point_y - reference_subset_center[1]
            double_nabla_C_term_2_Uy_U += ((Hsp_PiU * delta_y) * Hsp_PiU)
            a += 1
    double_nabla_C_term_2[3][0] = double_nabla_C_term_2_Uy_U
@njit
def double_nabla_C_second_term_Uy_V(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Uy_V = 0
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_y = original_point_y + i
            # 計算 Pi = Uy , Pj = V 偏微分
            Hsp_PiU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            delta_y = reference_subset_point_y - reference_subset_center[1]
            Hsp_PjV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            double_nabla_C_term_2_Uy_V += ((Hsp_PiU * delta_y) * Hsp_PjV)
            a += 1
    double_nabla_C_term_2[3][1] = double_nabla_C_term_2_Uy_V
@njit
def double_nabla_C_second_term_Uy_Ux(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Uy_Ux = 0
    original_point_x = reference_subset_center[0] - M
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            reference_subset_point_y = original_point_y + i
            # 計算 Pi = Uy , Pj = Ux 偏微分
            Hsp_PiU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            delta_y = reference_subset_point_y - reference_subset_center[1]
            delta_x = reference_subset_point_x - reference_subset_center[0]
            double_nabla_C_term_2_Uy_Ux += ((Hsp_PiU * delta_y) * (Hsp_PiU * delta_x))
            a += 1
    double_nabla_C_term_2[3][2] = double_nabla_C_term_2_Uy_Ux
@njit
def double_nabla_C_second_term_Uy_Uy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Uy_Uy = 0
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_y = original_point_y + i
            # 計算 Pi = Uy , Pj = Uy 偏微分
            Hsp_PiU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            delta_y = reference_subset_point_y - reference_subset_center[1]
            double_nabla_C_term_2_Uy_Uy += ((Hsp_PiU * delta_y) * (Hsp_PiU * delta_y))
            a += 1
    double_nabla_C_term_2[3][3] = double_nabla_C_term_2_Uy_Uy
@njit
def double_nabla_C_second_term_Uy_Vx(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Uy_Vx = 0
    original_point_x = reference_subset_center[0] - M
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            reference_subset_point_y = original_point_y + i
            # 計算 Pi = Uy , Pj = Vx 偏微分
            Hsp_PiU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            delta_y = reference_subset_point_y - reference_subset_center[1]
            Hsp_PjV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_x = reference_subset_point_x - reference_subset_center[0]
            double_nabla_C_term_2_Uy_Vx += ((Hsp_PiU * delta_y) * (Hsp_PjV * delta_x))
            a += 1
    double_nabla_C_term_2[3][4] = double_nabla_C_term_2_Uy_Vx
@njit
def double_nabla_C_second_term_Uy_Vy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Uy_Vy = 0
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_y = original_point_y + i
            # 計算 Pi = Uy , Pj = Vy 偏微分
            Hsp_PiU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            delta_y = reference_subset_point_y - reference_subset_center[1]
            Hsp_PjV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            double_nabla_C_term_2_Uy_Vy += ((Hsp_PiU * delta_y) * (Hsp_PjV * delta_y))
            a += 1
    double_nabla_C_term_2[3][5] = double_nabla_C_term_2_Uy_Vy
@njit
def double_nabla_C_second_term_Vx_U(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Vx_U = 0
    original_point_x = reference_subset_center[0] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            # 計算 Pi = Vx , Pj = U 偏微分
            Hsp_PiV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_x = reference_subset_point_x - reference_subset_center[0]
            Hsp_PjU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            double_nabla_C_term_2_Vx_U += ((Hsp_PiV * delta_x) * Hsp_PjU)
            a += 1
    double_nabla_C_term_2[4][0] = double_nabla_C_term_2_Vx_U
@njit
def double_nabla_C_second_term_Vx_V(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Vx_V = 0
    original_point_x = reference_subset_center[0] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            # 計算 Pi = Vx , Pj = V 偏微分
            Hsp_PiV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_x = reference_subset_point_x - reference_subset_center[0]
            double_nabla_C_term_2_Vx_V += ((Hsp_PiV * delta_x) * Hsp_PiV)
            a += 1
    double_nabla_C_term_2[4][1] = double_nabla_C_term_2_Vx_V
@njit
def double_nabla_C_second_term_Vx_Ux(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Vx_Ux = 0
    original_point_x = reference_subset_center[0] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            # 計算 Pi = Vx , Pj = Ux 偏微分
            Hsp_PiV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_x = reference_subset_point_x - reference_subset_center[0]
            Hsp_PjU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            double_nabla_C_term_2_Vx_Ux += ((Hsp_PiV * delta_x) * (Hsp_PjU * delta_x))
            a += 1
    double_nabla_C_term_2[4][2] = double_nabla_C_term_2_Vx_Ux
@njit
def double_nabla_C_second_term_Vx_Uy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Vx_Uy = 0
    original_point_x = reference_subset_center[0] - M
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            reference_subset_point_y = original_point_y + i
            # 計算 Pi = Vx , Pj = Uy 偏微分
            Hsp_PiV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_x = reference_subset_point_x - reference_subset_center[0]
            Hsp_PjU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            delta_y = reference_subset_point_y - reference_subset_center[1]
            double_nabla_C_term_2_Vx_Uy += ((Hsp_PiV * delta_x) * (Hsp_PjU * delta_y))
            a += 1
    double_nabla_C_term_2[4][3] = double_nabla_C_term_2_Vx_Uy
@njit
def double_nabla_C_second_term_Vx_Vx(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Vx_Vx = 0
    original_point_x = reference_subset_center[0] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            # 計算 Pi = Vx , Pj = Vx 偏微分
            Hsp_PiV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_x = reference_subset_point_x - reference_subset_center[0]
            double_nabla_C_term_2_Vx_Vx += ((Hsp_PiV * delta_x) * (Hsp_PiV * delta_x))
            a += 1
    double_nabla_C_term_2[4][4] = double_nabla_C_term_2_Vx_Vx
@njit
def double_nabla_C_second_term_Vx_Vy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Vx_Vy = 0
    original_point_x = reference_subset_center[0] - M
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            reference_subset_point_y = original_point_y + i
            # 計算 Pi = Vx , Pj = Vy 偏微分
            Hsp_PiV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_x = reference_subset_point_x - reference_subset_center[0]
            delta_y = reference_subset_point_y - reference_subset_center[1]
            double_nabla_C_term_2_Vx_Vy += ((Hsp_PiV * delta_x) * (Hsp_PiV * delta_y))
            a += 1
    double_nabla_C_term_2[4][5] = double_nabla_C_term_2_Vx_Vy
@njit
def double_nabla_C_second_term_Vy_U(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Vy_U = 0
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_y = original_point_y + i
            # 計算 Pi = Vy , Pj = U 偏微分
            Hsp_PiV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_y = reference_subset_point_y - reference_subset_center[1]
            Hsp_PjU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            double_nabla_C_term_2_Vy_U += ((Hsp_PiV * delta_y) * Hsp_PjU)
            a += 1
    double_nabla_C_term_2[5][0] = double_nabla_C_term_2_Vy_U
@njit
def double_nabla_C_second_term_Vy_V(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Vy_V = 0
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_y = original_point_y + i
            # 計算 Pi = Vy , Pj = V 偏微分
            Hsp_PiV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_y = reference_subset_point_y - reference_subset_center[1]
            double_nabla_C_term_2_Vy_V += ((Hsp_PiV * delta_y) * Hsp_PiV)
            a += 1
    double_nabla_C_term_2[5][1] = double_nabla_C_term_2_Vy_V
@njit
def double_nabla_C_second_term_Vy_Ux(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Vy_Ux = 0
    original_point_x = reference_subset_center[0] - M
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            reference_subset_point_y = original_point_y + i
            # 計算 Pi = Vy , Pj = Ux 偏微分
            Hsp_PiV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_y = reference_subset_point_y - reference_subset_center[1]
            Hsp_PjU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            delta_x = reference_subset_point_x - reference_subset_center[0]
            double_nabla_C_term_2_Vy_Ux += ((Hsp_PiV * delta_y) * (Hsp_PjU * delta_x))
            a += 1
    double_nabla_C_term_2[5][2] = double_nabla_C_term_2_Vy_Ux
@njit
def double_nabla_C_second_term_Vy_Uy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Vy_Uy = 0
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_y = original_point_y + i
            # 計算 Pi = Vy , Pj = Uy 偏微分
            Hsp_PiV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_y = reference_subset_point_y - reference_subset_center[1]
            Hsp_PjU = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_U")
            double_nabla_C_term_2_Vy_Uy += ((Hsp_PiV * delta_y) * (Hsp_PjU * delta_y))
            a += 1
    double_nabla_C_term_2[5][3] = double_nabla_C_term_2_Vy_Uy
@njit
def double_nabla_C_second_term_Vy_Vx(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Vy_Vx = 0
    original_point_x = reference_subset_center[0] - M
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_x = original_point_x + j
            reference_subset_point_y = original_point_y + i
            # 計算 Pi = Vy , Pj = Vx 偏微分
            Hsp_PiV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_y = reference_subset_point_y - reference_subset_center[1]
            delta_x = reference_subset_point_x - reference_subset_center[0]
            double_nabla_C_term_2_Vy_Vx += ((Hsp_PiV * delta_y) * (Hsp_PiV * delta_x))
            a += 1
    double_nabla_C_term_2[5][4] = double_nabla_C_term_2_Vy_Vx
@njit
def double_nabla_C_second_term_Vy_Vy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_zero_matrix, deformed_subset_aij_delta_y_zero_matrix, double_nabla_C_term_2):
    double_nabla_C_term_2_Vy_Vy = 0
    original_point_y = reference_subset_center[1] - M
    a = 0
    for i in range((2 * M) + 1):
        for j in range((2 * M) + 1):
            reference_subset_point_y = original_point_y + i
            # 計算 Pi = Vy , Pj = Vy 偏微分
            Hsp_PiV = partial_bicubic_spline(deformed_subset_aij_list[a], deformed_subset_aij_delta_x_zero_matrix[i][j], deformed_subset_aij_delta_y_zero_matrix[i][j], "partial_V")
            delta_y = reference_subset_point_y - reference_subset_center[1]
            double_nabla_C_term_2_Vy_Vy += ((Hsp_PiV * delta_y) * (Hsp_PiV * delta_y))
            a += 1
    double_nabla_C_term_2[5][5] = double_nabla_C_term_2_Vy_Vy

#####################################################################################################################
#####################################           Newton Raphson method           #####################################
#####################################################################################################################
@njit
def newton_raphson_method(reference_subset, reference_subset_center, M, right_rectified_gray, nabla_C_term_2, double_nabla_C_term_2, U, V, Ux, Uy, Vx, Vy, num, new_P):  # reference_subset_center(是子集中心座標(x,y)), shape(是用來將double_nabla_C_term_2轉成"6x6"矩陣)
    # 初始位移參數
    initial_U, initial_V, initial_Ux, initial_Uy, initial_Vx, initial_Vy = float(U), float(V), float(Ux), float(Uy), float(Vx), float(Vy)
    initial_P = np.array([initial_U, initial_V, initial_Ux, initial_Uy, initial_Vx, initial_Vy])
    # print(f'initial_P = \n{initial_P}')

    # 先計算nabla_C, double_nabla_C都用的到的參數矩陣。  (aij, graylevel, pointx, pointy)
    deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, deformed_subset_aij_list, deformed_subset_graylevel_matrix = aij_matrix(right_rectified_gray, reference_subset_center, M, U, V, Ux, Uy, Vx, Vy)

    ###########################################
    #### 計算一階梯度correlation coefficient ####
    ###########################################
    # 先來計算nabla_C的第一項
    nabla_C_term_1 = nabla_C_first_term(reference_subset, M)
    # 再來計算nabla_C的第二項，初始的"6x1"列表(list)會成形
    nabla_C_second_term_U(reference_subset, reference_subset_center, M, deformed_subset_aij_list, deformed_subset_graylevel_matrix, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, nabla_C_term_2)
    nabla_C_second_term_V(reference_subset, reference_subset_center, M, deformed_subset_aij_list, deformed_subset_graylevel_matrix, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, nabla_C_term_2)
    nabla_C_second_term_Ux(reference_subset, reference_subset_center, M, deformed_subset_aij_list, deformed_subset_graylevel_matrix, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, nabla_C_term_2)
    nabla_C_second_term_Uy(reference_subset, reference_subset_center, M, deformed_subset_aij_list, deformed_subset_graylevel_matrix, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, nabla_C_term_2)
    nabla_C_second_term_Vx(reference_subset, reference_subset_center, M, deformed_subset_aij_list, deformed_subset_graylevel_matrix, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, nabla_C_term_2)
    nabla_C_second_term_Vy(reference_subset, reference_subset_center, M, deformed_subset_aij_list, deformed_subset_graylevel_matrix, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, nabla_C_term_2)

    # 將初始的"6x1"矩陣的每一項乘上 nabla_C_term_1
    nabla_C_term_2_final = nabla_C_term_1 * nabla_C_term_2

    ###########################################
    #### 計算二階梯度correlation coefficient ####
    ###########################################
    # 先來計算double_nabla_C的第一項
    double_nabla_C_term_1 = double_nabla_C_first_term(reference_subset, M)
    # 再來計算double_nabla_C的第二項，初始的"6x6"列表(list)會成形
    double_nabla_C_second_term_U_U(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_U_V(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_U_Ux(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_U_Uy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_U_Vx(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_U_Vy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_V_U(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_V_V(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_V_Ux(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_V_Uy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_V_Vx(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_V_Vy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Ux_U(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Ux_V(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Ux_Ux(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Ux_Uy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Ux_Vx(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Ux_Vy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Uy_U(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Uy_V(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Uy_Ux(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Uy_Uy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Uy_Vx(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Uy_Vy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Vx_U(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Vx_V(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Vx_Ux(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Vx_Uy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Vx_Vx(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Vx_Vy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Vy_U(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Vy_V(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Vy_Ux(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Vy_Uy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Vy_Vx(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)
    double_nabla_C_second_term_Vy_Vy(reference_subset_center, M, deformed_subset_aij_list, deformed_subset_aij_delta_x_matrix, deformed_subset_aij_delta_y_matrix, double_nabla_C_term_2)

    # 將初始的"6x6"矩陣的每一項乘上 double_nabla_C_term_1
    double_nabla_C_term_2_final = double_nabla_C_term_1 * double_nabla_C_term_2

    ###########################################
    ####   進行 Newton Raphson 公式進行運算   ####
    ###########################################
    hessian_inv = np.linalg.inv(double_nabla_C_term_2_final)
    jacobian = nabla_C_term_2_final
    ## 計算delta_P
    delta_P = -np.dot(hessian_inv, jacobian)
    # print(f'delta_P = \n{delta_P}')

    for i in range(6):
        new_P[i] = delta_P[i] + initial_P[i]  # new_P 就是迭代第一次的6個位移參數

    num += 1
    if num == 5:
        #print(f'####### final_new_P = {new_P}')
        # print(f"第 {c} 幀右影像迭代完成。")
        pass
    else:
        for i in range(6):
            initial_P[i] = float(new_P[i].item())
        # print(f"第 {c} 幀右影像的第 {num} 次迭代...")
        newton_raphson_method(reference_subset, reference_subset_center, M, right_rectified_gray, nabla_C_term_2, double_nabla_C_term_2, initial_P[0], initial_P[1], initial_P[2], initial_P[3], initial_P[4], initial_P[5], num, new_P)
    # else:
    #     print(f"第 {c} 幀右影像迭代完成。")
    #     return new_P

#####################################################################################################################
#########################################           還原影像三維座標           ##########################################
#####################################################################################################################
@njit
def traiangulation(pointx_l, pointx_r, fx, baseline):
    disparity = pointx_l - pointx_r
    zdepth = (baseline * fx) / disparity

    return zdepth
@njit
def coordinate(baseline, pointx_l, pointy_l, pointx_r, fx, fy, ox, oy):
    X = (baseline * (pointx_l - ox)) / (pointx_l - pointx_r)
    Y = (baseline * fx * (pointy_l - oy)) / (fy * (pointx_l - pointx_r))

    return X, Y


#######################################################################################################################
#########################################               主程式                ##########################################
#######################################################################################################################

# 創建X、Y、Z array，用來儲存每一幀算出的三維座標
X_coor = []
Y_coor = []
Z_coor = []

# 讀取影像，用於計算位移
high_speed_camera = cv2.VideoCapture(r"D:\python\量測不同散斑效果(放論文的)\0219-1(software).avi")    # D:\python\振動量測-新\升速量測-找自然頻率\升速量測 - 第三次量測 - 光源高一點\雙目視覺\19V25V_6400 frames_1219.avi

# 創建影像格式
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('highspeed_matching.avi', fourcc, 20, (1280*2, 480))   # 原35 fps

##### 讀取excel檔案 #####
filename = r"D:\python\量測不同散斑效果(放論文的)\measure data.xlsx"   #D:\python\振動量測-新\升速量測-找自然頻率\升速量測 - 第三次量測 - 光源高一點\雙目視覺\19V25V-1219.xlsx
wb = openpyxl.load_workbook(filename, data_only=True)
s1 = wb["工作表2"]
c1, c2 = "A", "B"
c3, c4 = "C", "D"
A_n = 0
# A = 8/6400  #(seconds/frames)
A = 1/800
start = time.time()
while True:

    # 計算幀數
    c += 1

    retcam, frame = high_speed_camera.read()

    if retcam == False:
        print("Cannot read frame!")
        break

    else:
        # contrast = 70
        # brightness = 80
        # output = Frame * (contrast / 127 + 1) - contrast + brightness  # 轉換公式
        # # 轉換公式參考 https://stackoverflow.com/questions/50474302/how-do-i-adjust-brightness-contrast-and-vibrance-with-opencv-python
        # # 調整後的數值大多為浮點數，且可能會小於 0 或大於 255
        # # 為了保持像素色彩區間為 0～255 的整數，所以再使用 np.clip() 和 np.uint8() 進行轉換
        # output = np.clip(output, 0, 255)
        # output = np.uint8(output)
        # frame = output
        height = frame.shape[0]
        width = frame.shape[1]

        left_img = frame[0:height, 0:int(width/2)]
        right_img = frame[0:height, int(width/2):width]
        height_cut = left_img.shape[0]
        width_cut = left_img.shape[1]
        size = (width_cut, height_cut)

        # 對影像做 image rectification (影像糾正)
        def rectify(M1, d1, M2, d2, size, R, T, imgl, imgr):
            RL, RR, PL, PR, Q, roi_left, roi_right = cv2.stereoRectify(M1, d1, M2, d2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(size[0], size[1]))
            leftMapX, leftMapY = cv2.initUndistortRectifyMap(M1, d1, RL, PL, (size[0], size[1]), cv2.CV_32FC1)
            rightMapX, rightMapY = cv2.initUndistortRectifyMap(M2, d2, RR, PR, (size[0], size[1]), cv2.CV_32FC1)
            left_rectified = cv2.remap(imgl, leftMapX, leftMapY, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT)
            right_rectified = cv2.remap(imgr, rightMapX, rightMapY, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT)

            return left_rectified, right_rectified

        # 左右影像都糾正完成
        size_w, size_h = size[0]*10, size[1]
        size_rec = (size_w, size_h)
        left_rectified, right_rectified = rectify(M1, d1, M2, d2, size_rec, R, T, left_img, right_img)

        h, w = left_rectified.shape[:2]
        left_rectified_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        right_rectified_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)


        if c == 1:  # 只在影像的第一幀會執行，因為只尋找第一幀右影像的目標子集

            #####  選擇你要量測的位置  #####
            origin_pointx, origin_pointy = first_frame(left_rectified)
            origin_pointy_for_3D = round(origin_pointy)
            reference_subset = bicubic_reference_subset(left_rectified_gray, (origin_pointx, origin_pointy), M, reference_subset)   # 計算插值reference subset 以及 reference_subset的x與y座標(x與y座標分別儲存在reference_subset_x, reference_subset_y當中)
            cv2.imwrite("reference subset.jpg", reference_subset)

            ##### 選取ROI #####
            roi_x1, roi_y1, roi_x2, roi_y2 = selectroi(right_rectified)
            right_ROI_image = right_rectified_gray[roi_y1:roi_y2, roi_x1:roi_x2]
            roi_h, roi_w = right_ROI_image.shape
            cv2.imshow("test", right_ROI_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            #######################################  計算量測點三維座標  #######################################
            right_x = ZNSSD_correlation(right_rectified_gray, reference_subset, roi_x1, origin_pointy_for_3D, roi_w, M, ZNSSD_data)
            #####  開始計算Newton Raphson  #####
            U = right_x - origin_pointx  # 粗估位移量 "U分量"
            V = origin_pointy_for_3D - origin_pointy  # 粗估位移量 "V分量"
            newton_raphson_method(reference_subset, (origin_pointx, origin_pointy), M, right_rectified_gray, nabla_C_term_2, double_nabla_C_term_2, U, V, Ux, Uy, Vx, Vy, num, new_P) # 開始計算Newton-Raphson，最後會算出new_P，也就是迭代收斂出的位移參數
            right_x_new, pointy_new = shape_function((origin_pointx, origin_pointy), (origin_pointx, origin_pointy), new_P[0], new_P[1], new_P[2], new_P[3], new_P[4], new_P[5]) # 利用剛算出的new_P來找出right_rectified上的子集中心點
            right_x_new = float(right_x_new)
            pointy_new = float(pointy_new)
            #####  開始計算三維座標  #####
            reference_Z = traiangulation(origin_pointx, right_x_new, fx, baseline)  # 計算z座標
            reference_X, reference_Y = coordinate(baseline, origin_pointx, pointy_new, right_x_new, fx, fy, ox, oy)  # 計算x, y座標

            ##### 儲存XYZ座標 #####
            X_coor.append(reference_X)
            Y_coor.append(reference_Y)
            Z_coor.append(reference_Z)

            #####  在左右影像上畫上子集  #####
            cv2.line(left_rectified, (round(pointx), round(pointy)), (round(pointx), round(pointy)), (0, 255, 255), 2)
            cv2.rectangle(left_rectified, (round(pointx) - M, round(pointy) - M), (round(pointx) + M, round(pointy) + M), (0, 0, 255), 2)
            cv2.rectangle(right_rectified, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 3, cv2.LINE_AA)
            cv2.line(right_rectified, (round(right_x_new), round(pointy_new)), (round(right_x_new), round(pointy_new)), (0, 255, 255), 2)
            cv2.rectangle(right_rectified, (round(right_x_new) - M, round(pointy_new) - M), (round(right_x_new) + M, round(pointy_new) + M), (0, 0, 255), 2)

            #####  將第一幀的三維座標寫入excel  #####
            s1[f"{c1}{c + 2}"].value = 0
            s1[f"{c2}{c + 2}"].value = reference_X
            s1[f"{c3}{c + 2}"].value = reference_Y
            s1[f"{c4}{c + 2}"].value = reference_Z


        else:  # 第二幀開始，由上一幀儲存的子集來匹配左影像的目標子集；再去匹配右影像的目標子集
            search_size = left_rectified_gray[round(pointy) - (C * M):round(pointy) + (C * M) + 1, round(pointx) - (C * M):round(pointx) + (C * M) + 1]  # 左影像中設定搜尋範圍(ROI)
            #####  開始在左影像上進行子集匹配  #####
            search_size_h, search_size_w = search_size.shape
            size, position_1 = ZNSSD_correlation_for_left(search_size, reference_subset, search_size_h, search_size_w, M, ZNSSD_data)
            y = int(((position_1) / (math.sqrt(size))) + M)  # 商數為列
            x = int(((position_1) % (math.sqrt(size))) + M)  # 餘數為行
            pointx_1 = pointx - (C * M) + x  # 下一幀的ROI是原子集的四倍，可寫為 (2*C*M)+1，因此 pointx 才減掉 (C*M)
            pointy_1 = pointy - (C * M) + y  # pointx_1, pointy_1是下一幀左影像的子集中心
            #####  開始計算左影像中的子集中心座標  #####
            U = pointx_1 - origin_pointx
            V = pointy_1 - origin_pointy
            newton_raphson_method(reference_subset, (origin_pointx, origin_pointy), M, left_rectified_gray, nabla_C_term_2, double_nabla_C_term_2, U, V, Ux, Uy, Vx, Vy, num, new_P) # 由上一幀的reference subset進行Newton Raphson

            #####  開始計算左影像中的子集中心座標，並儲存reference subset  #####
            left_x_new, left_pointy_new = shape_function((origin_pointx, origin_pointy), (origin_pointx, origin_pointy), new_P[0], new_P[1], new_P[2], new_P[3], new_P[4], new_P[5])
            left_x_new = float(left_x_new)
            left_pointy_new = float(left_pointy_new)
            reference_subset = bicubic_reference_subset(left_rectified_gray, (left_x_new, left_pointy_new), M, reference_subset) # 將上一幀的reference subset覆寫過去

            #####  在右影像上進行子集匹配  #####
            right_x_1 = ZNSSD_correlation(right_rectified_gray, reference_subset, roi_x1, int(pointy_1), roi_w, M, ZNSSD_data)

            #####  開始計算右影像中的子集中心座標  #####
            U = right_x_1 - origin_pointx
            V = pointy_1 - origin_pointy
            newton_raphson_method(reference_subset, (origin_pointx, origin_pointy), M, right_rectified_gray, nabla_C_term_2, double_nabla_C_term_2, U, V, Ux, Uy, Vx, Vy, num, new_P) # 由此幀左影像所更新的reference subset進行Newton Raphson

            #####  開始計算左影像中的子集中心座標  #####
            right_x_1_new, pointy_1_new = shape_function((origin_pointx, origin_pointy), (origin_pointx, origin_pointy), new_P[0], new_P[1], new_P[2], new_P[3], new_P[4], new_P[5])
            right_x_1_new = float(right_x_1_new)
            pointy_1_new = float(pointy_1_new)

            ##### 覆蓋參考子集中心點 #####
            pointx, pointy = pointx_1, pointy_1  # 利用下一幀左影像參考子集的中心點(pointx_1, pointy_1)將第一幀左影像參考子集的中心點(pointx, pointy)覆蓋過去
            origin_pointx, origin_pointy = pointx_1, pointy_1

            #####  開始計算三維座標  #####
            deformed_Z = traiangulation(left_x_new, right_x_1_new, fx, baseline)  # 計算z座標
            deformed_X, deformed_Y = coordinate(baseline, left_x_new, left_pointy_new, right_x_1_new, fx, fy, ox, oy)  # 計算x, y座標

            ##### 儲存XYZ座標 #####
            X_coor.append(deformed_X)
            Y_coor.append(deformed_Y)
            Z_coor.append(deformed_Z)

            #####  在左右影像上畫上子集  #####
            cv2.rectangle(left_rectified, (round(pointx) - (C * M), round(pointy) - (C * M)), (round(pointx_1) + (C * M), round(pointy) + (C * M)), (255, 0, 0), 3)
            cv2.line(left_rectified, (round(left_x_new), round(left_pointy_new)), (round(left_x_new), round(left_pointy_new)), (255, 255, 0), 2)
            cv2.rectangle(left_rectified, (round(left_x_new) - M, round(left_pointy_new) - M), (round(left_x_new) + M, round(left_pointy_new) + M), (0, 0, 255), 2)
            cv2.rectangle(right_rectified, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 3)
            cv2.line(right_rectified, (round(right_x_1_new), round(pointy_1_new)), (round(right_x_1_new), round(pointy_1_new)), (255, 255, 0), 2)
            cv2.rectangle(right_rectified, (round(right_x_1_new) - M, round(pointy_1_new) - M), (round(right_x_1_new) + M, round(pointy_1_new) + M), (0, 0, 255), 2)

            #####  將每一幀的三維座標寫入excel  #####
            A_n+=A
            s1[f"{c1}{c + 2}"].value = A_n
            s1[f"{c2}{c + 2}"].value = deformed_X
            s1[f"{c3}{c + 2}"].value = reference_Y
            s1[f"{c4}{c + 2}"].value = reference_Z


        #####  開始計算每幀之間的位移量，並進行加總  #####
        if c == 1:
            pass

        else:
            delta_x = deformed_X - reference_X
            delta_y = deformed_Y - reference_Y
            delta_z = deformed_Z - reference_Z
            reference_X, reference_Y, reference_Z = deformed_X, deformed_Y, deformed_Z

            XY = np.sqrt(np.power((delta_x), 2) + np.power((delta_y), 2))
            disp = np.sqrt(np.power(XY, 2) + np.power(delta_z, 2))
            real_displacement = real_displacement + disp

            cv2.putText(left_rectified, f"EVERY FRAME'S DISP : {disp}", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            cv2.putText(left_rectified, f"TRACKING-Displacement : {real_displacement}", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            cv2.putText(left_rectified, f"Frame : {c}", (100, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        print(f"frame displacement : {disp}")
        print(f"displacement : {real_displacement}")

        cv2.namedWindow("Displacement Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Displacement Window", w, int(h/2))
        imgStack = np.hstack((left_rectified, right_rectified))
        cv2.imshow("Displacement Window", imgStack)
        out.write(imgStack)  # 將每一幀影像寫入影片

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

out.release()
cv2.destroyAllWindows()
end = time.time()
print(f"Using GPU : {end-start} seconds")
########### 以下開始維數值分析部分 ###########
total_disp_inter = 0
def calculate_interpolation_displacement(interpolated_new_coords, num, T):   # num為座標個數(例: X座標的座標個數)
    inter = T  # 欲插值倍數
    global total_disp_inter
    for i in range(num*inter):
        if i < (num*inter)-1:
            referX, referY, referZ = interpolated_new_coords[i,0], interpolated_new_coords[i,1], interpolated_new_coords[i,2]
            deforX, deforY, deforZ = interpolated_new_coords[i+1,0], interpolated_new_coords[i+1,1], interpolated_new_coords[i+1,2]
            delX, delY, delZ = (deforX-referX), (deforY-referY), (deforZ-referZ)
            referX, referY, referZ = deforX, deforY, deforZ
            XY_inter = np.sqrt(np.power((delX), 2) + np.power((delY), 2))
            disp_inter = np.sqrt(np.power(XY_inter, 2) + np.power(delZ, 2))
            total_disp_inter = total_disp_inter + disp_inter
    return total_disp_inter

##### 畫出原始運動軌跡 以及 擬合後的運動軌跡 #####
X_coor_array = np.array(X_coor)
Y_coor_array = np.array(Y_coor)
Z_coor_array = np.array(Z_coor)  # 原始運動軌跡的X、Y、Z座標
num = X_coor_array.shape[0]

# 畫原始運動軌跡3D圖
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X_coor_array, Y_coor_array, Z_coor_array, c='b', marker='o')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
middle_x = (max(X_coor_array)+min(X_coor_array))/2
middle_y = (max(Y_coor_array)+min(Y_coor_array))/2
middle_z = (max(Z_coor_array)+min(Z_coor_array))/2
ax.set_xlim3d(middle_x-0.5, middle_x+0.5)
ax.set_ylim3d(middle_y-0.5, middle_y+0.5)
ax.set_zlim3d(middle_z-0.5, middle_z+0.5)
ax.set_title(f'Origin motion trace ({num} points)')
ax.view_init(elev=90, azim=90)


# 定義擬合的函數
def func(x, a, b, c):
    return a * x[0] + b * x[1] + c
# 初始擬合參數
initial_guess = [1, 1, 1]

# 畫出原始軌跡擬合後的運動軌跡3D圖，並計算移動距離
# 進行擬合
params_2, covariance_2 = curve_fit(func, [X_coor_array, Y_coor_array], Z_coor_array, p0=initial_guess)
# 生成擬合後的Z座標
fitted_Z = func([X_coor_array, Y_coor_array], *params_2)
# 顯示擬合曲線3D圖
ax = fig.add_subplot(122, projection='3d')
ax.scatter(X_coor_array, Y_coor_array, fitted_Z, c='r', marker='o', label='Fitted Curve')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_xlim3d(middle_x-0.5, middle_x+0.5)
ax.set_ylim3d(middle_y-0.5, middle_y+0.5)
ax.set_zlim3d(middle_z-0.5, middle_z+0.5)
num3 = fitted_Z.shape[0]
ax.set_title(f'Fitted motion trace ({num3} points)')
ax.view_init(elev=90, azim=90)
fitted_coords = np.column_stack((X_coor_array, Y_coor_array, fitted_Z))
Fitted_displacement = calculate_interpolation_displacement(fitted_coords, num3, 1)
print(f"Fitted_displacement : {Fitted_displacement}")

# # 顯示原始曲線3D圖
# ax = fig.add_subplot(121, projection='3d')
# ax.plot(X_coor_array, Y_coor_array, Z_coor_array, label="Origen curve")
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')
# ax.set_xlim3d(middle_x-0.2, middle_x+0.2)
# ax.set_ylim3d(middle_y-0.2, middle_y+0.2)
# ax.set_zlim3d(middle_z-0.2, middle_z+0.2)
# ax.set_title(f'Origen motion trace ({num3} curve)')
# ax.view_init(elev=90, azim=90)
#
# # 顯示擬合曲線3D圖
# ax = fig.add_subplot(122, projection='3d')
# ax.plot(X_coor_array, Y_coor_array, fitted_Z, label="Fitted curve")
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')
# ax.set_xlim3d(middle_x-0.2, middle_x+0.2)
# ax.set_ylim3d(middle_y-0.2, middle_y+0.2)
# ax.set_zlim3d(middle_z-0.2, middle_z+0.2)
# ax.set_title(f'Fitted motion trace ({num3} curve)')
# ax.view_init(elev=90, azim=90)

plt.show()
wb.save(filename)

#################################  將X方向的位移訊號拿來做FFT頻譜轉換  #################################
amplification_factor = 30  # 放大因子
signal_normalized = (X_coor_array - np.min(X_coor_array)) / (np.max(X_coor_array) - np.min(X_coor_array))
signal_amplified_normalized = signal_normalized * amplification_factor

sampling_rate = 800  # 取樣頻率
T = 1 / sampling_rate  # 取樣間隔
N = len(signal_amplified_normalized)  # 取樣點數

# 對正弦波進行 FFT 轉換
y_fft = np.fft.fft(signal_amplified_normalized)
# 計算頻率
freq = np.fft.fftfreq(N, T)

# 只取前半部分的頻譜
half_n = N // 2
y_fft_half = y_fft[:half_n]
freq_half = freq[:half_n]

# 繪製X方向振動位移
plt.figure(figsize=(12, 6))
x = np.linspace(0, N, N)
plt.subplot(2, 1, 1)
plt.plot(x, signal_amplified_normalized, label='Displacement track')
plt.title('Rotor center vibration displacement (X axis)')
plt.xlabel("Frames")
plt.ylabel('Displacement (cm)')
plt.legend()

# 繪製半邊的 FFT 頻譜
plt.subplot(2, 1, 2)
plt.plot(freq_half, np.abs(y_fft_half), label='FFT of vibration displacement (X axis)')
plt.title('FFT spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()

# 顯示圖形
plt.tight_layout()
plt.show()
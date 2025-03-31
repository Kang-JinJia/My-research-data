import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# 此步驟是要把影相切半，
# 因為我是先用高速攝影機軟體將chessboard影像拍下來。

chessboard_photo = glob.glob(os.path.join("chessboard_stereo/mirror_chessboard", "*.png"))
photo = cv2.imread(chessboard_photo[0])
h, w = photo.shape[:2]

for i, fname in enumerate(chessboard_photo):
    img = cv2.imread(chessboard_photo[i])

    left_crop = img[0:h, 0:int(w/2)]
    right_crop = img[0:h, int(w/2):w]

    num = 1+i
    cv2.imwrite(f"chessboard_stereo/left_1/L{num}A.jpg", left_crop)
    cv2.imwrite(f"chessboard_stereo/right_1/R{num}A.jpg", right_crop)
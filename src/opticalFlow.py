# Libraries
import numpy as np
from PIL import Image, ImageDraw
from utils.draw import *
from utils.mixImages import *


# functions
def calculate_G(dif_x_arr, dif_y_arr, cx, cy):
    """
    calculate G matrix

    Parameters
    ----------
    dif_x_arr : Numpy.array
        x方向の空間微分
    dif_y_arr : Numpy.array
        y方向の空間微分
    cx : int
        中心のx座標
    cy : int
        中心のy座標

    Returns
    -------
    G : numpy.array (shape: (9, 2))
    """
    Gx = (dif_x_arr[cy-1:cy+2, cx-1:cx+2]).reshape((9, 1))
    Gy = (dif_y_arr[cy-1:cy+2, cx-1:cx+2]).reshape((9, 1))
    # print("Gx", Gx.shape)
    G = np.concatenate([Gx, Gy], 1)
    # print(G.shape)
    return G


# Parameters
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

# 変化があったかどうかを判定する閾値
thresh_length = 1

img_before_path = "../img/OpticalFlow/car-2.jpg"
img_after_path = "../img/OpticalFlow/car-3.jpg"

# open images
img_before = Image.open(img_before_path)
img_after = Image.open(img_after_path)

# image arrays
img_before_arr = np.array(img_before)
img_after_arr = np.array(img_after)

# mixed image
img_mix_arr = mixImages(img_before_arr, 0.5, img_after_arr, 0.5)

# gray images
img_before_gray = img_before.convert("L")
img_after_gray = img_after.convert("L")

# Image arrays
img_before_gray_arr = np.array(img_before_gray)  # [h, w]
img_after_gray_arr = np.array(img_after_gray)

# differential array
dif_t_arr = img_after_gray_arr - img_before_gray_arr
dif_x_arr = differential_x(img_before_gray_arr)
dif_y_arr = differential_y(img_before_gray_arr)

# 画像の高さと幅
height, width, _ = img_mix_arr.shape

# 計算結果を格納するarray
change_mat = np.zeros((40, 40))
angle_mat = np.zeros((40, 40))
length_mat = np.zeros((40, 40))

# 格子点の座標
hol = np.linspace(10, width-10, 40, dtype=int, endpoint=True)
var = np.linspace(10, height-10, 40, dtype=int, endpoint=True)

for i in range(40):
    for j in range(40):
        b = dif_t_arr[var[i]-1:var[i]+2, hol[j]-1:hol[j]+2]
        b = b.reshape((9, 1))
        # print(b.shape)
        G = calculate_G(dif_x_arr, dif_y_arr, hol[j], var[i])
        f = -np.dot(np.dot(np.linalg.pinv(np.dot(G.T, G)), G.T), b)
        # print(f.shape)
        u = f[0][0]
        v = f[1][0]
        length = np.sqrt(u**2+v**2)
        length_mat[i][j] = length

        if length >= thresh_length:
            drawDot(img_mix_arr, hol[j], var[i], color=green, size=3)
            angle = np.arctan2(v, u)
            angle_mat[i][j] = angle
            change_mat[i][j] = 1

        else:
            drawDot(img_mix_arr, hol[j], var[i], color=red, size=3)


pil_img = Image.fromarray(img_mix_arr)
d = ImageDraw.Draw(pil_img)
for i in range(40):
    for j in range(40):
        drawLine(d, hol[j], var[i], np.sqrt(length_mat[i][j]), angle_mat[i][j], color=blue, width=2)

pil_img.show()
pil_img.save("../img/OpticalFlow/out.jpg")

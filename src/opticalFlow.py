# Libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from utils.draw import *

# Parameters
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)


img_before_path = "../img/clock_before.jpg"
img_after_path = "../img/clock_after.jpg"

img_before = Image.open(img_before_path)
img_after = Image.open(img_after_path)

img_before_arr = np.array(img_before)  # [h, w, c]
img_after_arr = np.array(img_after)

img_before_gray = img_before.convert("L")
img_after_gray = img_after.convert("L")

img_before_gray_arr = np.array(img_before_gray)  # [h, w]
img_after_gray_arr = np.array(img_after_gray)

height, width = img_before_gray_arr.shape


change_mat = np.zeros((40, 40))
angle_mat = np.zeros((40, 40))
length_mat = np.zeros((40, 40))
hol = np.linspace(10, width-10, 40, dtype=int, endpoint=True)
var = np.linspace(10, height-10, 40, dtype=int, endpoint=True)

change_mat[:, 2] = 1

for i in range(40):
    for j in range(40):
        if change_mat[i][j] == 0:
            drawDot(img_before_arr, hol[j], var[i], color=red, size=3)
        else:
            drawDot(img_before_arr, hol[j], var[i], color=green, size=3)

pil_img = Image.fromarray(img_before_arr)
d = ImageDraw.Draw(pil_img)
drawLine(d, hol[12], var[1], 50, 0.7, color=blue, width=2)
pil_img.show()

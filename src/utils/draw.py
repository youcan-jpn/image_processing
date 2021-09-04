from PIL import Image
import numpy as np


def drawDot(img_arr, x, y, color=(255, 255, 255), size=15):
    """
    正方形のドットを1つ打つ

    Parameters
    ----------
    img_arr : numpy.array
        画像を表す画素値行列
    x : int
        ドットの左上のx座標
    y : int
        ドットの左上のy座標
    color : (R, G, B)
        ドットの色
    size : int
        ドット（正方形）の1辺の長さ

    Returns
    -------
    img_arr : numpy.array
        ドットを打たれた画素値行列
    """
    img_arr[y:y+size, x:x+size, :] = color
    return img_arr


def drawLine(d, sx, sy, length, angle, color=(255, 255, 255), width=5):
    """
    直線を描く

    Parameters
    ----------
    d : PIL ImageDraw object
    sx : int
        線分の端点のx座標
    sy : int
        線分の端点のy座標
    length : float
        線分の長さ
    angle : float
        弧度法
    """
    tx, ty = sx + length*np.cos(angle), sy + length*np.sin(angle)
    d.line([(sx, sy), (tx, ty)], fill=color, width=width)
    return None

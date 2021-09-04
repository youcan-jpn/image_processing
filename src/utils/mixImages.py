import numpy as np


def mixImages(image_arr_1, r1, image_arr_2, r2):
    """
    ふたつの画像の加重平均をとる

    Parameters
    ----------
    image_arr_i : numpy.array
        画像の画素値行列
    ri : float
        各画像の重み

    Returns
    -------
    ret : np.array (dtype=uint8)
        重ね合わせた画像の画素値行列
    """
    if image_arr_1.shape != image_arr_2.shape:
        raise TypeError("The shapes of arrays must be equal.")
    return (image_arr_1*r1 + image_arr_2*r2).astype("uint8")


def differential_x(arr):
    """
    画像の空間微分を求める

    Parameters
    ----------
    arr : numpy.array (h, w)
        画像の画素値行列

    Returns
    -------
    ret : numpy.array (h, w)
        x方向の勾配を格納した行列

    Examples
    --------
    input: numpy.array([[5, 9, 1],
                        [2, 4, 0],
                        [1, 8, 7]])
    output: numpy.array([[4, -8, -1],
                         [2, -4, 0],
                         [7, -1, -7]])
    """
    h, w = arr.shape
    zeros = np.zeros((h, 1))
    arr1 = np.concatenate([arr, zeros], 1)
    arr2 = np.concatenate([zeros, arr], 1)
    ret = arr1 - arr2
    ret = ret[:, 1:]  # shape: (h, w)
    return ret


def differential_y(arr):
    """
    画像の空間微分を求める

    Parameters
    ----------
    arr : numpy.array (h, w)
        画像の画素値行列

    Returns
    -------
    ret : numpy.array (h, w)
        y方向の勾配を格納した行列

    Examples
    --------
    input: numpy.array([[5, 9, 1],
                        [2, 4, 0],
                        [1, 8, 7]])
    output: numpy.array([[-3, -5, -1],
                         [-1, 4, 7],
                         [-1, -8, -7]])
    """
    _, w = arr.shape
    zeros = np.zeros((1, w))
    arr1 = np.concatenate([arr, zeros], 0)
    arr2 = np.concatenate([zeros, arr], 0)
    ret = arr1 - arr2
    ret = ret[1:, :]  # shape: (h, w)
    return ret

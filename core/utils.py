import cv2
import numpy as np


def draw_alarm_region(image):
    """
    Draws an alarm region on the input image.

    Parameters:
    - image: The input image to draw the alarm region on.

    Returns:
    - The input image with the alarm region drawn on it.
    """

    image = np.asarray(image)

    # 5邊形座標點
    points = np.array([[1460, 0], [1820, 0], [800, 1075], [0, 1075], [0, 900]], np.int32)

    # 畫出 mask
    zero = np.zeros((image.shape), dtype=np.uint8)
    zero_mask = cv2.fillPoly(zero, [points], color=(0, 0, 200), lineType=cv2.LINE_AA, shift=0)

    # 把 mask 疊到圖片上
    alpha = 1
    beta = 0.5
    gamma = 0
    mask_img = cv2.addWeighted(image, alpha, zero_mask, beta, gamma)

    return mask_img
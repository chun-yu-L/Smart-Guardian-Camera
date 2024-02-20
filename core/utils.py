import cv2
import numpy as np
from PIL import Image


def draw_inference_result(image, conf, bbox):
    """
    Draws inference results on the input image.
    
    Parameters:
    - image: the input image in cv2 format
    - conf: confidence values for the inference results
    - bbox: bounding boxes for the inference results
    
    Returns:
    - im_rgb: the image with the inference results drawn in PIL image format
    """
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f'person: {conf[i]:.2f}'
        t_size = cv2.getTextSize(label, fontface, 0.4, 2)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 12
        cv2.rectangle(image, (x1, y1), c2, (0, 255, 0), -1)

        cv2.putText(image, label, (x1, y1 - 8), fontface, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    im_rgb = Image.fromarray(image[..., ::-1])

    return im_rgb


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

    # 轉回 RGB
    mask_img_rgb = Image.fromarray(mask_img[..., ::-1])

    return mask_img
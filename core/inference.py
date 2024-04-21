from ultralytics import YOLO
import numpy as np


def human_bbox_prediction(image):
    """
    Predicts bounding boxes for humans in the given image using YOLO model.

    Args:
        image: The input image to be processed.

    Returns:
        conf: Numpy array of confidence values for the inference results.
        bbox: Numpy array of the predicted bounding box coordinates in the format (xmin, ymin, xmax, ymax).
    """
    # 預測 bbox 與 conf
    model = YOLO('yolov8n.pt')
    results = model(image, classes=[0])

    bbox = results[0].boxes.xyxy.numpy()
    conf = results[0].boxes.conf.numpy()


    # 超出圖片的 bbox 縮回來
    # 獲取圖片尺寸
    height, width, _ = image.shape

    # 修正 bbox 座標
    bbox[:, 0] = np.clip(bbox[:, 0], 0, width)
    bbox[:, 1] = np.clip(bbox[:, 1], 0, height)
    bbox[:, 2] = np.clip(bbox[:, 2], 0, width)
    bbox[:, 3] = np.clip(bbox[:, 3], 0, height)

    return bbox, conf
    






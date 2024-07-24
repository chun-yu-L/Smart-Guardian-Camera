import numpy as np
from ultralytics import YOLO


model = YOLO("yolov8n.pt")

def human_bbox_prediction(image):
    """
    Predicts bounding boxes for humans in the given image using YOLO model.

    Args:
        image: The input image to be processed.

    Returns:
        bbox: Numpy array of the predicted bounding box coordinates in the format (xmin, ymin, xmax, ymax).
        conf: Numpy array of confidence values for the inference results.
    """
    # 預測 bbox
    results = model(image, conf=0.4, classes=[0])

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


def human_tracker_yolo(image):
    """
    Predicts bounding boxes and IDs for humans in the given image using YOLO model.

    Args:
        image: The input image to be processed.

    Returns:
        bbox: Numpy array of the predicted bounding box coordinates in the format (xmin, ymin, xmax, ymax).
        conf: Numpy array of confidence values for the inference results.
        id: A numpy array of the IDs associated with the predicted bounding boxes.

    This function uses the YOLO model to predict bounding boxes and IDs for humans in the given image. It takes an RGB image as input and returns the predicted bounding boxes, confidence values, and IDs.
    """
    # 預測 bbox 與 ID
    results = model.track(image, conf=0.4, classes=[0], persist=True)

    bbox = results[0].boxes.xyxy.numpy()
    conf = results[0].boxes.conf.numpy()
    id = results[0].boxes.id.numpy()

    # 超出圖片的 bbox 縮回來
    # 獲取圖片尺寸
    height, width, _ = image.shape

    # 修正 bbox 座標
    bbox[:, 0] = np.clip(bbox[:, 0], 0, width)
    bbox[:, 1] = np.clip(bbox[:, 1], 0, height)
    bbox[:, 2] = np.clip(bbox[:, 2], 0, width)
    bbox[:, 3] = np.clip(bbox[:, 3], 0, height)

    return bbox, conf, id

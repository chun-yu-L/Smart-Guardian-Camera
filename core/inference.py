from ultralytics import YOLO


def human_bbox_prediction(image):
    """
    Predicts bounding boxes for humans in the given image using YOLO model.

    Args:
        image: The input image to be processed.

    Returns:
        conf: Numpy array of confidence values for the inference results.
        bbox: Numpy array of the predicted bounding box coordinates in the format (xmin, ymin, xmax, ymax).
    """
    model = YOLO('yolov8n.pt')
    results = model(image, classes=[0])
    bbox = results[0].boxes.xyxy.numpy()
    conf = results[0].boxes.conf.numpy()

    return conf, bbox
    






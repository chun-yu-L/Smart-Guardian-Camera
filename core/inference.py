from PIL import Image
import cv2
from ultralytics import YOLO


def human_bbox_prediction(image):
    """
    Predicts bounding boxes for humans in the given image using YOLO model.

    Parameters:
    - image: The input image to be processed.

    Returns:
    - conf: Numpy array of confidence values for the inference results.
    - bbox: Numpy array of the predicted bounding box coordinates in the format (xmin, ymin, xmax, ymax).
    """
    model = YOLO('yolov8n.pt')
    results = model(image, classes=[0])
    bbox = results[0].boxes.xyxy.numpy()
    conf = results[0].boxes.conf.numpy()

    return conf, bbox
    

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




import cv2
import numpy as np
from PIL import Image


def detect_person_in_alart_zone(bbox):
    alarm_region = np.array([[1460, 0], [1820, 0], [850, 1080], [0, 1080], [0, 900]], np.int32)

    bbox_inside = np.empty((0, 4),dtype=np.float32)
    bbox_outside = np.empty((0, 4),dtype=np.float32)

    # 判斷人物是否在警戒區
    for coord in bbox:
        # 計算人物的座標底部中點
        bottom_center = np.array([(coord[0] + coord[2]) / 2, coord[3]])

        # 檢查中點是否在指定框內
        detect_result = cv2.pointPolygonTest(alarm_region, bottom_center, False)

        if detect_result >= 0:
            # 人物在警戒區內
            bbox_inside = np.append(bbox_inside, [coord], axis=0)
        else:
            # 人物在警戒區外
            bbox_outside = np.append(bbox_outside, [coord], axis=0)
            
    return bbox_inside, bbox_outside


class YoloImageProcessing:
    def __init__(self):
        self.fontface = cv2.FONT_HERSHEY_SIMPLEX
        self.alarm_region = np.array([[1460, 0], [1820, 0], [800, 1075], [0, 1075], [0, 900]], np.int32)

    def draw_inference_result(self, image, conf, bbox, color = 'blue'):
        """
        Draws inference results on the input image.
        
        Args:
        - image: the input image in cv2 format
        - conf: confidence values for the inference results
        - bbox: bounding boxes for the inference results
        
        Returns:
        - im_rgb: the image with the inference results drawn in BGR format (openCV)
        """
        color_map = {
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
         }
        selected_color = color_map.get(color)

        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(image, (x1, y1), (x2, y2), selected_color, 2)

            label = f'person: {conf[i]:.2f}'
            t_size = cv2.getTextSize(label, self.fontface, 0.5, 2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 12
            cv2.rectangle(image, (x1, y1), c2, selected_color, -1)

            cv2.putText(image, label, (x1, y1 - 8), self.fontface, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return image


    def draw_alarm_region(self, image):
        """
        Draws an alarm region on the input image.

        Args:
        - image: The input image to draw the alarm region on.

        Returns:
        - The input image with the alarm region drawn on it.
        """
        # 畫出 mask
        zero = np.zeros((image.shape), dtype=np.uint8)
        zero_mask = cv2.fillPoly(zero, [self.alarm_region], color=(0, 0, 200), lineType=cv2.LINE_AA, shift=0)

        # 把 mask 疊到圖片上
        # alpha, beta 控制透明度；gamma 控制曝光度
        alpha = 1
        beta = 0.5
        gamma = 0
        mask_img = cv2.addWeighted(image, alpha, zero_mask, beta, gamma)

        return mask_img


    def save_image(self, image, path):
        """
        Save an image to the specified path.

        Args:
            image: The input image as a numpy array in BGR format.
            path: The file path where the image will be saved.
        """
        img_PIL = Image.fromarray(image[..., ::-1])
        img_PIL.save(path)

class PerspectiveTransform:
    def __init__(self, origin_points, transformed_width, transformed_height):
        self.origin_points = origin_points
        self.transformed_width = transformed_width
        self.transformed_height = transformed_height

    def get_matrix(self):

        src_points = self.origin_points
        # Define the destination points for perspective transformation
        dst_points = np.float32([[0, 0], [self.transformed_width, 0], [self.transformed_width, self.transformed_height], [0, self.transformed_height]])

        # Calculate the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        return matrix
    
    def plot(self, image):
        matrix = self.get_matrix()

        # Apply the perspective transformation
        transformed_image = cv2.warpPerspective(image, matrix, (self.transformed_width, self.transformed_height))

        return transformed_image

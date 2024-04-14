import cv2
import numpy as np
from PIL import Image


def detect_person_in_alert_zone(bbox):
    """
    Detects if a person is within the alarm zone.

    Args:
        bbox (numpy.array): An array of bounding boxes representing coordinates of detected objects.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]
            bbox_inside: Bounding boxes of the persons detected inside the alarm zone.
            bbox_outside: Bounding boxes of the persons detected outside the alarm zone.
    """

    alarm_region = np.array([[1460, 0], [1820, 0], [800, 1075], [0, 1075], [0, 900]], np.int32)

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

def calculate_distance_to_alert_zone(bbox, conf, perspective_transform):
    """
    Calculate the distance of each bounding box to the alert zone based on the provided perspective transform.
    
    Args:
        bbox (numpy.ndarray): An array of bounding boxes representing coordinates of detected objects.
        conf (numpy.ndarray): An array of confidence values for the inference results
        perspective_transform (PerspectiveTransform): An instance of the PerspectiveTransform class.
    
    Returns:
        dict: A dictionary with bounding boxes sorted into 'inside', 'safe', and 'near' zones, each with its distance to the alert zone.
    """

    alert_region = np.array([[1460, 0], [1820, 0], [800, 1075], [0, 1075], [0, 900]], np.float32)
    alert_region_transform = perspective_transform.transform_points(alert_region)

    bbox_distance_dict = {
        'inside': ([], [], []),
        'safe': ([], [], []),
        'near': ([], [], [])
        }
    
    # 判斷人物是否在警戒區
    for i, coord in enumerate(bbox):
        # 計算人物的座標底部中點
        bottom_center = np.array([(coord[0] + coord[2]) / 2, coord[3]])
        
        # 將底部中點從圖像坐標系轉換到世界坐標系
        transform_bottom_center = perspective_transform.transform_points(bottom_center)
        
        # 計算人物到警戒區邊界的距離
        distance = -cv2.pointPolygonTest(alert_region_transform, transform_bottom_center, True)

        if distance > 5:
            # 人物遠離警戒區
            bbox_distance_dict['safe'][0].append(coord)
            bbox_distance_dict['safe'][1].append(distance)
            bbox_distance_dict['safe'][2].append(conf[i])

        elif 0 < distance <= 5:
            # 人物靠近警戒區
            bbox_distance_dict['near'][0].append(coord)
            bbox_distance_dict['near'][1].append(distance)
            bbox_distance_dict['near'][2].append(conf[i])

        else:
            # 人物在警戒區內
            bbox_distance_dict['inside'][0].append(coord)
            bbox_distance_dict['inside'][1].append(distance)
            bbox_distance_dict['inside'][2].append(conf[i])

    return bbox_distance_dict
    

class YoloImageProcessing:
    def __init__(self):
        self.fontface = cv2.FONT_HERSHEY_SIMPLEX
        self.fontscale = 0.7
        self.alarm_region = np.array([[1460, 0], [1820, 0], [800, 1075], [0, 1075], [0, 900]], np.int32)

    def draw_inference_result(self, image, conf, bbox, color = 'blue'):
        """
        Draws inference results on the input image.
        
        Args:
            image (cv2 format): the input image in cv2 format
            conf (numpy.array): confidence values for the inference results
            bbox (numpy.array): bounding boxes for the inference results
        
        Returns:
            image (cv2 format): the image with the inference results drawn in BGR format (openCV)
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
            t_size = cv2.getTextSize(label, self.fontface, self.fontscale, 2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 12
            cv2.rectangle(image, (x1, y1), c2, selected_color, -1)

            cv2.putText(image, label, (x1, y1 - 8), self.fontface, self.fontscale, (255, 255, 255), 1, cv2.LINE_AA)

        return image
    
    def draw_alert_distances(self, image, bbox_distance_dict):
        """
        Draws alert distances on the input image based on the bbox_distance_dict provided.

        Parameters:
            image (cv2 format): The image on which to draw the alert distances.
            bbox_distance_dict (dict): A dictionary containing information about bounding boxes, distances, and confidences.

        Returns:
            image (cv2 format): The image with the alert distances drawn.
        """
        colors = {'inside': (0, 0, 255), 'safe': (255, 0, 0), 'near': (0, 220, 255)}  # 定義不同區域的顏色

        for zone, (bboxes, distances, confidences) in bbox_distance_dict.items():
            color = colors[zone]  # 獲取區域對應的顏色
            for box, distance, conf in zip(bboxes, distances, confidences):
                x1, y1, x2, y2 = [int(coord) for coord in box]  # 獲取邊界框的座標
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # 繪製邊界框

                # 寫上confidence
                conf_label = f'person: {conf:.2f}'
                t_size_conf = cv2.getTextSize(conf_label, self.fontface, self.fontscale, 2)[0]
                c2 = x1 + t_size_conf[0], y1 - t_size_conf[1] - 12 # 計算文字標籤框的右上角座標
                cv2.rectangle(image, (x1, y1), c2, color, -1)
                cv2.putText(image, conf_label, (x1, y1 - 8), self.fontface, self.fontscale, (255, 255, 255), 2, cv2.LINE_AA)

                # 寫上距離
                if distance > 0:
                    distance_label = f'Distance: {distance:.2f}'
                    t_size_dist = cv2.getTextSize(distance_label, self.fontface, self.fontscale, 2)[0]
                    c3 = x1 + t_size_dist[0], c2[1] - t_size_dist[1] - 12 # 計算文字標籤框的右上角座標
                    cv2.rectangle(image, (x1, c2[1]), c3, color, -1)
                    cv2.putText(image, distance_label, (x1, c2[1] - 8), self.fontface, self.fontscale, (255, 255, 255), 2, cv2.LINE_AA)

        return image
    

    def draw_alarm_region(self, image):
        """
        Draws an alarm region on the input image

        Args:
            image (cv2 format): The input image in cv2 format to draw the alarm region on

        Returns:
            mask_img (cv2 format): The input image with the alarm region drawn on it in BGR format (openCV)
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
            image (cv2 format): The input image as a numpy array in BGR format.
            path (str): The file path where the image will be saved.
        """
        img_PIL = Image.fromarray(image[..., ::-1])
        img_PIL.save(path)

class PerspectiveTransform:
    def __init__(self, source_points, transformed_width, transformed_height):
        self.source_points = source_points
        self.transformed_width = transformed_width
        self.transformed_height = transformed_height
        self._matrix = None

    @property
    def matrix(self):
        if self._matrix is None:
            src_points = self.source_points

            # Define the destination points for perspective transformation
            dst_points = np.float32([
                [0, 0],
                [self.transformed_width, 0],
                [self.transformed_width,self.transformed_height],
                [0, self.transformed_height]
            ])

            # Calculate the perspective transformation matrix
            self._matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        return self._matrix

    def transform_points(self, points):
        """
        A function that transforms given point(s) using a perspective transformation matrix.

        Args:
            points (np.ndarray): A Nx2 numpy array of (x, y) coordinates.

        Returns:
            transformed_points (np.ndarray): A Nx2 numpy array of transformed (x, y) coordinates.
        """

        # Apply the perspective transformation
        transformed_points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), self.matrix).squeeze()

        return transformed_points
    
    def plot(self, image):
        """
        Plot the given image using a perspective transformation.

        Args:
            self (object): The object instance
            image (cv2 format): The input image to be transformed

        Returns:
            image (cv2 format): The transformed image after applying the perspective transformation
        """

        # Apply the perspective transformation
        transformed_image = cv2.warpPerspective(image, self.matrix, (self.transformed_width, self.transformed_height))

        return transformed_image

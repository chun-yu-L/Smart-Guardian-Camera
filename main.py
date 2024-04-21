
import io
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from core.inference import human_bbox_prediction
from core.utils import YoloImageProcessing, PerspectiveTransform, AlertDetector

app = FastAPI()

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    # 讀取上傳的圖片
    pil_img = Image.open(io.BytesIO(await file.read()))
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 人物偵測
    bbox, conf = human_bbox_prediction(img)

    # 繪製結果
    YoloPlot = YoloImageProcessing()
    result = YoloPlot.draw_inference_result(img, conf, bbox)

    # 回傳圖片
    buffer = io.BytesIO()
    Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)).save(buffer, format="jpeg")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/jpeg")


@app.post("/inside_outside_alert/")
async def is_inside_alert_zone(file: UploadFile = File(...)):
    # 讀取上傳的圖片
    pil_img = Image.open(io.BytesIO(await file.read()))
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 人物偵測
    bbox, conf = human_bbox_prediction(img)

    # 判斷人物是否在警戒區
    detector = AlertDetector()
    bbox_dict = detector.is_inside_alert_zone(bbox, conf)

    # 繪製結果
    YoloPlot = YoloImageProcessing()
    result = YoloPlot.draw_inside_or_outside(img, bbox_dict)
    result = YoloPlot.draw_alert_zone(result)

    # 回傳圖片
    buffer = io.BytesIO()
    Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)).save(buffer, format="jpeg")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/jpeg")

@app.post("/distance_to_alert_zone/")
async def distance_to_alert_zone(file: UploadFile = File(...)):
    # 讀取上傳的圖片
    pil_img = Image.open(io.BytesIO(await file.read()))
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 人物偵測
    bbox, conf = human_bbox_prediction(img)

    # 視角轉換
    src_point = np.array([[0, 900], [1460, 0], [1820, 0], [800, 1075]], np.float32)
    PT = PerspectiveTransform(src_point, 50, 20)

    # 判斷與警戒區的距離
    detector = AlertDetector()
    bbox_dict = detector.distance_to_alert_zone(bbox, conf, PT)

    # 繪製結果
    YoloPlot = YoloImageProcessing()
    result = YoloPlot.draw_alert_distances(img, bbox_dict)
    result = YoloPlot.draw_alert_zone(result)

    # 回傳圖片
    buffer = io.BytesIO()
    Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)).save(buffer, format="jpeg")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/jpeg")
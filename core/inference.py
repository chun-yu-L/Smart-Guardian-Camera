from PIL import Image
import cv2
from ultralytics import YOLO


# load model weight
model = YOLO('yolov8n.pt')

video_path = '/app/example/Oxford Town Centre/10sec.mp4'
cap = cv2.VideoCapture(video_path)

idx = 0
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, classes=[0])

        # Plot results image
        annotated_frame = results[0].plot()
        im_rgb = Image.fromarray(annotated_frame[..., ::-1])

        # Save results to disk
        # 直接 result[0].save 也可以只是看起來同畫質檔案比較大
        im_rgb.save(f'/app/example/results/results_{idx}.jpg')

        idx += 1

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object
cap.release()




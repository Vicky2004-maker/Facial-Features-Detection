import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

from GenderDetection import GenderDetection

# %%

yolo = YOLO(r"C:\Users\vicky\Downloads\yolov8x-oiv7.pt")
gender_model = GenderDetection()

# %%
vid = cv2.VideoCapture(0)
while True:
    ret, frame = vid.read()
    results = yolo.predict(source=np.array(frame), classes=264)

    if len(results[0].boxes.cls) != 0:
        for result in results:
            x, y, w, h = result.boxes.xywh[0].int().cpu().numpy()
            x -= 75
            y -= 100
            w += 25
            h += 20

            gender_prediction = gender_model.predict_numpy(frame[x:(x + w), y:(y + h)], print_output=False, cv=False)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, gender_prediction.upper(), (x, y + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

    cv2.imshow('Faces', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
# %%

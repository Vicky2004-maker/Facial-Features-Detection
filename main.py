import cv2
import numpy as np
from ultralytics import YOLO

from helper import GenderDetection
from cv2 import CascadeClassifier

# %%

yolo = YOLO(r"C:\Users\vicky\Downloads\yolov8x-oiv7.pt")
cascade_path = r"C:\Users\vicky\Downloads\haarcascade_frontalface_default.xml"
gender_model = GenderDetection()

# %%
vid = cv2.VideoCapture(0)
while True:
    ret, frame = vid.read()
    # results = yolo.predict(source=np.array(frame), classes=264)
    harr_cascade = CascadeClassifier(cascade_path)
    frame = np.uint8(frame)
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = harr_cascade.detectMultiScale(frame, 1.03, minNeighbors=7, minSize=(40, 40))
    print('=' * 100)
    print(results)
    print('=' * 100)
    if len(results) != 0:
        for (x, y, w, h) in results:
            gender_prediction = gender_model.predict_numpy(frame[x:(x + w), y:(y + h)], print_output=False)
            cv2.putText(frame, gender_prediction.upper(), (x, y + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Faces', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
# %%

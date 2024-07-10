import cv2
import numpy as np
import ultralytics

# %%

vid = cv2.VideoCapture(0)
while (True):
    ret, frame = vid.read()
    haar_cascade = cv2.CascadeClassifier(r"C:\Users\vicky\Downloads\haarcascade_frontalface_default.xml")
    faces = haar_cascade.detectMultiScale(frame, 1.005, 6)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Faces', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()

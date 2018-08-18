
import json

import cv2
import numpy as np


# CONSTANTS
FONT = cv2.FONT_HERSHEY_DUPLEX
label_mapping = json.load(open("./label_mapping.json"))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('./face_model.yml')
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")


video_recorder = cv2.VideoCapture(0)

while True:

    rval, image_captured = video_recorder.read()

    # converting to Greyscale to reduce the processing
    grey_image = cv2.cvtColor(image_captured, cv2.COLOR_BGR2GRAY)

    # detecting the face frames using cascade
    faces = face_cascade.detectMultiScale(grey_image, 1.3, 5)

    # for each frame (length, breadth, width, height)
    for (x, y, w, h) in faces:

        cv2.rectangle(image_captured, (x-20, y-20), (x+w+20, y+h+20), (0, 255, 0), 4)

        label, confidence = face_recognizer.predict(grey_image[y:y+h, x:x+w])

        name = label_mapping[str(label)] + " {0:.2f}%".format(round(100 - confidence, 2))

        cv2.rectangle(image_captured, (x-22, y-90), (x+w+22, y-22), (0, 255, 0), -1)
        cv2.putText(image_captured, name, (x, y-40), FONT, 1, (255, 255, 255), 3)

    cv2.imshow('im', image_captured)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


video_recorder.release()
cv2.destroyAllWindows()

import os
import json

import numpy as np
import cv2
from PIL import Image

label_mapping = {}

def fetch_faces_name(detector, path="./GeneratedDataset/"):

    # FaceSamples list
    face_samples = []

    # FaceNames list
    names = []

    # fetching all the dataset paths
    dataset_path_list = [os.path.join(path, file_name) for file_name in os.listdir(path)]

    for img_path in dataset_path_list:

        # reading and converting into greyscale(L)
        pillow_image = Image.open(img_path).convert('L')

        # converting into numpy array
        numpy_img = np.array(pillow_image, 'uint8')

        # fetching the name of the image from path
        image_name = str(os.path.split(img_path)[-1].split("-")[1])
        id = str(os.path.split(img_path)[-1].split("-")[2].split(".")[0])

        # fetch faces based in detector
        faces = detector.detectMultiScale(numpy_img)

        for (x, y, w, h) in faces:

            face_samples.append(numpy_img[y:y+h, x:x+w])
            label_mapping[str(id)] = image_name
            names.append(int(id))

    return face_samples, names


face_detector = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

faces, names = fetch_faces_name(face_detector)

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(faces, np.array(names))

recognizer.save('face_model.yml')

with open("./label_mapping.json", "w") as f:
    f.write(json.dumps(label_mapping))



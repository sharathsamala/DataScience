
import cv2

# CONSTANTS

OUTPUT_PATH = "./GeneratedDataset"

# load cascade to detect facial object in video
faceCascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

# Start video recorder
video_recorder = cv2.VideoCapture(0)





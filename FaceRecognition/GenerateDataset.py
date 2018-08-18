
import cv2
import sys
import os
import time

# CONSTANTS
OUTPUT_PATH = "./GeneratedDataset"
SAMPLE_COUNT = 20

# load cascade to detect facial object in video
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")


def input_name():

    print("Select Option : \n---------------------\n1. Generate dataset for a person\n2. Quit\n---------------------")

    input_val = input().strip()

    if input_val not in ['1','2']:
        print("Invalid input, Exiting .....")
        try:
            cv2.destroyAllWindows()
        except:
            pass
        sys.exit(0)
    elif input_val == '2':
        print("Exiting .....")
        try:
            cv2.destroyAllWindows()
        except:
            pass
        sys.exit(0)

    print("Enter the name(without spaces) of the face : ")
    name = input().strip()

    generateDataSamples(name)


def generateDataSamples(name):

    # Start video recorder
    video_recorder = cv2.VideoCapture(0)

    count = 0

    try :
        while True:
            rval, image_caputred = video_recorder.read()

            # converting to Greyscale to reduce the processing
            grey_image = cv2.cvtColor(image_caputred, cv2.COLOR_BGR2GRAY)

            # detecting the face frames using cascade
            faces = face_cascade.detectMultiScale(grey_image, 1.3, 5)

            # for each frame (length, breadth, width, height)
            for (x, y, w, h) in faces:

                # cropping the image
                cv2.rectangle(image_caputred, (x, y), (x+w, y+h), (255, 0, 0), 2)

                #writing the image into the dir
                cv2.imwrite(os.path.join(OUTPUT_PATH, "IMG-"+name.lower()+"-"+str(int(time.time()))+".jpg"), grey_image[y:y+h, x:x+w])

                # show image
                cv2.imshow('frame', image_caputred)

                count = count + 1

                cv2.waitKey(1000)

            if count > SAMPLE_COUNT:
                video_recorder.release()
                input_name()

    except:
        video_recorder.release()
        cv2.destroyAllWindows()


# create the dir if not exists :

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Start job
input_name()






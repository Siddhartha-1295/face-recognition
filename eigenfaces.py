import cv2
import os
from matplotlib import pyplot as plt
# Initialize cascade classifier with pre-trained haar-like facial features
classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
# Read an example image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
im=os.path.join(BASE_DIR,"images\siddhu\me-1.jpg")
image = cv2.imread(im)
# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect faces in the image
face = classifier.detectMultiScale(gray,1.1,5,(30, 30),cv2.CV_HAAR_SCALE_IMAGE +cv2.CV_HAAR_DO_CANNY_PRUNING +cv2.CV_HAAR_FIND_BIGGEST_OBJECT +cv2.CV_HAAR_DO_ROUGH_SEARCH)
# Draw a rectangle around the faces
for (x, y, w, h) in face:
    cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
# Display image
plt.imshow(gray, 'gray')

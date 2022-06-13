import os
from PIL import Image
import numpy as np
import cv2
import pickle

#print(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#print(BASE_DIR)
image_dir=os.path.join(BASE_DIR,"images")
#image_dir='C:\Users\adithya\AppData\Local\Programs\Python\Python36-32\src\images'
#print(image_dir)
face_xml=os.path.join(BASE_DIR,"cascades/data/haarcascade_frontalface_alt2.xml")
face_cascade= cv2.CascadeClassifier(face_xml)
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id=0
label_ids={}
y_labels = []
x_train = []

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            #print(file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
           # print(label,path)
            if not label in label_ids:
                label_ids[label] =current_id
                current_id += 1
            id_= label_ids[label]
            #print(label_ids)
            #y_labels.append(label)
            #x_train.append(path)
            pil_image = Image.open(path).convert("L")
            size=(550,550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image,"uint8")
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array, 1.7, 5)
            
            for (x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

#print(y_labels)
#print(x_train)

with open("face-labels.pickle",'wb') as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_labels))
print("training process completed")
recognizer.save("face-trainner.yml")
                
            

import os
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
eye_xml=os.path.join(BASE_DIR,"cascades/data/haarcascade_eye.xml")
eye_cascade = cv2.CascadeClassifier(eye_xml)
smile_xml=os.path.join(BASE_DIR,"cascades/data/haarcascade_smile.xml")
smile_cascade = cv2.CascadeClassifier(smile_xml)

trainer_file=os.path.join(BASE_DIR,"face-trainner.yml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainner.yml")

labels={"person_name": 1}
lable_file=os.path.join(BASE_DIR,"face-labels.pickle")
with open(lable_file,'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
    
cap=cv2.VideoCapture(0)
Id=input("enter your name")
sampleNum=0

while True:
    #capture frame meeby frame
    ret,frame=cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    
    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        #recognise deep learned model predict keras tensorflow 
        id_,conf=recognizer.predict(roi_gray)
        if conf>=4 and conf<=85:
            #print(id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            #org=(50,50)
            stroke =2
            cv2.putText(frame, str(name), (x,y), font, 1, color, stroke, cv2.LINE_AA)

        #print(image_dir)
        #img_item="7.png"
        #cv2.imwrite(img_item, roi_color)
        #sampleNum=sampleNum+1
        #cv2.imwrite("'dataset/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
        if not os.path.exists(str(Id)):
            path = os.path.join(image_dir,str(Id))
            print(path);
            while(sampleNum<=100): 
                sampleNum=sampleNum+1
                cv2.imwrite(path+'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #cv2.imwrite(path+Id +'.'+ str(sampleNum) +".jpg",gray[y:y+h,x:x+w])
        color = (255,0,0) #rgb
        stroke = 2
        width=x+w
        height=y+h
        cv2.rectangle(frame,(x,y),(width,height),color,stroke)
        sampleNum=sampleNum+1
        
        subitems=smile_cascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in subitems:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    #display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
#when everything is done, release the capture
cap.release()
cv2.destroyAllWindows() 

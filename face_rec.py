import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
#features=np.load('features.npy')
#labels=np.load('labels.npy')
ppl = ['ben affleck','chris evans','matt damon']
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
img = cv2.imread('images/val/3.jpg',1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces_rect = face_cascade.detectMultiScale(gray,1.1,4)
for (x,y,w,h) in faces_rect:
    face_roi = gray[y:y+h,x:x+h]
    label,acc = face_recognizer.predict(face_roi)
    print('label :',ppl[label],acc)
    cv2.putText(img,str(ppl[label]),(20,20),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,0,0),2)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('person',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
import os
import cv2
import numpy as np

ppl = ['ben affleck','chris evans','matt damon']
DIR = r'C:\Users\user\Desktop\ML\opencv\images'
features = []
labels = []
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
def create_train():
    for person in ppl:
        path = os.path.join(DIR,person)
        label = ppl.index(person)
        #print(path,label)
        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_array = cv2.imread(img_path)
            gray = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
            face_rectangle = face_cascade.detectMultiScale(gray,1.1,5)
            for (x,y,w,h) in face_rectangle:
                faces_roi = gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
features= np.array(features,dtype='object')
labels= np.array(labels)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#train on features and labels

face_recognizer.train(features,labels)
face_recognizer.save('face_trained.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)
from mtcnn import MTCNN
import cv2
from pygame import mixer
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os

detector = MTCNN()
#Load a videopip TensorFlow
video_capture = cv2.VideoCapture(0)
model = load_model("mask_recog_ver2.h5")
folder="/home/perel/facemask-detection/test"
output_path="/home/perel/facemask-detection/test_out"

for filename in os.listdir(folder):
    frame = cv2.imread(os.path.join(folder,filename))
    faces_list=[]
    preds=[]
    label_list=[]
    #frame = cv2.resize(frame, (352, 240)) #resize to speedup 
    boxes = detector.detect_faces(frame)
    if boxes: 
        box = boxes[0]['box']
        conf = boxes[0]['confidence']
        x, y, w, h = box[0], box[1], box[2], box[3]
        if conf > 0.5:
            face_frame = frame[y:y+h,x:x+w]
            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            face_frame = cv2.resize(face_frame, (224, 224))
            face_frame = img_to_array(face_frame)
            face_frame = np.expand_dims(face_frame, axis=0)
            face_frame =  preprocess_input(face_frame)
            faces_list.append(face_frame)
            if len(faces_list)>0:
                preds = model.predict(faces_list)
            for pred in preds:
                (mask, withoutMask) = pred
            if (mask > withoutMask): 
                label = "Mask"
            else: 
                label="No Mask" 
            label_list.append(label)   
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (x, y- 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)
            cv2.imwrite(os.path.join(output_path ,filename), frame)
        

 
    
    
     



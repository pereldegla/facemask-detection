from mtcnn import MTCNN
import cv2

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import math

detector = MTCNN()
#Load a videopip TensorFlow
video_capture = cv2.VideoCapture("/home/perel/Téléchargements/WIN_20210106_16_58_43_Pro.mp4")
model = load_model("mask_recog_ver2.h5")
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
count=0
ratio=2
while (True):
    ret, frame = video_capture.read()
    faces_list=[]
    preds=[]
    #frame = cv2.resize(frame, (math.floor(frame_width/ratio), math.floor(frame_height/ratio))) #resize to speedup 
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
                #sound.play() #use with confidence score 
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (x, y- 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
     
            cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)
            out.write(frame)
            """count+=30
            video_capture.set(1,count)"""
    #cv2.imshow("Frame", frame)
    if cv2.waitKey(25) and 0xFF == ord('q'):
        break
     
video_capture.release()
out.release()
cv2.destroyAllWindows()
    


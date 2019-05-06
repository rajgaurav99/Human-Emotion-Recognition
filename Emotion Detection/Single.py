import cv2
from keras.preprocessing.image import img_to_array
import imutils
from keras.models import load_model
import numpy as np

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["Angry" ,"Disgust","Scared", "Happy", "Sad", "Surprised", "Neutral"]



# starting video streaming
camera = cv2.VideoCapture(0)
while True:
    frame = camera.read()[1]
    #reading the frame
    frame = imutils.resize(frame,width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

 
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)

                w = int(prob * 200)
                cv2.rectangle(frame, (0, (i * 35) + 5),
                (w, (i * 35) + 35), (0, 255, 255), -1)
                cv2.putText(frame, text, (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_DUPLEX, 0.6,
                (0, 0, 255), 2)
                cv2.putText(frame, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
                cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
                              (0, 255, 255), 2)

    cv2.imshow('Image Input', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

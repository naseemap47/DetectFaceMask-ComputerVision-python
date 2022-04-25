import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.models import load_model

face_classifier = cv2.CascadeClassifier(
    '/home/naseem/PycharmProjects/DetectFaceMask-ComputerVision-python/haarcascade_frontalface_default.xml'
)
model = load_model('/home/naseem/PycharmProjects/DetectFaceMask-ComputerVision-python/Model.h5')
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_classifier.detectMultiScale(img_rgb)
    # print(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(
            img, (x, y), (x+w, y+h),
            (0, 255, 0), 2
        )
        face_roi = img_rgb[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (70, 70), interpolation=cv2.INTER_AREA)
        # print(len(face_roi))

        if len(face_roi) != 0:
            roi = face_roi.astype('float32') / 255
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi)[0]
            print(prediction)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

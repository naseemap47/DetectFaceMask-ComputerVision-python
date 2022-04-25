import cv2

face_classifier = cv2.CascadeClassifier(
    '/home/naseem/PycharmProjects/DetectFaceMask-ComputerVision-python/haarcascade_frontalface_default.xml'
)
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

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

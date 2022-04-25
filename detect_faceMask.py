import cv2

face_classifier = cv2.CascadeClassifier(
    '/home/naseem/PycharmProjects/DetectFaceMask-ComputerVision-python/haarcascade_frontalface_default.xml'
)
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_classifier.detectMultiScale(img_rgb)
    print(faces)
    
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

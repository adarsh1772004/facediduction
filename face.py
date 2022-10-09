import cv2 
face_casecade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_casecade=cv2.CascadeClassifier('haarcascade_eye.xml')
video=cv2.VideoCapture(0)
while True:
    ret,frame=video.read()
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_casecade.detectMultiScale(grey, 1.1, 5)
    eyes=eye_casecade.detectMultiScale(grey, 1.1, 5)
    for (x, y, w, h) in faces :
        cv2.rectangle(frame, (x, y),(x+w, y+h),(255, 0, 0), 2)
    for (x, y, w, h) in eyes :
        cv2.rectangle(frame, (x, y),(x+w, y+h),(255, 255, 0), 2)
    cv2.imshow('faces', frame)
    if cv2.waitKey(1)==32:
        break
video.release()
cv2.destroyAllWindows() 
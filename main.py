import cv2
import time

cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 1080)
count = 0

while True:
    ret, frame = cap.read()
    cv2.imshow('Testing webcam', frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:         #ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:        #SPACE pressed
        img_name = "test_{}.png".format(count)
        cv2.imwrite(img_name, frame)
        count += 1

cap.release()
cv2.destroyAllWindows()

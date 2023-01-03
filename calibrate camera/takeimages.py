import cv2

cap1 = cv2.VideoCapture(0)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
cap2 = cv2.VideoCapture(2)

ret, frame = cap1.read()
cv2.imwrite('one.jpg', frame)
print(frame.shape)

ret, frame = cap2.read()
cv2.imwrite('two.jpg', frame)
print(frame.shape)
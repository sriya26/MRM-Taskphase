import cv2
import numpy as np
import torch


load_from_sys = True

if load_from_sys:
	hsv_value = np.load('hsv_value.npy')

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

model = torch.load('mnist.pt')

kernel = np.ones((5,5), np.uint8)

canvas = None

x1 = 0
y1 = 0

noise_thresh = 800

while True:
	_, frame = cap.read()
	frame = cv2.flip(frame, 1)

	if canvas is None:
		canvas = np.zeros_like(frame)

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	if load_from_sys:
		lower_range = hsv_value[0]
		upper_range = hsv_value[1]

	mask = cv2.inRange(hsv,lower_range, upper_range)

	mask = cv2.erode(mask, kernel, iterations = 1)
	mask = cv2.dilate(mask, kernel, iterations = 2)

	contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if contours  and cv2.contourArea(max(contours, key = cv2.contourArea)) > noise_thresh:
		c = max(contours, key = cv2.contourArea)
		x2, y2 ,w, h = cv2.boundingRect(c)

		if x1 == 0 and y1 == 0:
			x1,y1 = x2,y2
		else:
			canvas = cv2.line(canvas, (x1,y1), (x2,y2), [0,255,255], 4)

		x1,y1 = x2,y2
	
	else:
		x1,y1 = 0, 0

	frame = cv2.add(frame, canvas)

	stacked = np.hstack((canvas, frame))
	cv2.imshow('Screen_Pen', cv2.resize(stacked, None, fx = 0.6, fy = 0.6))

	if cv2.waitKey(1) == ord('e'):
		break

	newmask = cv2.resize(mask, (28, 28))
	newmask = np.array(newmask)
	dig = model.predict(newmask.reshape(1, 28, 28, 1))
	dig = np.argmax(dig)
	cv2.putText(canvas, str(dig), (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255), 1)

	#Clear the canvas when 'c' is pressed
	if cv2.waitKey(1) & 0xFF == ord('c'):
		canvas = None

cv2.destroyAllWindows()
cap.release()
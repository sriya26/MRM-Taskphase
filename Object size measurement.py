from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2



image = cv2.imread("sample.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 0)
kernel = np.ones((5,5))
edge = cv2.Canny(blur, 50, 100)
edge = cv2.dilate(edge, None, iterations=1)
edge = cv2.erode(edge, None, iterations=1)



conts = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
conts = imutils.grab_contours(conts)

(conts, _) = contours.sort_contours(conts)

conts = [x for x in conts if cv2.contourArea(x) > 100]


ref = conts[0]
box = cv2.minAreaRect(ref)
box = cv2.boxPoints(box)
box = np.array(box, dtype="int")
box = perspective.order_points(box)
(l1, r1, r2, l2) = box
dist_pixel = euclidean(l1, r1)
dist_cm = 2
pixel_per_cm = dist_pixel/dist_cm

for cont in conts:
	box = cv2.minAreaRect(cont)
	box = cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	box = perspective.order_points(box)
	(l1, r1, r2, l2) = box
	cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
	midpt_h = (l1[0] + int(abs(r1[0] - l1[0])/2), l1[1] + int(abs(r1[1] - l1[1])/2))
	midpt_v = (r1[0] + int(abs(r1[0] - r2[0])/2), r1[1] + int(abs(r2[1] - r2[1])/2))
	w = euclidean(l1, r1)/pixel_per_cm
	h = euclidean(r1, r2)/pixel_per_cm
	cv2.putText(image, "{:.1f}cm".format(w), (int(midpt_h[0] - 15), int(midpt_h[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
	cv2.putText(image, "{:.1f}cm".format(h), (int(midpt_v[0] + 10), int(midpt_v[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
import cv2
import numpy as np

src = cv2.imread("./Template_Matching/017A8500_HIGH.JPG")
if src is None: raise Exception("Image Load Error")
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
target = cv2.imread("./Template_Matching/017A8500_Template.jpg", cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(
    nfeatures=40000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=20,
)

kp1, des1 = orb.detectAndCompute(gray, None)
kp2, des2 = orb.detectAndCompute(target, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

for i in matches[:100]:
    idx = i.queryIdx
    x1, y1 = kp1[idx].pt
    cv2.circle(src, (int(x1), int(y1)), 3, (255, 0, 0), 3)

cv2.namedWindow("src", cv2.WINDOW_NORMAL)
cv2.imshow("src", src)
cv2.waitKey()
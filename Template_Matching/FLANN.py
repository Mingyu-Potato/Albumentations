import numpy as np, cv2

def FLANN(factor):
    img  = cv2.imread("./Template_Matching/017A8500_HIGH.JPG")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread("./Template_Matching/017A8500_Template.jpg", cv2.IMREAD_GRAYSCALE)
    res = None

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray, None)
    kp2, des2 = sift.detectAndCompute(template, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < factor * n.distance:
            good.append(m)

    res = cv2.drawMatches(img, kp1, template, kp2, good, res, flags=2)

    cv2.namedWindow("Feature Matching", cv2.WINDOW_NORMAL)
    cv2.imshow("Feature Matching", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

FLANN(0.7)
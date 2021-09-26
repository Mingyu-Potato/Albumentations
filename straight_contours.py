import numpy as np, os, glob, cv2, math

# 허프변환 검출 직선 그리기함수
def draw_houghLines(src, lines, nline):
    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    min_length = min(len(lines), nline)

    for i in range(min_length):
        rho, radian = lines[i, 0, 0:2]
        a, b = math.cos(radian), math.sin(radian)
        pt = (a * rho, b * rho)
        delta = (-1000 * b, 1000 * a)
        pt1 = np.add(pt, delta).astype('int')
        pt2 = np.subtract(pt, delta).astype('int')
        cv2.line(dst, tuple(pt1), tuple(pt2), (0, 255, 0), 2, cv2.LINE_AA)

    return dst

# hough transform line detection
def hough_detection(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(image_gray, (3, 3), 0) # 블러링
    
    # 모폴로지
    open_mask = np.array([[0,1,0],
                        [1,1,1],
                        [0,1,0]]).astype('uint8')
    open_mor = cv2.morphologyEx(blur, cv2.MORPH_OPEN, open_mask, iterations=10)

    close_mask = np.array([[0,1,0], 
                        [1,1,1],
                        [0,0,0]]).astype('uint8')
    mor = cv2.morphologyEx(open_mor, cv2.MORPH_CLOSE, close_mask, iterations=10)
    
    # 캐니 에지 검출(hard)
    canny = cv2.Canny(mor, 25, 28, 3)

    # 허프 변환
    lines = cv2.HoughLines(canny[230:420, 0:400], 1, np.pi/180, 10)
    hough = draw_houghLines(canny, lines, 2)

    return canny, hough


# contours detection
def contours_detection(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blurring
    blur = cv2.GaussianBlur(image_gray, (3, 3), 10)
    
    # morpology
    open_mask = np.array([[0,1,0],
                        [1,1,1],
                        [0,1,0]]).astype('uint8')
    open_mor = cv2.morphologyEx(blur, cv2.MORPH_OPEN, open_mask, iterations=10)

    close_mask = np.array([[0,1,0],
                        [1,1,1],
                        [0,0,0]]).astype('uint8')
    mor = cv2.morphologyEx(open_mor, cv2.MORPH_CLOSE, close_mask, iterations=10)
    
    # canny edges(hard code)
    canny = cv2.Canny(mor, 25, 28, 7)

    # contours extractionw
    ret, img_binary = cv2.threshold(canny, 127, 255, 0)
    contours, hierachy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)

    return image






\


# 이미지(color, gray) 불러오기
# image = cv2.imread("./SFR-03(Abnormal)/Albumentation_dataset/HARD_5.JPG")
image = cv2.imread("./017A8500.JPG")
if image is None: raise Exception("영상 읽기오류")

image = cv2.resize(image, dsize=(1024, 683), interpolation=cv2.INTER_LINEAR)

cv2.imshow("image", image)

# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# (x,y), (w, h) = (230,0), (190, 400)
# roi_img = image_gray[x:x+w, y:y+h]

# # # for row in roi_img:
# # #     for p in row:
# # #         print(p, end="  ")
# # #     print()

# cv2.rectangle(image_gray, (x,y,w,h), 255, 1)
# cv2.imshow("a", image_gray)
# cv2.waitKey(0)

canny, hough = hough_detection(image)
contours = contours_detection(image)

# _, g1, _ = cv2.split(hough)

# zr1 = np.zeros(shape=g1.shape, dtype=np.uint8)

# green_hough = cv2.merge((zr1, g1, zr1))

# _,g2,_ = cv2.split(contours)
# zr2 = np.zeros(shape=g2.shape, dtype=np.uint8)

# green_contours = cv2.merge((zr2, g2, zr2))

# cv2.imshow("canny", canny)
cv2.imshow("straight", hough)
cv2.imshow("contours", contours)
cv2.waitKey(0)
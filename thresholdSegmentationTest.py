import cv2
import numpy as np

def colorInversion(img):
    dst = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            gray = img[i, j]
            gray = int(gray)
            grayy = 255 - gray
            dst[i, j] = np.uint8(grayy)
    return dst
def unevenLightCompensate(img, blockSize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (3, 3), 0)
    # dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    return dst

def erode_demo(pic):
    #gray = cv2.cvtColor(pic,cv2.COLOR_RGB2GRAY) # 原图片类型转换为灰度图像
    #ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("binary", binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    dst = cv2.erode(pic, kernel)
    return dst

def dilate_demo(binary):
    #gray = cv2.cvtColor(pic,cv2.COLOR_RGB2GRAY)
    #ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("binary", binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    dst = cv2.dilate(binary, kernel)
    return dst

# Step1. Read image
# image = cv2.imread('data/1610425410.jpg')   # C1.0
# image = cv2.imread('data/1610426705.jpg')   # C0.5
# image = cv2.imread('data/1610429905.jpg')   # B1.0
# image = cv2.imread('data/1610435703.jpg')   # C1.5
# image = cv2.imread('data/1610437001.jpg')   # C1.0
# image = cv2.imread('data/1630448004.jpg')   # C1.0
# image = cv2.imread('data/1810861705.jpg')
# image = cv2.imread('data/1810861705.jpg')
image = cv2.imread('data/1710230204w.jpg')

# Step2. Preprogress
img_ori = image.copy()
# image = cv2.resize(image,(9000,2000))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (5,5), 0)
# image = unevenLightCompensate(image,blockSize=5)

# Step3. 运行封装OTSU函数并输出灰度化后的阈值
ret, result = cv2.threshold(image, 0, 256, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# ret, result = cv2.threshold(image, 0, 256, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
print(ret)

# Step4. 开运算：先腐蚀后膨胀，可消除细小物体或断开两个区域间的细小连接处。
result = erode_demo(result)
# result = cv2.GaussianBlur(result, (9,9), 0)
# result = colorInversion(result)


# Step5. Find contours 阈值处理后
# cnts = cv2.findContours(result.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cv2.findContours(result.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cv2.findContours(result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cv2.findContours(result.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# Obtain area for each contour
contour_sizes = [(cv2.contourArea(contour), contour) for contour in cnts]


# th2 = np.zeros(image.shape)
# for i in range(image.shape[0]):
#     for j in range(image.shape[1]):
#         if(data[i][j] > ret2):
#             th2[i][j] = 256
#         else:
#             th2[i][j] = 0
#Step 6 结果输出

cv2.namedWindow("out",0);#可调大小
cv2.resizeWindow('out', 1800, 400)
cv2.namedWindow("img",0);#可调大小
cv2.resizeWindow('img', 1800, 400)
# cv2.namedWindow("1",0);#可调大小
# cv2.resizeWindow('1', 1800, 400)

# cv2.imshow('1', img)
cv2.imshow('out', result)


# Find maximum contour and crop for ROI section
if len(contour_sizes) > 0:
    # contour_sizes.sort(key = lambda x: x[0])
    # largest_contour=contour_sizes[-1][1]
    largest_contour=max(contour_sizes,key = lambda x: x[0])[1]
    x,y,w,h = cv2.boundingRect(largest_contour) # 获取包含对象轮廓的最小矩形

    img = cv2.rectangle(img_ori, (x, y), (x + w, y + h), (0,0,255), 10)
    ROI = img_ori[y:y+h, x:x+w]
    cv2.namedWindow("ROI", 0);  # 可调大小
    cv2.resizeWindow('ROI', 1800, 400)
    cv2.imshow("ROI", ROI)
cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

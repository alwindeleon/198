import cv2
import numpy as np
from matplotlib import pyplot as plt

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
MIN_MATCH_COUNT = 10


img1 = cv2.imread('image_1.png', 0)
img2 = cv2.imread('image_2.png', 0)

# gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create(MAX_FEATURES)
# kp = sift.detect(gray,None)
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test. RANSAC
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    # h,w,l= img1.shape
    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # dst = cv2.perspectiveTransform(pts,M)

    # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    # disparity_left  = cv2.CreateMat(img1.height, img1.width, cv2.CV_16S)
    # disparity_right = cv2.CreateMat(img1.height, img1.width, cv2.CV_16S)
    # state = cv2.CreateStereoGCState(16,2)
    # cv2.FindStereoCorrespondenceGC(left,right,
    #                       disparity_left,disparity_right,state)
    stereo = cv2.StereoSGBM_create(numDisparities=48, blockSize=21)
    disparity = stereo.compute(img1,img2).astype(np.float32) / 16.0
    plt.imshow(disparity,'gray')
    plt.show()

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None

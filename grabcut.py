import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('J854/L/J854_fdrect_fd14_018588.png')
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (250,500,1600,1200)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
cv.imwrite('masked-3.png',img)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)),plt.show()

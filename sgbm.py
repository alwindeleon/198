import numpy as np
from sklearn.preprocessing import normalize
import cv2
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

print('loading images...')
# imgL = cv2.pyrDown(cv2.imread('J854/L/J854_fdrect_fd14_018588.png') ) # downscale images for faster processing if you like
# imgR = cv2.pyrDown(cv2.imread('J854/R/J854_fdrect_fd15_018588.png') )
imgL = cv2.pyrDown(cv2.imread('masked-2.png') ) # downscale images for faster processing if you like
imgR = cv2.pyrDown(cv2.imread('masked-1.png') )

# meshlab params
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


# SGBM Parameters -----------------
window_size = 5                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
 
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=160,            # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=2,
    P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

# FILTER Parameters
lmbda = 8000
sigma = 1.2
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)


print('computing disparity...')
displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
displ = np.int16(displ)
dispr = np.int16(dispr)
filteredImg = wls_filter.filter(displ, imgL, None, dispr)  

filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
filteredImg = np.uint8(filteredImg)

print('generating 3d point cloud...')
h, w = imgL.shape[:2]
f = 0.8*w                          # guess for focal length
Q = np.float32([[1, 0, 0, -0.5*w],
                [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                [0, 0, 0,     -f], # so that y-axis looks up
                [0, 0, 1,      0]])
points = cv2.reprojectImageTo3D(filteredImg, Q)

colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
mask = filteredImg > filteredImg.min()

x = points[mask][:,0]

y = points[mask][:,1]
print len(x), len(y)
# z = points[mask][:,2]
z = np.array([x,y])
# X, Y = np.meshgrid(x, y, sparse=True)
Z = z

# print z
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# # ax.plot_wireframe(x,y,z)
# ax.plot_surface(x-x.mean(), y-y.mean(), z, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
# print('surface done');
# ax.set_title('surface');
# plt.show()


out_points = points[mask]
out_colors = colors[mask]
out_fn = 'out.ply'
write_ply('out.ply', out_points, out_colors)
cv2.imwrite('disp_map.png',filteredImg)
cv2.imshow('Disparity Map', filteredImg)
cv2.waitKey()
cv2.destroyAllWindows()
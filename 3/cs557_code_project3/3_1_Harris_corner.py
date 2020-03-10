
import numpy as np
from matplotlib import pyplot as plt

import cv2
from scipy.ndimage.filters import convolve

#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/Lenna.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/building.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/hinge.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/hinges.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/keys.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/pillsetc.pnm')
img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/checkboard.png')

img = cv2.GaussianBlur(img,(5,5),1)


img_copy = img.copy()

# Sobel x-axis kernel
Dx = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int32")

# Sobel y-axis kernel
Dy = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int32")

# Gaussian kernel
Gauss = np.array((
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]), dtype="float64")

# Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

"""
Step 1. Image derivative #SLIDE 34
"""

dx = convolve(gray, Dx)
dy = convolve(gray, Dy)

"""
Step 2. Sq. of derivative
"""

dx2 = np.square(dx)
dy2 = np.square(dy)
dxdy = dx * dy

"""
Step 3. Gaussian filter
"""

g_Ix2 = convolve(dx2, Gauss)
g_Iy2 = convolve(dy2, Gauss)
g_IxIy = convolve(dxdy, Gauss)


"""
Step 4. Cornerness function
"""

alpha = 0.05

harris = g_Ix2*g_Iy2 - np.square(g_IxIy) - alpha*np.square(g_Ix2 + g_Iy2)
harris = cv2.normalize(harris, harris, 0, 1, cv2.NORM_MINMAX) # normalize value between 0-1


"""
Step 5. Non maxima supression
"""
#Find points with large corner response: R>threshold
#result is dilated for marking the corners, not important
# Threshold for an optimal value, it may vary depending on the image.

block_size = 2

def nms (img, block_size):

    img_nms = np.zeros(img.shape)

    M, N = img.shape

    for i in range(block_size, M - block_size):
            for j in range(block_size, N - block_size):

                # Find img_thresh[i,j] which is either 0 or img_thresh[i,j] in case of local maxima
                localMaxima = img[i - block_size][j - block_size]

                for a in range(i - block_size, i + block_size + 1):
                    for b in range(j - block_size, j + block_size + 1):
                        #print (img_thresh[a,b])
                        if img[a, b] > localMaxima:
                            localMaxima = img[a, b]

                if img[i, j] == localMaxima:
                    img_nms[i, j] = localMaxima
    return img_nms

nms_img = nms(harris, 2)


"""
Step 6. Thresholding
"""


#change threshold values according to images
threshold = .9

nms_img = cv2.dilate(nms_img,None)
# find all points above threshold
loc = np.where(nms_img >= threshold)
# loop though the points
for pt in zip(*loc[::-1]):
    # draw filled circle on each poiny
    cv2.circle(img_copy, pt, 1, (255, 0, 0), -1)

'''
OpenCV implementation
'''
dst = cv2.cornerHarris(gray,2,3,alpha)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
#cv2.imshow('cv2.cornerharris dilate',dst)
dst = cv2.normalize(dst, dst, 0, 1, cv2.NORM_MINMAX) # normalize value between 0-1

loc = np.where(dst >= threshold)
# loop though the points
for pt in zip(*loc[::-1]):
    # draw filled circle on each poiny
    cv2.circle(img, pt, 1, (255, 0, 0), -1)



"""
plotting
"""
plt.subplot(1,3,1)
plt.imshow(harris)
plt.title('Corner response function R')

plt.subplot(1,3,2)
plt.imshow(img_copy)
plt.title('Harris Code,'
          'thresh = 0.9')

plt.subplot(1,3,3)
plt.imshow(img)
plt.title('OpenCV, thresh = 0.9')


plt.show()
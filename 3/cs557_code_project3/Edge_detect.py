import scipy.ndimage as ndi
import numpy as np
from matplotlib import pyplot as plt
from math import pi,degrees
import cv2
from scipy.ndimage.filters import convolve, gaussian_filter

#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/Lenna.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/building.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/hinge.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/hinges.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/keys.pnm')
img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/pillsetc.pnm')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
"""
Step 1. Filter image with x and y derivative of gaussian
"""
'''def gauss2dfilter(sigma):
    length = 5*sigma# rule of thumb approximation
    ax = np.arange(-length // 2 + 1., length // 2 + 1.)
    #print(ax)
    xx, yy = np.meshgrid(ax, ax)
    #print(xx)
    kernel = (np.exp(-0.5 * (np.square(xx) + np.square(yy)) / (2*np.square(sigma))))/(2*pi*np.square(sigma))
    #print(kernel)
    kernel2d = kernel / np.sum(kernel)
    #print(kernel2d)
    #plt.imshow(kernel2d, interpolation='none')
    #plt.title('Gaussian kernel')
    #plt.show()
    return kernel2d

"""Gaussian filter"""

kernel2d = gauss2dfilter(sigma=1)

"""first order derivative of a Gaussian in x-direction"""

gauss_dx=ndi.gaussian_filter(kernel2d,sigma=1,order=[0,1],output=np.float64, mode='nearest')
#plt.imshow(gauss_dx, interpolation='none')
#plt.title('Gaussian kernel dx')
#plt.show()

"""first order derivative of a Gaussian in y-direction"""

gauss_dy=ndi.gaussian_filter(kernel2d,sigma=1,order=[1,0],output=np.float64, mode='nearest')

#plt.imshow(gauss_dy, interpolation='none')
#plt.title('Gaussian kernel dy')
#plt.show()

"""Convolve with image"""
Ix = ndi.filters.convolve(img, gauss_dx)
#Ix = (Ix/Ix.max()) * 255
Ix = np.uint8(Ix)
#plt.imshow(Ix, interpolation='bicubic', cmap = 'gray')
#plt.title('Ix')
#plt.show()

Iy = ndi.filters.convolve(img, gauss_dy)
#Iy = (Iy/Iy.max()) * 255
Iy = np.uint8(Iy)
#plt.imshow(Iy, interpolation='bicubic', cmap = 'gray')
#plt.title('Iy')
#plt.show()'''

im = np.array(img, dtype=float)  # Convert to float to prevent clipping values

# Gaussian blur to reduce noise
im2 = gaussian_filter(im,1) #sigma = 1


# Use sobel filters to get horizontal and vertical gradients
Ix = convolve(im2, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Iy = convolve(im2, [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])




"""
Step 2. Find Magnitude and Orientation of gradient
"""

magnitude = np.hypot(Ix, Iy)
#magnitude = (magnitude/magnitude.max()) * 255
magnitude = np.uint8(magnitude)

#plt.imshow(magnitude, interpolation='bicubic', cmap = 'gray')
#plt.title('Image magnitude')
#plt.show()


theta = np.arctan2(Iy, Ix) #slide 75
#theta = (theta/theta.max()) * 255
theta = np.uint8(theta)

#plt.imshow(theta, interpolation='bicubic', cmap = 'gray')
#plt.title('Image orientation')
#plt.show()

"""
Step 3. Non maximum supression
"""
def round_angle(angle):
    """ Input must be each pixel value of theta"""
    angle = degrees(angle) % 180 #divide by 180 deg and output reminder
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        angle = 0
    elif (22.5 <= angle < 67.5):
        angle = 45
    elif (67.5 <= angle < 112.5):
        angle = 90
    elif (112.5 <= angle < 157.5):
        angle = 135
    return angle

M, N = magnitude.shape
Z = np.zeros((M,N), dtype=np.int32) #create a black image of same shape of magnitude (where we apply non max suppress)
for i in range(M):
        for j in range(N):
            # Suppress pixels at the image edge
            if i == 0 or i == M - 1 or j == 0 or j == N - 1:
                Z[i, j] = 0
                continue
            # find neighbour pixels to visit from the gradient directions
            location = round_angle(theta[i, j])
            #print(location)
            """
            | i-1, j-1 | i, j-1 | i+1, j-1 |
            | i-1, j   | i, j   | i+1, j   |
            | i-1, j+1 | i, j+1 | i+1, j+1 |
            supression done perpendicular to edge direction
            output angle we get from "round_angle(theta[i, j]", add 90 deg to that
            start from (i,j) : see which pixel is indicated by "round_angle(theta[i, j]+90"
            same logic apply foe the opposite pixel
            """
            if location == 0:
                if (magnitude[i, j] >= magnitude[i, j - 1]) and (magnitude[i, j] >= magnitude[i, j + 1]):
                    Z[i, j] = magnitude[i, j]

            elif location == 90:
                if (magnitude[i, j] >= magnitude[i - 1, j]) and (magnitude[i, j] >= magnitude[i + 1, j]):
                    Z[i, j] = magnitude[i, j]

            elif location == 135:
                if (magnitude[i, j] >= magnitude[i - 1, j - 1]) and (magnitude[i, j] >= magnitude[i + 1, j + 1]):
                    Z[i, j] = magnitude[i, j]

            elif location == 45:
                if (magnitude[i, j] >= magnitude[i - 1, j + 1]) and (magnitude[i, j] >= magnitude[i + 1, j - 1]):
                    Z[i, j] = magnitude[i, j]

#print(Z)
Non_max_sup = Z
#plt.imshow(Z, interpolation='bicubic', cmap = 'gray')
#plt.title('Non-max Supression')
#plt.show()

"""
Step 4. Thresholding and linking
"""

Upper_thresh = 255
Lower_thresh = 50

high_i, high_j = np.where(Z > Upper_thresh)

low_i, low_j = np.where((Z <= Upper_thresh) & (Z >= Lower_thresh))

zero_i, zero_j = np.where(Z < Lower_thresh)

# set values
Z[high_i, high_j] = np.int32(Upper_thresh)
Z[low_i, low_j] = np.int32(Lower_thresh)
Z[zero_i, zero_j] = np.int32(0)

#plt.imshow(Z, interpolation='bicubic', cmap = 'gray')
#plt.title('Thresholding')
#plt.show()


"""
Start at a pixel with lower threshold, if 
                                        connected to Upeer threshold
                                            keep
                                        else:
                                            delete
"""

M, N = Z.shape
for i in range(M):
    for j in range(N):
        # Suppress pixels at the image edge
        if i == 0 or i == M - 1 or j == 0 or j == N - 1:
            Z[i, j] = 0
            continue
        if img[i, j] == Lower_thresh:
            Upper_thresh = 255
            # check if one of the neighbours is strong (=255 by default)
            if ((Z[i + 1, j] == Upper_thresh) or (Z[i - 1, j] == Upper_thresh)
                    or (Z[i, j + 1] == Upper_thresh) or (Z[i, j - 1] == Upper_thresh)
                    or (Z[i + 1, j + 1] == Upper_thresh) or (Z[i - 1, j - 1] == Upper_thresh)):
                Z[i, j] = Upper_thresh
            else:
                Z[i, j] = 0

Canny_detect = Z
#plt.imshow(Canny_detect, cmap = 'gray')
#plt.title('Canny Edge detection')
#plt.show()

"""
OpenCV implementation
"""
CV_canny = cv2.Canny(img,Lower_thresh,Upper_thresh)
#plt.imshow(CV_canny, cmap = 'gray')
#plt.title('OpenCV Canny Implementaion')
#plt.show()

"""
Plot
"""
plt.subplot(231)
plt.imshow(img,cmap = 'gray')
plt.title('Original Image')

plt.subplot(232)
plt.imshow(magnitude, interpolation='bicubic', cmap = 'gray')
plt.title('Image magnitude')

plt.subplot(233)
plt.imshow(theta, interpolation='bicubic', cmap = 'gray')
plt.title('Image orientation')

plt.subplot(234)
plt.imshow(Non_max_sup, interpolation='bicubic', cmap = 'gray')
plt.title('Non-max Supression')

plt.subplot(235)
plt.imshow(Canny_detect, cmap = 'gray')
plt.title('Applied Canny')

plt.subplot(236)
plt.imshow(CV_canny, cmap = 'gray')
plt.title('OpenCV Canny')

plt.show()


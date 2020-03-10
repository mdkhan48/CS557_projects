import math
import numpy as np
from matplotlib import pyplot as plt
from math import pi,degrees
import cv2
from scipy.ndimage.filters import convolve, gaussian_filter


#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/Lenna.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/building.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/hinge.pnm')
img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/hinges.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/keys.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/pillsetc.pnm')

img1 = img.copy()
img2 = img.copy()

rows, cols, channels = img.shape
img = cv2.GaussianBlur(img,(5,5),1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
magnitude = np.uint8(magnitude)

theta = np.arctan2(Iy, Ix) #slide 75
theta = np.uint8(theta)


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

Non_max_sup = Z


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
            #Upper_thresh = 255
            # check if one of the neighbours is strong (=255 by default)
            if ((Z[i + 1, j] == Upper_thresh) or (Z[i - 1, j] == Upper_thresh)
                or (Z[i, j + 1] == Upper_thresh) or (Z[i, j - 1] == Upper_thresh)
                or (Z[i + 1, j + 1] == Upper_thresh) or (Z[i - 1, j - 1] == Upper_thresh)):
                Z[i, j] = Upper_thresh
            else:
                Z[i, j] = 0

edges = Z
edges = np.uint8(edges)
print(edges.max())
"""
Hough transform implementation
"""

"""
Step.1 Gradient orientation of edge 
"""

gradient = np.arctan2(Iy, Ix) * 180/np.pi #degree

"""
step.2 Creating table and plotting accumulator H
"""
angle_resolution = 1

thetas = np.deg2rad(np.arange(-180.0, 180.0, angle_resolution))

y_length, x_length = gray.shape

max_distance = int(math.ceil(math.sqrt(y_length*y_length + x_length*x_length)))

rhos = np.linspace(-max_distance, max_distance, max_distance * 2)

number_of_thetas = len(thetas)

'''
Algorithm:

Initialize accumulator H to all zeros

For each edge point (x,y) in the image
    For θ = 0 to 180
        ρ = x cos θ + y sin θ
        H(θ, ρ) = H(θ, ρ) + 1
    end
end

Find the value(s) of (θ, ρ) where H(θ, ρ) is a local maximum
    The detected line in the image is given by ρ = x cos θ + y sin θ
'''
"""
Incorporating image gradients

Modified Hough transform:

For each edge point (x,y)
    θ = gradient orientation at (x,y)
    ρ = x cos θ + y sin θ
    H(θ, ρ) = H(θ, ρ) + 1
end

theta in this case is a float, but for H input, theta has to be integer. 
Unable to figure out the data structure for theta input in modified case.

"""

#Hough space is the accumulator which contains votes
#x-axis theta, y-axis rho
hough_space = np.zeros((2*max_distance, number_of_thetas),dtype=np.uint8)



cos_thetas = np.cos(thetas)
sin_thetas = np.sin(thetas)

for y in range(y_length):
	for x in range(x_length):
		if edges[y][x] == 50: #looking for white pixels in image
			for i_theta in range(number_of_thetas):
				pho = int(math.ceil(x*cos_thetas[i_theta] + y*sin_thetas[i_theta]))
				hough_space[max_distance + pho][i_theta] += 1 #adding votes in hough space



"""
writing lines on image : backprojection
following the steps indicated in OpenCV
"""
max_bin_size = 20

for i_pho in range(hough_space.shape[0]):
			for i_theta in range(hough_space.shape[1]):
				if hough_space[i_pho][i_theta] >= max_bin_size:
					theta = thetas[i_theta]
					pho = i_pho - max_distance
					cos = math.cos(theta)
					sin = math.sin(theta)
					x0 = pho * cos
					y0 = pho * sin
					x1 = int(x0 + max_distance * sin)
					y1 = int(y0 - max_distance * cos)
					x2 = int(x0 - max_distance * sin)
					y2 = int(y0 + max_distance * cos)
					cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 1)






"""
OpenCV Implementation
"""

#  Standard Hough Line Transform

'''dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)

lines: A vector that will store the parameters (r,θ) of the detected lines

rho : The resolution of the parameter r in pixels. We use 1 pixel.

theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)

threshold: The minimum number of intersections to "*detect*" a line :: binsize=50

srn and stn: Default parameters to zero. Check OpenCV reference for more info.'''

lines = cv2.HoughLines(edges,1,np.pi/180,max_bin_size)


#print(lines.shape)
maxLength = np.sqrt(rows ** 2 + cols ** 2)

for rho, theta in lines[:, 0]:
    cos = np.cos(theta)
    sin = np.sin(theta)
    x0 = rho * cos
    y0 = rho * sin
    x1 = int(x0 + maxLength * sin)
    y1 = int(y0 - maxLength * cos)
    x2 = int(x0 - maxLength * sin)
    y2 = int(y0 + maxLength * cos)
    cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 1)

"""
Plot
"""
plt.subplot(221)
plt.imshow(edges,cmap = 'gray')
plt.title('Canny Edges')

plt.subplot(222)
plt.imshow(hough_space,cmap='jet',extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[0], rhos[-1]])
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Hough Space (cumulator)')

plt.subplot(223)
plt.imshow(img1)
plt.title('Implemented Hough Transform, bin size = 20')

plt.subplot(224)
plt.imshow(img2)
plt.title('OpenCV Hough Transform, bin size = 20')

plt.show()
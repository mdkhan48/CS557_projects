import numpy as np
import matplotlib.pyplot as plt
import cv2
import mahotas
from numpy import NaN, Inf, arange, isscalar, asarray, array

img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/hinge.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/hinges.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/shapes1.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/shapes2.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/shapes3.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/shapes4.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/keys.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/pillsetc.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/coins.pnm')

print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = gray.shape

#calculating histogram of gray image
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# smoothing histogram
avg_mask = np.array([1 / 9, 2 / 9, 3 / 9, 2 / 9, 1 / 9])
smooth_hist = np.convolve(hist[:, 0], avg_mask)


# peak is the highest point betweem "valleys". What makes a peak is the fact that there are lower points around it
#Look for the highest point, around which there are points lower by X on both sides.
#for minima, there are higher points around it
#http://billauer.co.il/peakdet.html
delta = 10
max_value = []
min_value = []

x = arange(len(smooth_hist))
v = asarray(smooth_hist)

min, max = Inf, -Inf
min_position, max_position = NaN, NaN

lookformax = True

for i in arange(len(v)):
    this = v[i]
    if this > max:
        max = this
        max_position = x[i]
    if this < min:
        min = this
        min_position = x[i]

    if lookformax:
        if this < max - delta:
            max_value.append((max_position, max))
            min = this
            min_position = x[i]
            lookformax = False
    else:
        if this > min + delta:
            min_value.append((min_position, min))
            max = this
            max_position = x[i]
            lookformax = True


#max_value contains the [position value] of the maximum points
#min_value contains the [position value] of the minimum points

plt.plot(smooth_hist)
plt.scatter(array(max_value)[:,0], array(max_value)[:,1], color='blue')
plt.scatter(array(min_value)[:,0], array(min_value)[:,1], color='red')
plt.title('Original Smooth histogram with Max-Min points') # original histogram
plt.show()

min_value_x = array(min_value)[:,0] # posistion of min points
min_value_y = array(min_value)[:,1] #value of min points
threshold_value = np.min(array(min_value)[:,1]) #finding minimum of min values
threshold_position = min_value_x[np.isclose(min_value_y, threshold_value)] #finding positition for the minimum of min values
threshold = threshold_position[0]
threshold = int(threshold)
print("Implemented threshold", threshold)
segment = gray.copy()
segment[segment > threshold] = 255
segment[segment < threshold] = 0
segment = cv2.bitwise_not(segment)


#Otsu's method is used for thresholding
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
T = mahotas.thresholding.otsu(blurred)
print('Otsu’s threshold: %d' % T)
thresh = gray.copy()
thresh[thresh > T] = 255
thresh[thresh < T] = 0
thresh = cv2.bitwise_not(thresh)


#plotting implemented and otsu segmentation
plt.subplot(131)
plt.imshow(gray)
plt.title('original image')
plt.set_cmap('gray')

plt.subplot(132)
plt.imshow(segment)
plt.title('Implemented segmentation with threshold =%i' %threshold)

plt.subplot(133)
plt.imshow(thresh)
plt.title('Otsu segmentation with threshold =%i' %T)

plt.show()




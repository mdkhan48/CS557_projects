import cv2
import numpy as np
import matplotlib.pyplot as plt

#img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/auto.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/building.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/child.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/ct_scan.pnm')
img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/tire.pnm')


img_shape = img.shape
#print (img_shape)

#converted to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_shape = gray.shape
#print (gray_shape)
#cv2.imshow('Original image', img)
#cv2.imshow('Gray image', gray)
height, width = gray.shape

#function to calculate normalized histogram
def norm_hist(img):
    height, width = img.shape
    h = [0.0] * 256
    for i in range(height):
	    for j in range(width):
	        h[img[i, j]]+=1
    return np.array(h)/(height*width)

#original histogram of gray image
norm_h = norm_hist(gray)
#print(norm_h)
c_I = np.cumsum(norm_h)
#print(c_I)
convert = np.uint8(255 * c_I) #converting c_i values to 0-255
blank_im = np.zeros_like(gray) #creating a blank image of same size as gray image

# applying transfered values for each pixels
for i in range(0, height):
	for j in range(0, width):
		blank_im[i, j] = convert[gray[i, j]]

converted_im = blank_im #eq. hist. values written in blank image
new_h = norm_hist(converted_im) #new image histogram


histeq_openCV = cv2.equalizeHist(gray)
#cv2.imwrite('result.jpg',histeq_openCV)
#cv2.imshow('openCV',histeq_openCV)

#plotting images
plt.subplot(131)
plt.imshow(gray)
plt.title('original image')
plt.set_cmap('gray')
# show original image
plt.subplot(132)
plt.imshow(converted_im)
plt.title('Hist. Eq. image')
plt.set_cmap('gray')
#Show OpenCV image
plt.subplot(133)
plt.imshow(histeq_openCV)
plt.title('OpenCV image')
plt.set_cmap('gray')
plt.show()

# plot histograms
fig = plt.figure()
fig.add_subplot(131)
plt.hist(gray.flatten(),256,[0,256], color = 'g')
#plt.plot(norm_h)
plt.title('Original histogram') # original histogram

fig.add_subplot(132)
#plt.plot(new_h)
plt.hist(converted_im.flatten(),256,[0,256], color = 'b')
plt.title('Converted histogram') #hist of eqlauized image

fig.add_subplot(133)
plt.hist(histeq_openCV.flatten(),256,[0,256], color = 'r')
plt.title('OpenCV histogram') #hist of eqlauized image
#bottom, top = plt.ylim()

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

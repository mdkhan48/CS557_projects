import numpy as np
import matplotlib.pyplot as plt
import cv2
import mahotas
from numpy import pi

img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/hinge.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/hinges.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/shapes1.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/shapes2.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/shapes3.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/shapes4.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/keys.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/pillsetc.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/coins.pnm')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = gray.shape

#Otsu's method is used for thresholding
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
T = mahotas.thresholding.otsu(blurred)
print('Otsu’s threshold: %d' % T)
thresh = gray.copy()
thresh[thresh >= T] = 255
thresh[thresh < T] = 0
segment = thresh


kernel = np.ones((3,3),np.uint8)
segment = cv2.morphologyEx(segment, cv2.MORPH_OPEN, kernel)


im2, contours, hierarchy = cv2.findContours(segment, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# We know that the input image has n shapes, and lets see if the library could find all these n contours
print("Shapes found : ", len(contours))
#print(contours)
#print(hierarchy)
#plt.imshow(im2)
#plt.show()

# now lets draw these contours
for i in range(len(contours)):
    cnt = contours[i]

    M = cv2.moments(cnt)  # getting the hu moments of each contour
    #print(M)
    cx = int(M['m10'] / M['m00'])  # center x
    cy = int(M['m01'] / M['m00'])  # center y

    print('Moments for shape',i+1)
    center = (cx, cy)
    print('Center', center)
    area = cv2.contourArea(cnt)  # getting the contour area
    print('Area', area)
    #2nd order moments
    print('mu20',M['nu20'])
    print('mu02', M['nu02'])
    print('mu11', M['nu11'])
    perimeter = cv2.arcLength(cnt, True)  # getting the contour perimeter
    print('Perimeter', perimeter)
    compactness = (perimeter**2)/(4*pi*area)#getting compactness
    print('Compactness', compactness)
    #calculate elongation
    gamma =2
    m00_gam = (M['m00'])**gamma
    mu20_bar = (M['nu20'])/m00_gam
    mu02_bar = (M['nu02'])/m00_gam
    mu11_bar = (M['nu11'])/m00_gam

    num = np.sqrt((mu20_bar-mu02_bar)**2 + 4*(mu11_bar)**2)
    den = mu20_bar + mu02_bar + np.sqrt((mu20_bar-mu02_bar)**2 + 4*(mu11_bar)**2)

    elongation = np.sqrt(num/den)
    print('Elogation', elongation)

    # A -> Area P -> Permimeter
    cv2.putText(segment, str(i+1), center,
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (127, 127, 127), 1)

    #cv2.putText(segment, "P: {0:2.1f}".format(perimeter), (cx, cy + 30),
                #cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, (255, 0, 0), 3)

plt.figure(" 4_3")
plt.imshow(segment,cmap='gray')
plt.title('Calculating Moments of Shapes in a Binary Image')
plt.show()



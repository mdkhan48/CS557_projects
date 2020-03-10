import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi
import math

#img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/auto.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/building.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/child.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/ct_scan.pnm')
img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/tire.pnm')
print(img.shape)

def rotate_image(img,angle):
    max_length = int(np.amax(img.shape))
    img = cv2.resize(img,(max_length,max_length))
    height, width, channel = img.shape
    #plt.imshow(img)
    #plt.show()
    #print(max_length)
    blank_im = np.zeros_like(img) #creating a blank image of same size as input image

    angle_rad = -math.radians(angle) # angle in radian
    center_x = width / 2
    center_y = height / 2

    for x in range(width):
        for y in range(height):
            # Compute coordinate in input image
            xp = int((x - center_x) * cos(angle_rad) - (y - center_y) * sin(angle_rad) + center_x)
            yp = int((x - center_x) * sin(angle_rad) + (y - center_y) * cos(angle_rad) + center_y)
            if 0 <= xp < width and 0 <= yp < height:
                blank_im[x, y] = img[xp, yp]
    return blank_im,angle

rotated_image,angle = rotate_image(img,90)

#openCV implementation
height, width, channel = img.shape
M = cv2.getRotationMatrix2D(((width-1)/2.0,(height-1)/2.0),angle,1)
dst = cv2.warpAffine(img,M,(width,height))



#plotting images
plt.subplot(121)
plt.imshow(rotated_image)
plt.title('%i degree rotated image' %angle)
plt.set_cmap('gray')
# show original image
plt.subplot(122)
plt.imshow(dst)
plt.title('%i degree rotated image in openCV' %angle)
plt.set_cmap('gray')
plt.show()

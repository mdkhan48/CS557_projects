import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/auto.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/building.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/child.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/ct_scan.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/tire.pnm')

#constant = 255/(np.log(1+np.max(img)))
img_log = np.log(np.absolute(img)+1)*255/(np.log(1+np.max(img)))

img_log = np.array(img_log,dtype=np.uint8)

# Display the image
#plotting images
plt.subplot(121)
plt.imshow(img)
plt.title('original image')
plt.set_cmap('gray')
# show original image
plt.subplot(122)
plt.imshow(img_log)
plt.title('Log transformed image')
plt.set_cmap('gray')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
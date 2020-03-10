import numpy as np
from math import pi
import matplotlib.pyplot as plt
import cv2


#img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/auto.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/building.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/child.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/ct_scan.pnm')
img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/tire.pnm')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



#creating different types of noise
def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 2
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.04
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return noisy

#adding noise to image-output gray image
noisy_img = noisy('gauss',img)#creating noisy image
noisy_img = np.uint8(noisy_img) #converting unit8, float64 shows error in cv2.cvtcolor
gray_noisy = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)
print(gray_noisy.shape)

#resizing image to sq. form
max_length = int(np.amax(gray_noisy.shape))
gray_noisy = cv2.resize(gray_noisy,(max_length,max_length))


blank_im = np.zeros_like(gray_noisy)#creating blank image same shape as input image

median_filter = [(0,0)] * 9 #array with 9 (0,0) memebers

height, width = gray_noisy.shape

for i in range(1,width-1):
    for j in range(1,height-1):
        median_filter[0] = gray_noisy[i-1,j-1]
        median_filter[1] = gray_noisy[i-1,j]
        median_filter[2] = gray_noisy[i-1,j+1]
        median_filter[3] = gray_noisy[i,j-1]
        median_filter[4] = gray_noisy[i,j]
        median_filter[5] = gray_noisy[i,j+1]
        median_filter[6] = gray_noisy[i+1,j-1]
        median_filter[7] = gray_noisy[i+1,j]
        median_filter[8] = gray_noisy[i+1,j+1]
        median_filter.sort()
        blank_im[i,j] = median_filter[4]

filtered_im = blank_im #median filtered image
cv_filter = cv2.medianBlur(gray_noisy,5)#filtered using opencv



#plotting images
plt.subplot(221)
plt.imshow(img)
plt.title('original image')
plt.set_cmap('gray')
# show noisy image
plt.subplot(222)
plt.imshow(noisy_img)
plt.title('Gaussian noisy image')
#plt.title('Salt & pepper noisy image')
plt.set_cmap('gray')

plt.subplot(223)
plt.imshow(filtered_im)
plt.title('Median filter image')
plt.set_cmap('gray')

plt.subplot(224)
plt.imshow(cv_filter)
plt.title('OpenCV median filter')
plt.set_cmap('gray')

plt.show()
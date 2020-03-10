import numpy as np
from math import pi
import matplotlib.pyplot as plt
import cv2


#img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/auto.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/building.pnm')
img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/child.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/ct_scan.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/CS557_code_project2/tire.pnm')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = gray.shape

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

def gauss2dfilter(sigma):
    length = 5*sigma# rule of thumb approximation
    ax = np.arange(-length // 2 + 1., length // 2 + 1.)
    #print(ax)
    xx, yy = np.meshgrid(ax, ax)
    #print(xx)
    kernel = (np.exp(-0.5 * (np.square(xx) + np.square(yy)) / (2*np.square(sigma))))/(2*pi*np.square(sigma))
    #print(kernel)
    kernel2d = kernel / np.sum(kernel)
    #print(kernel2d)
    plt.imshow(kernel2d, interpolation='none')
    plt.title('Gaussian kernel')
    plt.show()
    return kernel2d



def convolution2d(image, kernel):
    m, n = kernel.shape
    if (m == n):
        y, x= image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros_like(image)
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel)
    return new_image

kernel2d = gauss2dfilter(sigma=1)

noisy_img = noisy('gauss',img)#creating noisy image
noisy_img = np.uint8(noisy_img) #converting unit8, float64 shows error in cv2.cvtcolor
#print('noisy image data type',noisy_img.dtype)
gray_noisy = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)

filtered_image = convolution2d(gray_noisy,kernel2d) #creating filtered image

cv_filter = cv2.GaussianBlur(gray_noisy,(5,5),1) #openCV filter fixed at sigma 1 and length 5

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
plt.imshow(filtered_image)
plt.title('Gaussian filter image, sig=1')
plt.set_cmap('gray')

plt.subplot(224)
plt.imshow(cv_filter)
plt.title('OpenCV gaussian filter')
plt.set_cmap('gray')

plt.show()
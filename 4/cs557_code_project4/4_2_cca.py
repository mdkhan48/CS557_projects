import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy import NaN, Inf, arange, asarray, array

#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/hinge.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/hinges.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/shapes1.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/shapes2.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/shapes3.pnm')
img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/shapes4.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/keys.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/pillsetc.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project4/coins.pnm')

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

amin, amax = Inf, -Inf
min_position, max_position = NaN, NaN

lookformax = True

for i in arange(len(v)):
    this = v[i]
    if this > amax:
        amax = this
        max_position = x[i]
    if this < amin:
        amin = this
        min_position = x[i]

    if lookformax:
        if this < amax - delta:
            max_value.append((max_position, amax))
            amin = this
            min_position = x[i]
            lookformax = False
    else:
        if this > amin + delta:
            min_value.append((min_position, amin))
            amax = this
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
segment[segment >= threshold] = 255
segment[segment < threshold] = 0
#segment = cv2.bitwise_not(segment)

# show original image
plt.imshow(segment,cmap='gray')
plt.show()

# make matrix of dimension that of image
height, width = img.shape[0], img.shape[1]
matrix = np.zeros((height, width))
matrix = matrix.astype(int,order=2)
print (matrix.dtype)

# Algorithm implementation
label = 1
dic = {}
for x in range(height):
    for y in range(width):
        value = segment[x, y]
        """
        Non recursive algorithm for 4-CCA
                                       | Xu |
                                  X_L  | Xc |        
        """
        if value > 0: #looking at pixels BOT background
            left = matrix[x, y - 1]
            top = matrix[x - 1, y]
            if left > 0 and top > 0: #Xu=Xl then Xc=Xl
                matrix[x, y] = left
                if left != top:
                    dic.update([(max(left, top), min(left, top))]) # Xl!=Xu then Xl equivalent Xu
            elif left > 0 and top == 0: #Xl=1 and Xu=0 then Xc=Xl
                matrix[x, y] = left
            elif top > 0 and left == 0: #Xl=0 and Xu=1 then Xc=Xu
                matrix[x, y] = top
            else:
                label = label + 1 #Xl=0 and Xu=0 then Xc get new label
                matrix[x, y] = label

plt.imshow(matrix, cmap="Paired")
plt.show()

#print(dic)
key = list(dic.keys()) # remembers max label
#print(key)
#print(key[len(key)-1])
val = list(dic.values()) #remembers which min label is eq to key label
#print(val)

for z in range(len(key)):
    for x in range(0, height):
        for y in range(0, width):
            values = matrix[x, y]
            if values == key[len(key)-1]: #when pixel value is equal to max(or final) label in key
                matrix[x, y] = val[len(val)-1] #replace pixel value with eq min value
    #after replacing value for whole image
    #delete the final label in key and val
    #which is already replaced and go through the whole
    #image again for next set of values
    key.remove(key[len(key)-1])
    val.remove(val[len(val)-1])


retval, cca_CV = cv2.connectedComponents(segment,connectivity=4)

print(np.unique(cca_CV))

plt.subplot(131)
plt.imshow(segment,cmap='gray')
plt.subplot(132)
plt.title('Implemented CCA')
plt.imshow(np.ma.array(matrix, mask=matrix==0),interpolation='nearest')

plt.subplot(133)
#plt.imshow(result_im, cmap="Paired")
plt.imshow(np.ma.array(cca_CV, mask=cca_CV==0),interpolation='nearest')
plt.title('CCA Opencv')


plt.show()
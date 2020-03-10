
import numpy as np
from matplotlib import pyplot as plt
import cv2



#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/Lenna.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/building.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/hinge.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/hinges.pnm')
#img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/keys.pnm')
img = cv2.imread('/home/sayem/Desktop/cs557_project/cs557_code_project3/pillsetc.pnm')

img1 = img.copy()
img2 = img.copy()

rows, cols, channels = img.shape
img = cv2.GaussianBlur(img,(5,5),1)


Upper_thresh = 255
Lower_thresh = 50

#Edge detection
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
y_length, x_length = gray.shape
edges = cv2.Canny(gray,Lower_thresh,Upper_thresh,apertureSize = 3)
#print(edges[100,205])

loc = np.where(edges ==255) #looking for white pixels
X = []
y = []
# loop though the points
for pt in zip(*loc[::-1]):
	#print(pt)
	X.append(pt[0])
	y.append(pt[1])

#print(X)
X = np.asarray(X) #changing list to array
X = X.reshape(X.shape[0],1)# changing (value,) to (value,1)
#print(X.shape)
y = np.asarray(y)


#Fits model for a given data using least squares.
def fit_with_least_squares(X, y):
    """
    Fits model for a given data using least squares.
    X should be an mxn matrix, where m is number of samples, and n is number of independent variables.
    y should be an mx1 vector of dependent variables.
    """
    b = np.ones((X.shape[0], 1))
    A = np.hstack((X, b))
    theta = np.linalg.lstsq(A, y)[0]
    return theta

#Evaluates model and returns total number of inliers.
def evaluate_model(X, y, theta, inlier_threshold):
	"""
    Evaluates model and returns total number of inliers.
    X should be an mxn matrix, where m is number of samples, and n is number of independent variables.
    y should be an mx1 vector of dependent variables.
    theta should be an (n+1)x1 vector of model parameters.
    inlier_threshold should be a scalar.
    """
	b = np.ones((X.shape[0], 1))
	y = y.reshape((y.shape[0], 1))
	A = np.hstack((y, X, b))
	theta = np.insert(theta, 0, -1.)

	distances = np.abs(np.sum(A * theta, axis=1)) / np.sqrt(np.sum(np.power(theta[:-1], 2)))
	inliers = distances <= inlier_threshold
	num_inliers = np.count_nonzero(inliers == True)

	return num_inliers


def ransac(X, y, fit_fn, evaluate_fn, max_iters=5000, samples_to_fit=2, inlier_threshold=0.99, min_inliers=100):
	best_model = None
	best_model_performance = 0

	num_samples = X.shape[0]

	for i in range(max_iters):
		sample = np.random.choice(num_samples, size=samples_to_fit, replace=False)
		model_params = fit_fn(X[sample], y[sample])
		model_performance = evaluate_fn(X, y, model_params, inlier_threshold)

		if model_performance < min_inliers:
			continue

		if model_performance > best_model_performance:
			best_model = model_params
			best_model_performance = model_performance

	return best_model

print ("Least Sq. m b:", ransac(X, y, fit_fn = fit_with_least_squares, evaluate_fn = evaluate_model, max_iters=500, samples_to_fit=2, inlier_threshold=0.1, min_inliers=10))

m = ransac(X, y, fit_fn = fit_with_least_squares, evaluate_fn = evaluate_model, max_iters=100, samples_to_fit=2, inlier_threshold=0.1, min_inliers=10)[0]
b = ransac(X, y, fit_fn = fit_with_least_squares, evaluate_fn = evaluate_model, max_iters=100, samples_to_fit=2, inlier_threshold=0.1, min_inliers=10)[1]

plt.scatter(X,y,0.1)
plt.plot(X, m * X + b, 'r-')
plt.show()
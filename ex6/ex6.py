from matplotlib import use, cm
use ('TkAgg')
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn import svm

def main():

	# =============== Part 1: Loading and Visualizing Data ================
	
	print("Loading and Visualizing Data")

	data = scipy.io.loadmat("ex6data1.mat")
	X = data['X']
	y = data['y'].flatten()
	plotData(X,y)

	# ==================== Part 2: Training Linear SVM ====================

	print("Training Linear SVM")
	C = 1
	clf = svm.SVC(C=C, kernel = 'linear', tol = 1e-3, max_iter = 50)
	model = clf.fit(X,y)
	visualizeBoundaryLinear(X,y,model)

	# =============== Part 3: Implementing Gaussian Kernel ===============

	print("Evaluating the Gaussian Kernel")

	x1 = np.array([1,2,1])
	x2 = np.array([0,4,-1])
	sigma = 2
	sim = gaussianKernel(x1,x2,sigma)

	print("Gaussian Kernel between x1 = [1,2,1] and x2 = [0,4,-1] is %f" % sim)
	print("This value should be about 0.324652")

	# =============== Part 4: Visualizing Dataset 2 ================

	data = scipy.io.loadmat("ex6data2.mat")
	X = data['X']
	y = data['y'].flatten()

	plotData(X,y)

	# ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========

	print("Training SVM with RBF Kernel (may take 1 to 2 minutes)")

	C= 1
	sigma = 0.1
	gamma = 1/(2*sigma*sigma)

	clf = svm.SVC(C=C, kernel='rbf', tol = 1e-3, max_iter =400, gamma = gamma)
	model = clf.fit(X,y)
	visualizeBoundary(X,y,model)

	# =============== Part 6: Visualizing Dataset 3 ================

	print("Loading and Visualizing Data")

	data = scipy.io.loadmat('ex6data3.mat')
	X = data['X']
	y = data['y'].flatten()
	Xval = data['Xval']
	yval = data['yval'].flatten()

	plotData(X,y)

	# ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

	C, sigma = dataset3Params(X,y,Xval,yval)
	gamma = 1/(2*sigma*sigma)

	clf = svm.SVC(C=C, kernel= 'rbf', tol = 1e-3, gamma = gamma)
	model = clf.fit(X,y)
	visualizeBoundary(X,y,model)


def dataset3Params(X,y,Xval,yval):
	C = 1
	sigma = 0.3
	values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
	minerror = np.inf
	for C in values:
		for sigma in values:
			gamma = 1/(2*sigma*sigma)
			clf = svm.SVC(C=C, kernel = 'rbf', tol = 1e-3, gamma = gamma)
			model = clf.fit(X,y)
			error = np.mean(np.double(model.predict(Xval)!= yval))
			if(error <= minerror):
				C_final = C
				sigmafinal = sigma
				minerror = error

	C = C_final
	sigma = sigmafinal

	return C, sigma



def gaussianKernel(x1,x2,sigma):
	x1 =x1[:]
	x2 =x2[:]
	sim = 0

	x1minus2 = x1-x2
	sim = np.exp(-(np.sum(x1minus2*x1minus2)/(2*sigma*sigma)))

	return sim

def visualizeBoundary(X,y,model):
	x1plot = np.linspace(min(X[:,0]), max(X[:,0]), X.shape[0]).T
	x2plot = np.linspace(min(X[:,1]), max(X[:,1]), X.shape[0]).T
	X1,X2 = np.meshgrid(x1plot,x2plot)
	vals =np.zeros(X1.shape)
	for i in range(X1.shape[1]):
		thisX = np.column_stack((X1[:,i],X2[:,i]))
		vals[:,i] = model.predict(thisX)

	plt.contour(X1,X2,vals)
	plotData(X,y)


def visualizeBoundaryLinear(X,y,model):
	w = model.coef_.flatten()
	b = model.intercept_.flatten()
	xp = np.linspace(min(X[:,0]), max(X[:,0]), 100)
	yp = -(w[0]*xp + b)/w[1]
	plt.plot(xp,yp, '-b')
	plotData(X,y)


def plotData(X,y):

	pos = np.where(y==1, True, False).flatten()
	neg = np.where(y==0, True, False).flatten()

	plt.plot(X[pos,0], X[pos,1], 'k+', linewidth = 1, markersize = 7)
	plt.plot(X[neg,0], X[neg,1], 'ko', markersize = 7, color = 'yellow')
	plt.show()




main()
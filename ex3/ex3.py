from matplotlib import use
use('TkAgg')
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def main():
	inputlayersize = 400
	numlabels = 10

	# =========== Part 1: Loading and Visualizing Data =============

	print("Loading and Visualizing Data")
	data = scipy.io.loadmat('ex3data1.mat')
	X = data['X']
	y = data['y']
	m,n = X.shape

	randindices = np.random.permutation(range(m))
	selection = X[randindices[0:100],:]

	displayData(selection)

	# ============ Part 2: Vectorize Logistic Regression ============

	print ("Training One-vs-All Logistic Regression")

	Lambda = 0.1
	alltheta = oneVsAll(X,y,numlabels,Lambda)

	## ================ Part 3: Predict for One-Vs-All ================

	prediction = predictOnevsAll(alltheta,X)
	acc = np.mean(np.double(prediction == np.squeeze(y)))*100
	print("Training set accuracy : %f" % acc)


def oneVsAll(X,y,numlabels,Lambda):
	m,n = X.shape
	alltheta = np.zeros((numlabels, n+1))
	X = np.column_stack((np.ones((m,1)),X))
	itheta = np.zeros((n+1))
	for i in range(numlabels):
		tempy = np.where(y==i+1,1,0)
		result = minimize(lrCostFunction, itheta, method = 'TNC' , tol = 1e-6, jac = True, args =(X,tempy,Lambda), options = {'disp': False, 'maxiter': 50})
		alltheta[i,:] = result.x
	
	return alltheta

def lrCostFunction(theta,X,y,Lambda):
	m = y.shape[0]
	y=y.T[0]
	J= 0.0
	grad = np.zeros(theta.shape)
	mtheta = theta[1:theta.size]
	ttheta = np.insert(mtheta,0,0)
	h  = sigmoid(X.dot(theta))
	J = (1/m)*(np.sum((-y)*np.log(h)-(1-y)*np.log(1-h))) + (Lambda/(2*m))*np.sum(ttheta*ttheta)
	grad = (1/m)*(X.T.dot(h-y)+(Lambda*ttheta))
	return J, grad


def sigmoid(z):
	g = np.zeros(z.size)
	g = 1/(1 + np.exp(-z))

	return g

def predictOnevsAll(alltheta,X):
	m = X.shape[0]
	X = np.column_stack((np.ones((m,1)),X))
	predict = sigmoid(X.dot(alltheta.T))
	#Everything must be bumped by 1 since 0 = 10 in our representation
	predindex = np.argmax(predict, axis = 1) + 1

	return predindex
	

def displayData(X):
	m,n = X.shape
	exwidth = round(np.sqrt(n))
	exheight = (n/exwidth)

	disprows = np.floor(np.sqrt(m))
	dispcols = np.ceil(m/disprows)
	pad = 1
	#called as int to avoid deprecation warning
	disparray = np.ones((int(pad+disprows*(exheight+pad)), pad + int(dispcols*(exwidth + pad))))
	curr_ex = 0

	for j in np.arange(disprows):
		for i in np.arange(dispcols):
			if curr_ex > m:
				break
			maxval = np.max(np.abs(X[curr_ex,:]))
			rows = [pad + j*(exheight + pad)+ x for x in np.arange(exheight+1)]
			cols = [pad + i*(exwidth + pad)+ x for x in np.arange(exwidth+1)]
			disparray[int(min(rows)):int(max(rows)), int(min(cols)):int(max(cols))] = X[curr_ex,:].reshape(int(exheight),int(exwidth))/maxval
			curr_ex = curr_ex+1
		if curr_ex > m:
			break

	disparray = disparray.astype('float32')
	plt.imshow(disparray.T)
	plt.set_cmap('gray')
	plt.axis('off')
	plt.show()


main()	
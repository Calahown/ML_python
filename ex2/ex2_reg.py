from matplotlib import use
use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from math import log
import pandas as pd


def main():
	data = pd.read_csv('ex2data2.txt', header=None, names=[1,2,3])
	X = data[[1,2]]
	y = data[[3]]

	plotData(X.values,y.values)
	plt.xlabel('Microchip Test 1')
	plt.ylabel('Microchip Test 2')
	plt.show()

	# =========== Part 1: Regularized Logistic Regression ============

	X = X.apply(mapFeature, axis = 1)

	itheta = np.zeros(X.shape[1])
	Lambda = 0.0

	cost, grad = costFunctionReg(itheta,X.as_matrix(),y,Lambda)

	print("Cost at initial theta: %f" % cost)
	

	# ============= Part 2: Regularization and Accuracies =============

	Lambda = 1.0
	result = minimize(costFunctionReg, itheta, method = 'L-BFGS-B', jac = True, args = (X.as_matrix(),y,Lambda), options = {'gtol': 1e-4, 'disp': False, 'maxiter': 1000})
	theta = result.x
	cost = result.fun

	print('lambda = ' + str(Lambda))
	print("Cost at theta by scipy: %f" % cost)
	print("theta: ", ["%0.4f" % i for i in theta])

	plotBoundry(theta,X,y,Lambda)

	p = predict(theta,X)
	acc = np.mean(np.where(p==y.T,1,0))*100
	print("Train accuracy: %f" % acc)




def plotData(X,y):
	pos = X[np.where(y==1,True,False).flatten()]
	neg = X[np.where(y==0,True,False).flatten()]
	plt.plot(pos[:,0],pos[:,1],'+',markersize=7,markeredgecolor='black',markeredgewidth=2)
	plt.plot(neg[:,0],neg[:,1],'o',markersize=7,markeredgecolor='black',markerfacecolor='yellow')

def plotBoundry(theta,X,y,Lambda):
	plotDecisionBoundary(theta,X.values, y.values)
	plt.title(r'$\lambda$ = ' + str(Lambda))

	plt.xlabel('Microchip Test1')
	plt.ylabel('Microchip Test2')
	plt.show()


def plotDecisionBoundary(theta,X,y):
	plt.figure()
	plotData(X[:,1:],y)

	if(X.shape[1]) <=3:
		plot_x = np.array([min(X[:,2]), max(X[:,2])])

		plot_y = (-1./theta[2])*(theta[1]*plot_x + theta[0])

		plt.plot(plot_x,plot_y)


	else:
		u = np.linspace(-1,1.5,50)
		v = np.linspace(-1,1.5,50)
		z = [np.array([mapFeature2(u[i],v[j]).dot(theta) for i in range(len(u))]) for j in range(len(v))]
		plt.contour(u,v,z, levels = [0])



def mapFeature(X, degree = 6):
	quads = pd.Series([X.iloc[0]**(i-j) * X.iloc[1]**j for i in range(1,degree+1) for j in range(i+1)])
	return pd.Series([1]).append([quads])

def mapFeature2(X1,X2,degree=6):
	quads = pd.Series([X1**(i-j) * X2**j for i in range(1,degree+1) for j in range(i+1)])
	return pd.Series([1]).append([quads])


def costFunctionReg(theta,X,y,Lambda):
	m = y.shape[0]
	J= 0.0
	grad = np.zeros(theta.shape)
	mtheta = theta[1:theta.size]
	ttheta = np.insert(mtheta,0,0)

	h = pd.DataFrame(sigmoid(X.dot(theta)), columns =[3])

	J = (1/m)*np.sum(((-y)*np.log(h))-((1-y)*np.log(1-h))) + (Lambda/(2*m))*np.sum(ttheta*ttheta)

	grad = (1/m)*(X.T.dot(h-y)+pd.DataFrame(Lambda*ttheta, columns =[3]))
	return J.values[0], grad.values.T[0]

def sigmoid(z):
	g = np.zeros(z.size)
	g = 1/(1 + np.exp(-z))

	return g

def predict(theta, X):
	m = X.size
	p = np.zeros(m)
	p = sigmoid(X.dot(theta)) >= 0.5

	return p

main()
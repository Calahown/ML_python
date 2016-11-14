from matplotlib import use
use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

def main():

	# ================ Part 1: Feature Normalization ================

	print("Loading data...")
	data = np.loadtxt("ex1data2.txt", delimiter = ',')
	X = np.array(data[:, :2])
	y = np.array(data[:,2])
	m = y.shape[0]

	print("First 10 examples of the dataset")
	print(np.column_stack( [X[:10],y[:10]]))

	print("Normalizing features")

	X,mu,sigma = featureNormalize(X)


	X = np.concatenate((np.ones((m,1)),X),axis = 1)

	# ================ Part 2: Gradient Descent ================

	print("Running gradient decent")

	alpha = 1
	num_iters = 100
	theta = np.zeros(3)
	theta, J_history = gradientDecent(X,y,theta,alpha,num_iters)

	plt.plot(J_history, 'r')
	plt.xlabel('Iteration')
	plt.ylabel('Cost J')
	plt.show()

	print('Theta: ' + str(theta))
	# Estimate price of a 1650ft^2 house

	price = np.array([1,(1650-mu[0])/(sigma[0]),(3-mu[1])/(sigma[1])]).dot(theta)

	print("Predicted price of a 1650^2 ft house")
	print(price)

	# ================ Part 3: Normal Equations ================

	print("Solving with normal equations")
	data = np.loadtxt('ex1data2.txt', delimiter = ',')
	X = np.array(data[:, :2])
	y = np.array(data[:,2])
	m = y.shape[0]

	X = np.concatenate((np.ones((m,1)),X),axis = 1)

	theta = normalEqn(X,y)

	print('Theta :' + str(theta))

	price = np.array([1,1650,3]).dot(theta)

	print("new prediction for 1650^2 ft house")
	print(price)


def featureNormalize(X):
	X_norm = X
	mu = np.zeros(X.shape[1])
	sigma = np.zeros(X.shape[1])

	for i in range(mu.shape[0]):
		mu[i] = np.mean(X[:,i])
		sigma[i] = np.std(X[:,i])

	for i in range(mu.shape[0]):
		X_norm[:,i] = (X[:,i]-mu[i])/(sigma[i])

	return X_norm , mu , sigma


def computeCost(X,y,theta):
	m = y.size
	J = 0
	h = theta[0]*X[:,0]+theta[1]*X[:,1]
	J= (sum((h-y)*(h-y))/(2*m))
	return J

def gradientDecent(X,y,theta,alpha,iterations):
	m = y.size
	alpha = alpha/m
	J_history = []
	k = np.zeros(X.shape[1])
	for i in range(iterations):
		h = X.dot(theta)
		k = alpha*(X.T.dot(h-y))
		theta = theta - k
		J_history.append(computeCost(X,y,theta))
	return theta, J_history

def normalEqn(X,y):

	#theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
	theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
	return theta


main()
from matplotlib import use
use('TkAgg')
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.optimize import minimize
np.set_printoptions(threshold=np.inf)

def main():

	input_layer_size = 400
	hidden_layer_size = 25
	num_labels = 10

	# =========== Part 1: Loading and Visualizing Data =============

	print("Loading and Visualizing Data")
	data = scipy.io.loadmat('ex4data1.mat')
	X = data['X']
	y = data['y']
	m, n = X.shape
	y = (y-1)%10
	randindex = np.random.permutation(range(m))
	sel = X[randindex[0:100],:]
	displayData(sel)

	# ================ Part 2: Loading Parameters ================

	print("Loading Saved Neural Network parameters")
	data = scipy.io.loadmat('ex4weights.mat')
	theta1 = data['Theta1']
	theta2 = data['Theta2']
	y = np.squeeze(y)
	nn_params = np.append(theta1,theta2).reshape(-1)

	# ================ Part 3: Compute Cost (Feedforward) ================

	print("Feedforward using Neural Network")

	Lambda = 0

	J, grad = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda)

	print("Cost at loaded parameters should be about 0.287629")
	print("J: %f" % J)

	# =============== Part 4: Implement Regularization ===============

	print("Checking Cost function with regularization")

	Lambda = 1;

	J, grad = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda)

	print("Cost at loaded parameters with regularization should be about 0.383770")
	print("J: %f" % J)

	# ================ Part 5: Sigmoid Gradient  ================

	print("Evaluating sigmoid gradient")
	g = sigmoidGradient(np.array([1, -0.5, 0 , 0.5, 1]))
	print("Sigmoid gradient evaluated at [1, -0.5, 0 , 0.5, 1]")
	print(g)

	# ================ Part 6: Initializing Parameters ================

	print("Initializing Neural Network Parameters")
	itheta1 = randInitializeWeights(input_layer_size+1,hidden_layer_size)
	itheta2 = randInitializeWeights(hidden_layer_size+1,num_labels)

	initialnnparams = np.append(itheta1,itheta2).reshape(-1)

	# =============== Part 7: Implement Backpropagation ===============

	print("Checking Backpropagation")

	checkNNGradients(Lambda)

	# =============== Part 8: Implement Regularization ===============

	print("Checking Backpropagation with regularization")

	Lambda = 3
	checkNNGradients(Lambda)

	debugJ , grad = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda)

	print("Lambda: " + str(Lambda))
	print("Cost at fixed debugging parameters with Lambda = 10 %f" % debugJ)
	print("This value should be about 0.576051")

	# =================== Part 8: Training NN ===================

	print("Training Neural Network")
	Lambda = 1

	#costFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)[0]
	result = minimize(nnCostFunction,initialnnparams, args = (input_layer_size,hidden_layer_size,num_labels,X,y,Lambda)
		,method = "CG", jac = True, options={'disp': True, 'maxiter': 50})

	#result = minimize(costFunc,initialnnparams,method = "CG",jac = False, options ={'disp': True, 'maxiter': 50})

	nn_params = result.x
	cost = result.fun
	theta1 = nn_params[:(hidden_layer_size*(input_layer_size +1))].reshape((hidden_layer_size, input_layer_size +1))
	theta2 = nn_params[-((hidden_layer_size+1)*num_labels):].reshape((num_labels,hidden_layer_size+1)) 

	# ================= Part 9: Visualize Weights =================

	print("Visualizing Neural Network")

	displayData(theta1[:,1:])

	# ================= Part 10: Implement Predict =================

	pred = predict(theta1,theta2,X)
	acc = np.mean(np.double(pred == y)) *100
	print("Training set Accuracy: %f" % acc)


def checkNNGradients(Lambda):
	input_layer_size = 3
	hidden_layer_size = 5
	num_labels = 3
	m= 5
	theta1 = debugInitializeWeights(hidden_layer_size,input_layer_size)
	theta2 = debugInitializeWeights(num_labels,hidden_layer_size)
	X = debugInitializeWeights(m,input_layer_size-1)
	y = np.mod(range(1,m+1), num_labels)
	nn_params = np.hstack((theta1.T.ravel(), theta2.T.ravel()))

	costFunc = lambda p: nnCostFunction(p,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda)

	cost, grad = costFunc(nn_params)
	numgrad = computeNumericalGradient(costFunc,nn_params)

	print(np.column_stack((numgrad,grad)))
	print("The above two columns should be close to equal")


	diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)

	print("If the backprop is correct then the relative difference will be small")
	print("Relative diff %f" % diff)

def debugInitializeWeights(fout,fin):
	W = np.zeros((fout,fin+1))
	W = np.reshape(np.sin(range(1, W.size+1)), W.T.shape).T / 10.0
	return W

def computeNumericalGradient(J, theta):
	numgrad = np.zeros(theta.shape[0])
	perturb = np.zeros(theta.shape[0])
	e = 1e-4
	for p in range(theta.size):
		perturb[p] = e
		loss1 = J(theta-perturb)
		loss2 = J(theta+perturb)

		numgrad[p] = (loss2[0] - loss1[0])/(2*e)
		perturb[p] = 0
	return numgrad


def nnCostFunction(nn_params, input_layer_size,hidden_layer_size,num_labels,X,y,Lambda):
	J=0.0
	m,n = X.shape
	theta1 = nn_params[:(hidden_layer_size*(input_layer_size +1))].reshape((hidden_layer_size, input_layer_size +1))
	theta2 = nn_params[-((hidden_layer_size+1)*num_labels):].reshape((num_labels,hidden_layer_size+1))
	theta1grad = np.zeros(theta1.shape)
	theta2grad = np.zeros(theta2.shape)
	inity = np.zeros((m, num_labels))
	for i in range(len(y)):
		inity[i][y[i]] = 1 

	a1 = np.hstack((np.ones((m,1)), X))
	z2 = a1.dot(theta1.T)
	a2 = np.hstack((np.ones((z2.shape[0],1)), sigmoid(z2)))
	z3 = a2.dot(theta2.T)
	H = sigmoid(z3)

	reg = (Lambda/(2*m))*(np.sum(np.sum(theta1[:,1:]**2)) + np.sum(theta2[:,1:]**2))

	J = (1/m)*np.sum(np.sum((-inity)*np.log(H) - (1-inity)*np.log(1-H)))
	J = J + reg

	sigma3 = H - inity
	sigma2 = (((sigma3.dot(theta2)))* sigmoidGradient(np.hstack((np.ones((z2.shape[0],1)), z2))))[:,1:]

	delta1 = sigma2.T.dot(a1)
	delta2 = sigma3.T.dot(a2)
	
	theta1grad = (delta1/m) + (Lambda/m)*np.hstack((np.zeros((theta1.shape[0],1)), theta1[:,1:]))
	theta2grad = (delta2/m) + (Lambda/m)*np.hstack((np.zeros((theta2.shape[0],1)), theta2[:,1:]))

	grad = np.append(theta1grad, theta2grad).reshape(-1)

	return J, grad

def randInitializeWeights(Lin,Lout):
	return np.random.uniform(low=-.12,high=.12,size=(Lin,Lout))

def sigmoid(z):
	g = np.zeros(z.size)
	g = 1/(1 + np.exp(-z))
	return g

def sigmoidGradient(z):
	return sigmoid(z)*(1-sigmoid(z))

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

def predict(theta1, theta2, X):
	m = X.shape[0]
	numlabels = theta2.shape[0]

	p = np.zeros((m,1))

	a1 = np.hstack((np.ones((m,1)), X))
	z2 = a1.dot(theta1.T)
	a2 = np.hstack((np.ones((z2.shape[0],1)), sigmoid(z2)))
	z3 = a2.dot(theta2.T)
	a3 = sigmoid(z3)
	p = np.argmax(a3, axis = 1)

	return p


main()
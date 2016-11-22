import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize 

def main():


	# =========== Part 1: Loading and Visualizing Data =============

	print("Loading and Visualizing data")

	data = scipy.io.loadmat('ex5data1.mat')
	X = data['X']
	y = data['y']
	Xtest = data['Xtest']
	Xval = data['Xval']
	ytest = data['ytest']
	yval = data['yval']

	plt.plot(X,y,'rx', markersize = 10, linewidth = 1.5)
	plt.xlabel('Change in water level (x)')
	plt.ylabel('Water flowing out of the dam (y)')
	plt.show()

	# =========== Part 2: Regularized Linear Regression Cost =============
	m = X.shape[0]
	theta = np.array([1,1])
	J , grad = linearRegCostFunction(theta, np.column_stack((np.ones((m,1)), X)), y , 1)

	print('Cost at theta = [1 1]: %f' % J)
	print("This value should be about 303.993192")

	# =========== Part 3: Regularized Linear Regression Gradient =============

	print('Gradient at theta = [1,1]: ' + str(grad))
	print('This value should be about [-15.303016; 598.250744]')

	# =========== Part 4: Train Linear Regression =============

	Lambda = 0
	theta = trainLinearReg(np.hstack((np.ones((m,1)), X)), y , Lambda)

	plt.plot(X,y,'rx', markersize=10, linewidth=1.5)
	plt.plot(X,np.column_stack((np.ones((m,1)), X)).dot(theta), '--', linewidth = 2)
	plt.xlabel("Change in water level")
	plt.ylabel("Water flowing out of the dam")
	plt.show()

	# =========== Part 5: Learning Curve for Linear Regression =============

	errortrain, errorval = learningCurve(np.column_stack((np.ones((m,1)), X)),y,
		np.column_stack((np.ones((Xval.shape[0],1)), Xval)), yval, Lambda)

	plt.plot(range(1,m+1), errortrain ,range(1,m+1), errorval)
	plt.title("Learning curve for linear regression")
	plt.legend(['Train','Cross Validation'])
	plt.xlabel("Number of training examples")
	plt.ylabel("Error")
	plt.axis([0,13,0,150])
	plt.show()

	# =========== Part 6: Feature Mapping for Polynomial Regression =============

	p = 8

	Xpoly = polyFeatures(X,p)
	Xpoly, mu , sigma = featureNormalize(Xpoly)
	Xpoly = np.hstack((np.ones((m,1)), Xpoly))

	Xpolytest = polyFeatures(Xtest,p)
	Xpolytest = (Xpolytest-mu)/(sigma)
	Xpolytest = np.hstack((np.ones((Xtest.shape[0],1)), Xpolytest))

	Xpolyval = polyFeatures(Xval,p)
	Xpolyval = (Xpolyval-mu)/(sigma)
	Xpolyval = np.hstack((np.ones((Xval.shape[0],1)), Xpolyval))

	# =========== Part 7: Learning Curve for Polynomial Regression =============

	theta = trainLinearReg(Xpoly, y , Lambda)

	plt.plot(X,y,'rx', markersize=10,linewidth = 1.5)
	plotFit(np.min(X), np.max(X), mu, sigma, theta, p)
	plt.xlabel("Change in water level")
	plt.ylabel("Water flowing out of the dam")
	plt.title("Polynomial Regression Fit (Lambda = %f)" % Lambda)
	plt.show()

	errortrain, errorval = learningCurve(Xpoly, y , Xpolyval, yval, Lambda)

	plt.plot(range(m), errortrain, range(m), errorval)
	plt.title("Polynomial Regression Learning Curve (lambda = %f)" % Lambda)
	plt.xlabel("Number of training examples")
	plt.ylabel("Error")
	plt.axis([0, 13, 0, 100])
	plt.legend(['Train', 'Cross Validation'])
	plt.show()

	#lambdavec , errortrain, errorval = validationCurve(np.column_stack((np.ones((m,1)), X)), y
		#,np.column_stack((np.ones((Xval.shape[0],1)), Xval)), yval)
	lambdavec, errortrain, errorval = validationCurve(Xpoly,y, Xpolyval, yval)

	plt.plot(lambdavec, errortrain, lambdavec, errorval)
	plt.title("Selecting lambda using a cross validation set")
	plt.legend(['Train', 'Cross Validation'])
	plt.xlabel('lambda')
	plt.ylabel('Error')
	print("lambda vec" + '\t' + "error train" + '\t' + "error val")
	for i in range(len(lambdavec)):
		print (str(lambdavec[i])+ '\t\t' + str(errortrain[i]) + '\t' + str(errorval[i]))
	plt.axis([0, 10 , 0, 30]) 
	plt.show()



def plotFit(minx, maxx, mu, sigma, theta, p):
	x = np.arange(minx -15, maxx + 25, 0.05).reshape((-1,1))
	Xpoly = polyFeatures(x,p)
	Xpoly = (Xpoly - mu)/(sigma)
	Xpoly = np.hstack((np.ones(Xpoly.shape[0]).reshape((-1,1)), Xpoly))
	plt.plot(x,Xpoly.dot(theta), '--', linewidth = 2)
	return

def polyFeatures(X,p):
	Xpoly = np.zeros((len(X), p))
	X = X.flatten()
	for i in range(1,p+1):
		Xpoly[:,i-1] = X**i
	return Xpoly

def featureNormalize(Xpoly):
	mu = np.mean(Xpoly, axis = 0)
	sigma = np.std(Xpoly, axis = 0)
	Xnorm = (Xpoly-mu)/(sigma)
	return Xnorm, mu, sigma


def learningCurve(X,y,Xval,yval,Lambda):
	m = X.shape[0]
	errortrain = np.zeros((m,1))
	errorval = np.zeros((m,1))

	for i in range(1,m+1):
		Xtrain = X[:i,:]
		ytrain = y[:i]
		theta = trainLinearReg(Xtrain,ytrain,Lambda)
		errortrain[i-1], grad = linearRegCostFunction(theta, Xtrain , ytrain, 0)
		errorval[i-1],grad = linearRegCostFunction(theta, Xval,yval,0)

	return errortrain, errorval


def trainLinearReg(X,y,Lambda):

	itheta = np.zeros(np.size(X,1))

	result = minimize(linearRegCostFunction,itheta, args = (X,y,Lambda)
		,jac = True, options ={'disp' : False, 'maxiter': 400})


	return result.x

def linearRegCostFunction(theta,X,y,Lambda):
	m = y.shape[0]
	J = 0.0
	grad = np.zeros((theta.shape[0]))
	theta = np.squeeze(theta)
	thetareg = np.hstack((0, theta[1:]))
	h = X.dot(theta)
	y = np.squeeze(y)

	J = (1/(2*m))*np.sum((h-y)**2) + (Lambda/(2*m))*(thetareg.T.dot(thetareg))

	grad = (1/m)*(X.T.dot(h-y) + Lambda*thetareg)

	return J , grad

def validationCurve(X,y, Xval,yval):
	lambdavec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
	errortrain = np.zeros(len(lambdavec))
	errorval = np.zeros(len(lambdavec))

	for i in range(len(lambdavec)):
		Lambda = lambdavec[i]
		theta = trainLinearReg(X,y,Lambda)
		errortrain[i] ,grad = linearRegCostFunction(theta,X,y,0)
		errorval[i], grad = linearRegCostFunction(theta,Xval,yval,0)	


	return lambdavec, errortrain, errorval
main()
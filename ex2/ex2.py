from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series
from mpl_toolkits.mplot3d import axes3d
from scipy.optimize import minimize
#for formatting
from decimal import * 

def main():
	
	data = np.loadtxt('ex2data1.txt',delimiter=',')
	X = data[:,0:2]
	y = data[:,2]

	# ==================== Part 1: Plotting ====================
	print("Plotting data...")
	print("+ indicates y = 1, and o indicates y = 0")

	plotData(X,y)
	plt.legend(['Admitted', "Not Admitted"], loc ='upper right', shadow = True, fontsize = 'small',numpoints=1)
	plt.xlabel('Exam 1 score')
	plt.ylabel('Exam 2 score')
	plt.show()


	# ============ Part 2: Compute Cost and Gradient ============

	m, n = X.shape

	X = np.concatenate((np.ones((m,1)),X),axis=1)

	initial_theta = np.zeros(n+1)

	J = costFunction(initial_theta,X,y)
	print("Cost at initial theta: %f" % J)

	grad = gradientFunction(initial_theta,X,y)
	print("Gradient at initial theta: " + str([float(Decimal("%f" % e)) for e in grad]))

	#  ============= Part 3: Optimizing using scipy  ==============

	res = minimize(costFunction, initial_theta, method='TNC',jac=False, args=(X, y), options={'gtol': 1e-3, 'disp': False, 'maxiter': 1000})

	theta = res.x
	cost = res.fun

	print("Cost at theta by scipy: %f" % cost)

	print('theta:', str([float(Decimal("%f" % e)) for e in theta]))

	plotDecisionBoundary(theta,X,y)

	plt.legend(['Admitted','Not Admitted'], loc='upper right', shadow=True, fontsize= 'small',numpoints=1)
	plt.xlabel('Exam 1 score')
	plt.ylabel('Exam 2 score')
	plt.show()


	#  ============== Part 4: Predict and Accuracies ==============

	prob = sigmoid(np.array([1,45,85]).dot(theta))
	print ("For a student with scores 45 and 85, we predict an admission probability of :" + str(prob))

	p = predict(theta, X)

	acc = 1.0*np.where(p==y)[0].size/len(p)*100
	print("Train accuracy : %f" % acc)


def plotData(X,y):
	pos = X[np.where(y==1,True,False)]
	neg = X[np.where(y==0,True,False)]
	plt.plot(pos[:,0],pos[:,1],'+',markersize=7,markeredgecolor='black',markeredgewidth=2)
	plt.plot(neg[:,0],neg[:,1],'o',markersize=7,markeredgecolor='black',markerfacecolor='yellow')

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
		z = [np.array([mapFeature(u[i],v[j]).dot(theta) for i in range(len(u))]) for j in range(len(v))]
		plt.contour(u,v,z,levels=[0,0])

def mapFeature(X1,X2,degree=6):
	quads = Series([X1**(i-j) * X2**j for i in range(1,degree+1) for j in range(i+1)])
	return Series(([1]).append([Series(X1),Series(X2),quads]))




#split this into two functions next
def costFunction(theta,X,y):
	m = y.shape[0]
	J=0.0
	h = sigmoid(X.dot(theta))

	J = (1/m)*np.sum((-y)*np.log(h) - (1-y)*np.log(1-h))

	return J
	
def gradientFunction(theta,X,y):
	m = y.shape[0]
	grad = np.zeros(theta.size)
	h = sigmoid(X.dot(theta))

	grad = (1/m)*(X.T.dot(h-y))

	return grad



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
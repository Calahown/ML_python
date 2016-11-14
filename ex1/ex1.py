from matplotlib import use, cm
use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d

def main():

	# ==================== Part 1: Basic Function ====================
	print("Running warmUpExcercise \n")
	print("5x5 Identity Matrix \n")
	I = warmUpExercise()
	print(I)

	print("Plotting Data \n")



	# ======================= Part 2: Plotting =======================
	data = np.loadtxt('ex1data1.txt', delimiter=',')

	m = data.shape[0]
	X = np.vstack(zip(np.ones(m),data[:,0]))
	y = data[:,1]
	plotData(data)

	# =================== Part 3: Gradient descent ===================
	print("Running Gradientdecent")
	theta = np.zeros(2)
	J = computeCost(X,y,theta)
	print ('cost: %0.4f ' % J)
	iterations = 1500
	alpha = 0.01
	theta , J_history = gradientDecent(X,y,theta,alpha,iterations)
	print('theta: ' + str(theta))
	plt.plot(X[:, 1], X.dot(theta), '-', label='Linear regression')
	plt.show()

	predict1 = np.array([1,3.5]).dot(theta)
	predict2 = np.array([1,7]).dot(theta)

	print('Profit prediction for a population of 35,000 is ' + str(predict1*10000))
	print('Profit prediction for a population of 70,000 is ' + str(predict2*10000))


	# ============= Part 4: Visualizing J(theta_0, theta_1) =============

	print("Visualizing J(theta_0, theta_1)")

	theta0_vals = np.linspace(-10,10, X.shape[0])
	theta1_vals = np.linspace(-1, 4, X.shape[0])

	J_vals = np.array(np.zeros(X.shape[0]).T)

	for i in range(theta0_vals.size):
		column = []
		for j in range(theta1_vals.size):
			t = np.array([theta0_vals[i],theta1_vals[j]])
			column.append(computeCost(X,y,t))
		J_vals = np.column_stack((J_vals,column))


	J_vals = J_vals[:,1:].T

	theta0_vals, theta1_vals = np.meshgrid(theta0_vals,theta1_vals)

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_surface(theta0_vals,theta1_vals,J_vals, rstride = 10, cstride = 10, alpha = 0.3, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
	ax.set_xlabel(r'$\theta_0$')
	ax.set_ylabel(r'$\theta_1$')
	ax.set_zlabel(r'J$\theta$')

	plt.show()

	fig = plt.figure()
	ax = plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2,3,20))
	plt.clabel(ax,inline=1,fontsize=9)
	plt.xlabel(r'$\theta_0$')
	plt.ylabel(r'$\theta_1$')
	plt.plot(0.0,0.0,'rx',linewidth=2,markersize=10)
	plt.show()




def warmUpExercise():
	I = np.identity(5)
	return I

def plotData(data):
	plt.figure()
	plt.scatter(data[:,0],data[:,1])
	plt.axis([0,25,0,25])
	#plt.show()

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


main()
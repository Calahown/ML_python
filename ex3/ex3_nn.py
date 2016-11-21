from matplotlib import use
use('TkAgg')
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def main():

	input_layer_size = 400
	hidden_layer_size = 25
	numlabels = 10

	# =========== Part 1: Loading and Visualizing Data =============

	print("Loading and Visualizing Data")
	data = scipy.io.loadmat('ex3data1.mat')
	X = data['X']
	y = data['y']
	m,n = X.shape
	sel = np.random.permutation(range(m))
	selection = X[sel[0:100],:]
	displayData(selection)

	# ================ Part 2: Loading Pameters ================

	print("Loading Saved Neural Network Parameters")

	data = scipy.io.loadmat('ex3weights.mat')
	theta1 = data['Theta1']
	theta2 = data['Theta2']

	# ================= Part 3: Implement Predict =================

	pred = predict(theta1, theta2, X)

	acc = np.mean(np.double(pred == np.squeeze(y)))*100
	print("Training set accuracy : %f" % acc)

	rp = np.random.permutation(range(m))

	plt.figure()
	#changed to 10 random examples
	for i in range(10):
		Xtemp = X[rp[i],:]
		print ("Example Image")
		Xtemp = np.matrix(X[rp[i]])
		displayData(Xtemp)

		pred = predict(theta1,theta2,Xtemp.getA())
		pred = np.squeeze(pred)
		print ('previous predicteed value: %d (digit %d)' % (pred, np.mod(pred,10)))
		plt.close()


def predict(theta1, theta2, X):
	m = X.shape[0]
	numlabels = theta2.shape[0]

	p = np.zeros((m,1))

	a1 = np.hstack((np.ones((m,1)), X))
	z2 = a1.dot(theta1.T)
	a2 = np.hstack((np.ones((z2.shape[0],1)), sigmoid(z2)))
	z3 = a2.dot(theta2.T)
	a3 = sigmoid(z3)
	p = np.argmax(a3, axis = 1) + 1 
	print(p)

	return p

def sigmoid(z):
	g = np.zeros(z.size)
	g = 1/(1 + np.exp(-z))

	return g


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
from matplotlib import use
use('TkAgg')
import numpy as np
import scipy.io 
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():

	# ================== Part 1: Load Example Dataset  ===================
	
	print("Visualizing example data for PCA")
	data = scipy.io.loadmat('ex7data1.mat')
	X = data['X']

	plt.scatter(X[:,0], X[:,1], marker = 'o')
	plt.axis([0.5,6.5,2,8])
	plt.axis('equal')

	# =============== Part 2: Principal Component Analysis ===============

	print("Running PCA on example dataset")

	Xnorm , mu, sigma = featureNormalize(X)

	U, S= pca(Xnorm)
	S = np.diag(S)
	moo = mu + 1.5* S.dot(U.T)

	plt.plot([mu[0], moo[0,0]],[mu[1], moo[0,1]], '-k' , linewidth = 2)
	plt.plot([mu[0], moo[1,0]],[mu[1], moo[1,1]], '-k' , linewidth = 2)
	plt.show()

	print("Top eigenvector:")
	print("U[:,0] = " + str(U[:,0]))
	print("these values are expected to be -0.707107 and -0.707107")

	# =================== Part 3: Dimension Reduction ===================

	print("Dimension reduction on example dataset")

	plt.scatter(Xnorm[:,0],Xnorm[:,1], marker = 'o')
	plt.axis([-4,3,-4,3])
	plt.axis('equal')

	K = 1
	Z = projectData(Xnorm, U, K)

	print("Projection of the first example: %f" % Z[0])
	print("This value should be about 1.481274")

	Xrec = recoverData(Z,U,K)
	print("Approximation of the first example: " + str(Xrec[0]))
	print("This value should be about [-1.047419, -1.047419]")

	plt.scatter(Xrec[:, 0], Xrec[:, 1], marker='o', color='r', lw=1.0)
	for i in range(len(Xnorm)):
		plt.plot([Xnorm[i, 0], Xrec[i, 0]], [Xnorm[i, 1], Xrec[i, 1]], '--b')
	plt.show()

	# =============== Part 4: Loading and Visualizing Face Data =============

	print("Loading face dataset")

	data = scipy.io.loadmat('ex7faces.mat')

	X = data['X']

	displayData(X[0:100,:])
	plt.show()

	# =========== Part 5: PCA on Face Data: Eigenfaces  ===================

	print("Running PCA on face dataset")
	print("(this might take one or two minutes)")

	Xnorm, mu, sigma = featureNormalize(X)

	U,S = pca(Xnorm)

	displayData(U[:,:36].T)
	plt.show()

	# ============= Part 6: Dimension Reduction for Faces =================

	print("Dimension reduction for face dataset")
	K = 100
	Z = projectData(Xnorm, U, K)
	print("The projected data Z has a size of:" + str(Z.shape))

	# ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====

	print("Visualizing the projected (reduced dimension) faces")
	Xrec = recoverData(Z,U,K)

	plt.subplot(1,2,1)
	plt.axis('equal')
	displayData(Xnorm[:100,:])
	plt.title('Original')

	plt.subplot(1,2,2)
	plt.title('Reduced')
	plt.axis('equal')
	displayData(Xrec[:100,:])
	plt.show()





def recoverData(Z,U,K):
	Xrec = np.zeros((len(Z), len(U)))
	Ureduce = U[:,:K]
	for i in range(len(Z)):
		Xrec[i,:] = (Ureduce.dot(Z[i,:].T)).T
	return Xrec


def projectData(X,U,K):
	Ureduce = U[:,:K]
	Z = np.zeros((len(X), K))
	for i in range(len(X)):
		Z[i,:] = (Ureduce.T.dot(X[i,:].T)).T
	return Z



def pca(X):
	m = X.shape[0]
	sigma = (1/m)*(X.T.dot(X))
	U,S,V = np.linalg.svd(sigma)

	return U, S


def featureNormalize(X):
	mu = np.mean(X,axis=0)
	Xnorm = X-mu
	sigma = np.std(Xnorm, axis = 0, ddof=1)
	Xnorm = Xnorm/sigma

	return Xnorm, mu, sigma


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



main()
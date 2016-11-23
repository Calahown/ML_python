from matplotlib import use, cm
use ('TkAgg')
import numpy as np
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
import itertools

def main():

	# ================= Part 1: Find Closest Centroids ====================

	print("Finding closest centroids")

	data = scipy.io.loadmat('ex7data2.mat')
	X = data['X']
	K = 3
	icentroids = np.array([[3,3], [6,2] , [8,5]])
	val , idx = findClosestCentroids(X, icentroids)
	print("Closest centroids for the first 3 examples")
	print(idx[:3])
	print("Closest three should be 0 ,2 ,1 respectively")

	# ===================== Part 2: Compute Means =========================

	print("Computing centroids means")

	centroids = computeCentroids(X, idx, K)

	print("Centroids computed after initial finding of closest centroids")
	print(centroids)
	print("These centroids should be close to:")
	print('[ 2.428301 3.157924 ]')
	print('[ 5.813503 2.633656 ]')
	print('[ 7.119387 3.616684 ]')

	# =================== Part 3: K-Means Clustering ======================

	print("Running K-means clustering on example dataset")
	# items already in X, k and icentroids
	max_iters = 10

	centroids, idx = runkMeans(X,icentroids, max_iters, True)

	print("K-means done")

	# ============= Part 4: K-Means Clustering on Pixels ===============

	print("Running K-means clustering on pixels from an image")

	image = scipy.misc.imread('bird_small.png')

	image = image/255.0

	imgsize = image.shape

	imgreshape = image.reshape(imgsize[0] * imgsize[1], 3)

	K = 16
	max_iters = 10

	icentroids = kMeansInitCentroids(imgreshape,K)

	centroids, idx = runkMeans(imgreshape,icentroids,max_iters)

	# ================= Part 5: Image Compression ======================

	print("Applying Kmeans to compress an image")

	throwaway, idx = findClosestCentroids(imgreshape, centroids)

	Xrecovered = np.array([centroids[int(e)] for e in idx])
	Xrecovered = Xrecovered.reshape(imgsize[0], imgsize[1],3)

	plt.subplot(1,2,1)
	plt.imshow(image)
	plt.title('Original')
	plt.subplot(1,2,2)
	plt.imshow(Xrecovered)
	plt.title("Compressed, with %d colors." % K)
	plt.show()


def kMeansInitCentroids(X,K):
	centroids = np.zeros((K,X.shape[1]))

	randidx = np.random.permutation(X.shape[0])
	centroids = X[randidx[:K], :]

	return centroids



def runkMeans(X,icentroids, max_iters, plot_progress = False):

	if plot_progress:
		plt.figure()

	m,n = X.shape
	K = len(icentroids)
	centroids = icentroids
	pcentroids = centroids
	idx = np.zeros(m)

	for i in range(max_iters):
		print("K-means iteration # :" +str(i+1) + '/' + str(max_iters))

		val , idx = findClosestCentroids(X, centroids)

		if plot_progress:
			plotprogressKmeans(X, np.array(centroids), np.array(pcentroids), idx, K, i, 'k')
			pcentroids = centroids

		centroids = computeCentroids(X,idx, K)

	if plot_progress:
		plt.show()

	return centroids, idx


def plotprogressKmeans(X, centroids, pcentroids, idx, K, i, color):
	plotDataPoints(X,idx)

	plt.scatter(centroids[:,0], centroids[:,1], marker = 'x', linewidth = 5, edgecolors = 'r' , c = 'k')
	for j in range(len(centroids)):
		plt.plot([centroids[j,0], pcentroids[j,0]], [centroids[j,1], pcentroids[j,1]], c = color)

	plt.title("Iteration number")


def plotDataPoints(X,idx):
	palette = plt.get_cmap("hsv")
	idxm = (idx.astype('float')+1)/(max(idx.astype('float'))+1)
	colors = palette(idxm)
	plt.scatter(X[:,0],X[:,1], 15, c = colors , marker = 'o', linewidth = 0.5)



def computeCentroids(X, idx, K):
	m,n = X.shape
	centroids = np.zeros((K,n))
	xidx = np.hstack((idx[:,None],X))
	for i in range(K):
		centroids[i] = np.mean(X[xidx[:,0] == i], axis = 0)

	#for k in range(K):
	# 	instances = 0
	# 	sum = np.zeros((n,1))
	# 	for i in range(m):
	# 		if (idx[i] == k):
	# 			centroids[k,:] = centroids[k,:] + X[i,:]
	# 			instances = instances +1

	#centroids[k] = (centroids[k]/instances)
	return centroids

def findClosestCentroids(X, centroids):
	k = centroids.shape[0]
	idx = np.zeros(X.shape[0])

	for i in range(X.shape[0]):
		mindistance = np.inf
		for j in range(k):
			distance = X[i,:].T - centroids[j,:].T
			dtwo = distance.T.dot(distance)
			if (dtwo < mindistance):
				idx[i] = j
				mindistance = dtwo

	return dtwo, idx

main()
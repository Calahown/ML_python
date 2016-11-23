import numpy as np
import scipy.io
import re
import string
from sklearn import svm
from collections import OrderedDict
from porterstemmer import porterStemmer 


def main():

	# ==================== Part 1: Email Preprocessing ====================

	print("Preprocessing sample email (emailSample1.txt)")

	file = open('emailSample1.txt','r')
	file_contents = file.readlines()
	word_indeces = processEmail(''.join(file_contents))

	print("Word Indeces:")
	print(word_indeces)


	# ==================== Part 2: Feature Extraction ====================

	print("Extracting features from sample email (emailSample1.txt)")

	features = emailFeatures(word_indeces)
	print("Length of feature vector:" + str(len(features)))
	print("Number of non-zero entries:" + str(np.sum(features > 0)))


	file.close()


	# =========== Part 3: Train Linear SVM for Spam Classification ========

	data = scipy.io.loadmat('spamTrain.mat')
	X = data['X']
	y = data['y'].flatten()

	print("Training Linear SVM (Spam Classification)")
	print("(this may take 1 to 2 minutes)")

	C= 0.1
	clf = svm.SVC(C=C, kernel = 'linear', tol = 1e-3)
	model = clf.fit(X,y)

	p = model.predict(X)

	print ("Training Accuracy: %f", np.mean(np.double(p==y)) * 100)

	# =================== Part 4: Test Spam Classification ================

	print("Evaluating the trained linear SVM on a test set")

	data = scipy.io.loadmat('spamTest.mat')
	Xtest = data['Xtest']
	ytest = data['ytest'].flatten()

	p = model.predict(Xtest)

	print("Test Accuracy: %f", np.mean(np.double(p ==ytest))* 100)

	# ================= Part 5: Top Predictors of Spam ====================

	t = sorted(list(enumerate(model.coef_[0])), key = lambda e: e[1], reverse = True)
	d = OrderedDict(t)
	idx = list(d.keys())
	weight = list(d.values())
	vocabList = getVocablist()

	print("Top predictors of spam:")
	for i in range(15):
		print("%-15s (%f)" %(vocabList[idx[i]], weight[i]))

	# =================== Part 6: Try Your Own Emails =====================

	file = open('spamSample1.txt')
	file_contents = file.readlines()
	word_indeces = processEmail(''.join(file_contents))
	x = emailFeatures(word_indeces)
	p = model.predict(x.reshape(1,-1))

	print('Processed %s\n\n Spam Classification: %d' % (file, p))
	print('1 for spam, 0 for not spam')

	file.close()



def emailFeatures(word_indeces):
	n = 1899
	x = np.zeros(n)

	for i in word_indeces:
		x[i] = 1
	
	return x


def processEmail(email_contents):
	vocabList = getVocablist()

	word_indeces = []

	email_contents = email_contents.lower()
	remove = re.compile('<[^<>]+>|\n')
	email_contents = remove.sub(' ', email_contents)
	remove = re.compile('[0-9]+')
	email_contents = remove.sub('number ', email_contents)
	remove = re.compile('(http|https)://[^\s]*')
	email_contents = remove.sub('httpaddr ', email_contents)
	remove = re.compile('[^\s]+@[^\s]+')
	email_contents = remove.sub('emailaddr ', email_contents)
	remove = re.compile('[$]+')
	email_contents = remove.sub('dollar ', email_contents)

	print ("Processed Email")

	l = 0

	remove = re.compile('[^a-zA-Z0-9 ]')
	email_contents = remove.sub('', email_contents).split()

	for stringer in email_contents:

		#stringer = re.split('[' + re.escape(' @$/#.-:&*+=[]?!(){},''">_<#')+ chr(10) + chr(13) + ']', stringer)

		try:
			stringer = porterStemmer(stringer.strip())
		except:
			stringer = ''
			continue

		if len(stringer) <1:
			continue

		for i in range(len(vocabList)):
			if stringer == vocabList[i]:
				word_indeces.append(i)

		if (l + len(stringer) + 1) > 78:
			print('\n')
			l = 0;
		else:
			print(stringer, end = ' ')
			l = l + len(stringer)+ 1

	print('\n\n=========================\n')

	return np.array(word_indeces)



def getVocablist():
	with open ('vocab.txt') as f:

		vocabList = []
		for line in f:
			idx,w = line.split()
			vocabList.append(w)

	f.close()

	return vocabList


main()
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.special import expit 


def sigmoid(z):
    return expit(z)

# Hypothesis function and cost function for logistic regression
def h(mytheta,myX): # Logistic hypothesis function
    return sigmoid(myX.dot(mytheta))

# Cost function
def computeCost(mytheta,myX,myy,mylambda=0):
    """
    mytheta is an n-dimensional vector of initial theta guess
    myX is matrix with m-rows and n-columns
    myy is a matrix with m-rows and 1-column
    """
    term1 = np.dot(-np.array(myy).T,np.log(h(mytheta,myX)))
    term2 = np.dot((1-np.array(myy)).T,np.log(1-h(mytheta,myX)))
    regterm = (mylambda/2) * np.sum(np.dot(mytheta[1:].T,mytheta[1:]))
    return float( (1./m) * ( np.sum(term1 - term2) + regterm) )

def optimizeTheta(mytheta,myX,myy,mylambda=0):
    
    result = optimize.fmin(computeCost, x0=mytheta, args=(myX, myy), maxiter=10000, full_output=True)
    return result[0], result[1]

def makePrediction(mytheta, myx):
    result = sigmoid(myx.dot(mytheta.T)) >= 0.8
    return result

datafile = 'data/training_processed.csv'
# Read in comma separated data in datafile
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),unpack=True)

# Form the usual "X" data matrix and "y" label vector
X = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))
m = y.size # number of training examples

# Insert the usual column of 1's into the "X" matrix (for bias incorporation)
X = np.insert(X,0,1,axis=1)


pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])

initial_theta = np.zeros((X.shape[1],1))
computeCost(initial_theta,X,y)
theta, mincost = optimizeTheta(initial_theta,X,y)

# Compute the percentage of samples I got correct:
pos_correct = float(np.sum(makePrediction(theta,pos)))
neg_correct = float(np.sum(np.invert(makePrediction(theta,neg))))
tot = len(pos)+len(neg)
prcnt_correct = float(pos_correct+neg_correct)/tot
print("Fraction of training samples correctly predicted: %f." % prcnt_correct)

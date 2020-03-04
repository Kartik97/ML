import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import axes3d, Axes3D
import time

# Function for normalizing the data
def normalize(X):
    m = X.shape[0]
    mean = np.sum(X,axis=0)/m
    var = np.sum((X-mean)**2,axis=0)/m
    sd = np.sqrt(var)
    X = (X-mean)/sd
    return X
    
# Initializing the parameters
def init():
    X = np.array(pd.read_csv("../ass1_data/data/q3/logisticX.csv",header=None))
    X = normalize(X)
    X = np.append(np.ones((X.shape[0],1)),X,axis=1)
    Y = np.array(pd.read_csv("../ass1_data/data/q3/logisticY.csv",header=None))
    param = np.array([[0],[0],[0]])
    return X,Y,param

# Hypothesis function
def hyp(X,param):
    return 1/(1+np.exp((-1)*np.dot(X,param)))

# Newton's Method for convergence
def newton(X,Y,p,epsilon=1e-12,steps=10000):
    conv = False
    i=0
    while(not conv):
        fd = np.dot((Y-hyp(X,p)).T,X).T
        factor = hyp(X,p)*(1-hyp(X,p))
        H = (-1)*np.dot((X*factor).T,X)
        diff = np.dot(np.linalg.inv(H),fd)
        pNext = p-diff
        i=i+1
        if(i>steps or abs(pNext-p).all() < epsilon):
            p = pNext
            conv=True
        p = pNext
        i = i+1
    return p

if __name__ == '__main__':

    # PART A----Learning the parameters
    X,Y,param = init()
    theta = newton(X,Y,param,steps=1000)
    print(theta)

    # PART B----Plotting the linear boundary created by the parameters

    yPred = np.dot(X,theta)
    yPred[yPred>0.5] = 1
    yPred[yPred<=0.5] = 0

    pos = []
    neg = []
    for i in range(0,Y.shape[0]):
        if(Y[i]==1):
            pos.append(X[i,1:3])
        else:
            neg.append(X[i,1:3])
    pos = np.array(pos)
    neg = np.array(neg)

    fig = plt.figure(0)
    ax = plt.gca()
    ax.scatter(pos[:,0:1],pos[:,1:2],color="blue",marker="+",label="Positive Examples")
    ax.scatter(neg[:,0:1],neg[:,1:2],color="red",marker=".",label="Negative Examples")
    ax.set_title("Decision Boundary")
    ax.set_xlabel("Feature A")
    ax.set_ylabel("Feature B")
    lineX1 = X[:,1:2]
    lineX2 = -(theta[0][0]+theta[1][0]*lineX1)/theta[2][0]
    
    plt.plot(lineX1,lineX2,color="green",label="Decision Boundary")
    ax.legend(loc="upper left")

    plt.show()
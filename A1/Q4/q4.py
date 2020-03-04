import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function for normalizing the data
def normalize(X):
    m = X.shape[0]
    mean = np.sum(X,axis=0)/m
    var = np.sum((X-mean)**2,axis=0)/m
    sd = np.sqrt(var)
    X = (X-mean)/sd
    return X

# Initilizing the parameters
def init():
    dataX = np.loadtxt('../ass1_data/data/q4/q4x.dat')
    dataY = np.reshape(np.genfromtxt('../ass1_data/data/q4/q4y.dat',dtype=None,encoding='ascii'),(dataX.shape[0],-1))
    dataX = normalize(dataX)
    return dataX,dataY

# Function for separating the classes
# 0 = Alaska    1 = canada
def seperateData(dataX,dataY):
    alaska=[]
    canada=[]
    for i in range(dataY.shape[0]):
        if(dataY[i]=="Alaska"):
            alaska.append(dataX[i])
        else:
            canada.append(dataX[i])
    phi = len(alaska)/dataY.shape[0]
    return phi,np.asarray(alaska),np.asarray(canada)

# Calculating mean of data
def calculateMean(data):
    return np.sum(data,axis=0)/data.shape[0]

# Pass data from which mean has already been subtracted to calculate Covariance
def calculateCovariance(data):
    return np.dot(data.T,data)/data.shape[0]

# Calculating Coefficients of the linear equation formed
def calculateCoeffLinear(phi,m0,m1,c0,c1):
    temp = (np.dot(np.dot(m0,np.linalg.inv(c0)),m0.T)-np.dot(np.dot(m1,np.linalg.inv(c1)),m1.T))/2
    intercept = np.log(phi/(1-phi)) + temp
    coeff = np.dot(np.linalg.inv(c1),m1.T)-np.dot(np.linalg.inv(c0),m0.T)
    return intercept[0],coeff[0],coeff[1]

# Finding the linear boundary created
def calLinBound(X,_int,A,C):
    return (_int-X*A)/C

# Calculating the quadratic coefficients
def calculateCoeffQuad(phi,m0,m1,c0,c1):
    temp = (np.dot(np.dot(m0,np.linalg.inv(c0)),m0.T)-np.dot(np.dot(m1,np.linalg.inv(c1)),m1.T))/2
    c = np.log(phi/(1-phi)) + np.log(np.linalg.det(c0)/np.linalg.det(c1)) + temp
    t1 = np.dot(np.linalg.inv(c1),m1.T)-np.dot(np.linalg.inv(c0),m0.T)
    t2 = np.linalg.inv(c1)-np.linalg.inv(c0)
    return t2,t1,c

# Finding the quadratic boundary formed
def calQuadBound(X,t2,t1,c1):
    a = t2[1][1]
    b = X*(t2[1][0]+t2[0][1]) - 2*t1[1][0]
    c = t2[0][0]*(X**2) - 2*X*t1[0][0] - 2*c1
    roots = []
    for i in range(len(X)):
        roots.append(max(np.roots([a,b[i],c[0][i]])))
    return roots

if __name__ == "__main__":

    dataX,dataY = init()

    # PART A----Implementing GDA and finding the values of mean and covariance matrix

    phi,alaska,canada = seperateData(dataX,dataY)
    mu0 = np.reshape(calculateMean(alaska),(-1,2))
    mu1 = np.reshape(calculateMean(canada),(-1,2))
    covData = np.append(alaska-mu0,canada-mu1,axis=0)
    covMat = calculateCovariance(covData)
    print("Mean0 = ",mu0)
    print("Mean1 = ",mu1)
    print("Covariance Matrix = ",covMat)

    # PART B----Plotting the data avaialable

    fig0 = plt.figure(0)
    ax0 = plt.gca()
    ax0.scatter(alaska[:,0:1],alaska[:,1:2],color="blue",marker=".",label="Alaska")
    ax0.scatter(canada[:,0:1],canada[:,1:2],color="red",marker="+",label="Canada")
    ax0.legend(loc="upper left")
    ax0.set_xlabel("Feature 1")
    ax0.set_ylabel("Feature 2")
    ax0.set_title("Given Data")

    # PART C----Plotting the linear boundary formed by the parameters

    _intL,pAL,pCL = calculateCoeffLinear(phi,mu0,mu1,covMat,covMat)
    fig1 = plt.figure(1)
    ax1 = plt.gca()
    ax1.scatter(alaska[:,0:1],alaska[:,1:2],color="blue",marker=".",label="Alaska")
    ax1.scatter(canada[:,0:1],canada[:,1:2],color="red",marker="+",label="Canada")
    x1 = np.linspace(-1.7,2,100)
    y1 = calLinBound(x1,_intL,pAL,pCL)
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.set_title("Linear Boundary")
    plt.plot(x1,y1,color="black",label="Linear Boundary")
    ax1.legend(loc="upper left")

    # PART D----Calculating the covariance matrices in the case when they are not equal

    covMat0 = calculateCovariance(alaska-mu0)
    covMat1 = calculateCovariance(canada-mu1)
    print("Mean0 = ",mu0)
    print("Mean1 = ",mu1)
    print("Covariance Matrix0 = ",covMat0)
    print("Covariance Matrix1 = ",covMat1)

    # PART E----Plotting the quadratic boundary formed

    t2,t1,c = calculateCoeffQuad(phi,mu0,mu1,covMat0,covMat1)

    fig2 = plt.figure(2)
    ax2 = plt.gca()
    ax2.scatter(alaska[:,0:1],alaska[:,1:2],color="blue",marker=".",label="Alaska")
    ax2.scatter(canada[:,0:1],canada[:,1:2],color="red",marker="+",label="Canada")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")
    ax2.set_title("Quadratic Boundary")
    x1 = np.linspace(-1.7,2,100)
    y1 = calLinBound(x1,_intL,pAL,pCL)

    plt.plot(x1,y1,color="black",label="Linear Boundary")

    x2 = np.linspace(-2,1.7,100)
    y2 = calQuadBound(x2,t2,t1,c)

    plt.plot(x2,y2,color="green",label="Quadratic Boundary")
    ax2.legend(loc="upper left")
    plt.show()
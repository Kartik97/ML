import numpy as np
import pandas as pd
import cvxopt
from cvxopt import solvers,matrix
from sklearn.svm import SVC
import math
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
import itertools
import threading
import pickle
from joblib import Parallel, delayed
import time
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold,cross_val_score,StratifiedKFold

def init():
    train = pd.read_csv("fashion_mnist/train.csv",header=None)
    val = pd.read_csv("fashion_mnist/val.csv",header=None)
    test = pd.read_csv("fashion_mnist/test.csv",header=None)
    return train,val,test

def stripData(data,c1,c2):
    data = data[data[784]==c1].append(data[data[784]==c2])
    data.loc[data[784]==c1,784]=-1
    data.loc[data[784]==c2,784]=1
    Y = np.reshape(np.array(data[784]),(-1,1))
    pos = data[data[784]==1]
    neg = data[data[784]==-1]
    data = np.array(data.drop(columns=[784]))/255
    return data,Y,pos,neg

def createPq(data,out):
    P = (out@out.T)*(data@data.T)
    q = (-1)*np.ones((P.shape[0],1))
    return matrix(P),matrix(q)

def gaussianPq(data,out):
    m = data.shape[0]
    kernelMatrix=np.exp((-0.05)*np.square(distance.cdist(data,data,'euclidean')))
    P = (out@out.T)*kernelMatrix
    q = (-1)*np.ones((P.shape[0],1))
    return matrix(P),matrix(q)

def createGh(m):
    u = np.identity(m)
    d = (-1)*np.identity(m)
    G = np.append(u,d,axis=0)
    h = np.append(np.ones((m,1)),np.zeros((m,1)),axis=0)
    return matrix(G),matrix(h)

def createAb(out):
    m = out.shape[0]
    A = out.copy()
    A = A.T
    b = 0
    return matrix(A,(1,m),'d'),matrix(b,(1,1),'d')

def stripAlpha(alpha,E):
    index = alpha < E
    alpha[index] = 0
    return alpha

def calwb(alpha,X,Y,pos,neg):
    w = X.T@(alpha*Y)
    x = (np.array(pos.drop(columns=[784]))/255)
    y = (np.array(neg.drop(columns=[784]))/255)
    b = (-1)*((x@w).min()+(y@w).max())/2
    return w,b

def calb(alpha,X,Y,pos,neg):
    x = (np.array(pos.drop(columns=[784]))/255)
    y = (np.array(neg.drop(columns=[784]))/255)
    Y.reshape((-1,1))
    p = ((alpha*Y)*(np.exp((-0.05)*np.square(distance.cdist(X,x,'euclidean'))))).sum(axis=0).min()
    n = ((alpha*Y)*(np.exp((-0.05)*np.square(distance.cdist(X,y,'euclidean'))))).sum(axis=0).max()
    b = (-1)*(p+n)/2
    return b

def predClasses(w,b,data):
    pred = []
    for i in data:
        temp = i.reshape((1,-1))@w+b
        if temp<0:
            pred.append(-1)
        else:
            pred.append(1)
    return pred

def predGaussianClasses(alpha,b,X,Y,data,return_score=False):
    pred=[]
    pred = ((alpha*Y)*(np.exp((-0.05)*np.square(distance.cdist(X,data,'euclidean'))))).sum(axis=0)
    pred = pred+b
    if(return_score):
        return pred
    index = pred < 0
    pred[index]=-1
    index = pred >= 0
    pred[index]=1
    return pred

def checkAccuracy(y,pred):
    y = np.array(y).reshape((-1,1))
    pred = np.array(pred).reshape((-1,1))
    check = (y==pred)
    t=0
    f=0
    for i in check:
        if i:
            t=t+1
        else:
            f=f+1
    print(t/(t+f))

if __name__=="__main__":

    # PART 1(A)
    train,val,test = init()
    c1=3
    c2=4

    train = train[train[784]==c1].append(train[train[784]==c2])
    train.loc[train[784]==c1,784]=-1
    train.loc[train[784]==c2,784]=1
    pos = train[train[784]==1]
    neg = train[train[784]==-1]
    Y = np.array(train[784])
    trainD = np.array(train.drop(columns=[784]))/255
    Y = Y.reshape((Y.shape[0],1))

    t = time.time()
    P,q = createPq(trainD,Y)
    G,h = createGh(trainD.shape[0])
    A,B = createAb(Y)
    sol=solvers.qp(P, q, G, h, A, B)
    alpha = np.array(sol['x'])
    alpha = stripAlpha(alpha,1e-4)

    index = alpha != 0
    print("Number of support Vectors",index.sum())

    w,b = calwb(alpha,trainD,Y,pos,neg)
    print("w,b in linear kernel",w,b)
    print("Linear Kernel Training",time.time()-t)

    predictionTrain = predClasses(w,b,trainD)
    print("Accuracy on Train using Linear Kernel")
    checkAccuracy(train[784],predictionTrain)

    testX,testY,_,_ = stripData(test,c1,c2)
    valX,valY,_,_ = stripData(val,c1,c2)

    predictionTest = predClasses(w,b,testX)
    print("Accuracy on Test using Linear Kernel")
    checkAccuracy(testY,np.array(predictionTest))

    predictionVal = predClasses(w,b,valX)
    print("Accuracy on Val using Linear Kernel")
    checkAccuracy(valY,predictionVal)

    # PART 1(B)

    t = time.time()
    P,q = gaussianPq(trainD,Y)
    G,h = createGh(trainD.shape[0])
    A,B = createAb(Y)
    sol=solvers.qp(P, q, G, h, A, B)
    alpha = np.array(sol['x'])
    alpha = stripAlpha(alpha,1e-4)
    
    index = alpha != 0
    print("No of support vectors in Gaussian Kernel",index.sum())
    b = calb(alpha,trainD,Y,pos,neg)
    print("b on gaussian kernel",b)
    print("Gaussian Kernel Training",time.time()-t)

    predictionTrain = predGaussianClasses(alpha,b,trainD,Y,trainD)
    predictionTest = predGaussianClasses(alpha,b,trainD,Y,testX)
    predictionVal = predGaussianClasses(alpha,b,trainD,Y,valX)

    print("Gaussian Accuracy on Train")
    checkAccuracy(Y,predictionTrain)
    print("Gaussian Accuracy on Test")
    checkAccuracy(testY,predictionTest)
    print("Gaussian Accuracy on Val")
    checkAccuracy(valY,predictionVal)

    # PART 1(C)

    clfLinear = SVC(C=1.0,kernel='linear')
    clfGaussian = SVC(C=1.0,kernel='rbf',gamma=0.05)
    clfLinear.fit(trainD, train[784])
    clfGaussian.fit(trainD, train[784])

    print("SVM linear kernel support vectors",clfLinear.support_vectors_.shape)
    print("SVM gaussian kernel support vectors",clfGaussian.support_vectors_.shape)

    testX,testY,_,_ = stripData(test,c1,c2)
    valX,valY,_,_ = stripData(val,c1,c2)

    predictionTrain = clfLinear.predict(trainD)
    predictionTest = clfLinear.predict(testX)
    predictionVal = clfLinear.predict(valX)

    print("On Linear Kernel")
    print("Train")
    checkAccuracy(Y,predictionTrain)
    print("Test")
    checkAccuracy(testY,predictionTest)
    print("Val")
    checkAccuracy(valY,predictionVal)

    predictionTrain = clfGaussian.predict(trainD)
    predictionTest = clfGaussian.predict(testX)
    predictionVal = clfGaussian.predict(valX)

    print("On Gaussian Kernel")
    print("Train")
    checkAccuracy(Y,predictionTrain)
    print("Test")
    checkAccuracy(testY,predictionTest)
    print("Val")
    checkAccuracy(valY,predictionVal)

    # PART 2(A)

    
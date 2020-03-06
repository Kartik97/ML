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

def createClf(data,classes):
    c1 = classes[0]
    c2 = classes[1]
    filteredX,filteredY,pos,neg = stripData(data,c1,c2)

    P,q = gaussianPq(filteredX,filteredY)
    G,h = createGh(filteredX.shape[0])
    A,b = createAb(filteredY)
    sol=solvers.qp(P, q, G, h, A, B)
    alpha = np.array(sol['x'])
    alpha = stripAlpha(alpha,1e-4)
    print("Alpha Done.")
    b = calb(alpha,filteredX,filteredY,pos,neg)
    print("b Done.")
    return (alpha,b,c1,c2)

def classify(paraList,X,data):
    pred=[]
    for i in tqdm(paraList):
        filteredX,filteredY,pos,neg = stripData(X,i[2],i[3])
        pred.append(predGaussianClasses(i[0],i[1],filteredX,filteredY,data,return_score=True))
    return pred

def findMajorityClass(predictions,l,calculations='cal'):
    pred = []
    n = len(predictions)
    m = len(predictions[0])
    for i in tqdm(range(m)):
        temp = np.zeros((10,2))
        for j in range(n):
            if(calculations=="cal"):
                if(predictions[j][i]<0):
                    temp[l[j][0],0]+=1
                    temp[l[j][0],1]+=abs(predictions[j][i])
                else:
                    temp[l[j][1],0]+=1
                    temp[l[j][1],1]+=abs(predictions[j][i])
            elif(calculations=="SVC"):
                if(predictions[j][i][0]>predictions[j][i][1]):
                    temp[l[j][0],0]+=1
                    temp[l[j][0],1]+=abs(predictions[j][i][0])
                else:
                    temp[l[j][1],0]+=1
                    temp[l[j][1],1]+=abs(predictions[j][i][1])
        m=max(temp[:,0])
        index=0
        score=-1
        for i in range(10):
            if(temp[i][0]==m and temp[i][1]>score):
                index=i
                score=temp[i][1]
        pred.append(index)
    return pred

def createSVMs(data,classes):
    c1 = classes[0]
    c2 = classes[1]
    filteredX,filteredY,pos,neg = stripData(data,c1,c2)
    filteredY.reshape((1,-1))
    clf = SVC(C=1.0,kernel='rbf',gamma=0.05,probability=True)
    clf.fit(filteredX,filteredY.ravel())
    print("Done")
    return clf

def allPredictions(clfList,data):
    pred = []
    for i in tqdm(clfList):
        x = i.predict_proba(data)
        pred.append(x)
    return pred

def confusionMatrix(test,prediction):
    mat = np.zeros((10,10))
    index=0
    for i in range(len(prediction)):
        mat[int(test[i]),int(prediction[i])] += 1
    return mat

def draw(confusion,title="Confusion Matrix"):
    df_cm = pd.DataFrame(confusion.astype(int), index = [i for i in "0123456789"],
                      columns = [i for i in "0123456789"])
    plt.figure(figsize = (10,8))
    sns.set(font_scale=1.2)
    sns.heatmap(df_cm,cmap="Blues", annot=True,fmt="d",linewidth=1,annot_kws={"size": 14})
    plt.title(title)
    plt.show()

def findAccuracyParallel(data,c):
    clf = SVC(C=c,kernel='rbf',gamma=0.05)
    clf.fit(data[0],data[1])
    print("Done")
    v = clf.score(data[2],data[3])
    t = clf.score(data[4],data[5])
    return (v,t,c)
    
def verifyKfold(trainX,trainY,testX,testY):
    C = [1e-5,1e-3,1,5,10]
    accuracies = []
    skf = StratifiedKFold(n_splits=5)
    C = [1e-5,1e-3,1,5,10]
    loopTime = time.time()
    accuracies=[]
    for trainId,testId in skf.split(trainX,trainY):
        X_train, X_val = trainX[trainId], trainX[testId]
        Y_train, Y_val = trainY[trainId], trainY[testId]
        st = time.time()
        data = [X_train,Y_train,X_val,Y_val,testX,testY]        
        temp = Parallel(n_jobs=-2)(delayed(findAccuracyParallel)(data,i) for i in C)
        print(time.time()-st)
        print(temp)
        accuracies.append(temp)
    print(time.time()-loopTime)
    return accuracies

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
    alpha = stripAlpha(alpha,1e-5)

    index = alpha != 0
    print("Number of support Vectors",index.sum())

    w,b = calwb(alpha,trainD,Y,pos,neg)
    print("w,b in linear kernel",w.shape,b)
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
    alpha = stripAlpha(alpha,1e-5)
    
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
    print("w,b on linear SVM",clfLinear.intercept_)
    print("SVM gaussian kernel support vectors",clfGaussian.support_vectors_.shape)
    print("b on gaussian SVM",clfGaussian.intercept_)

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

    globalTrain,globalVal,globalTest = init()
    l = list(itertools.combinations([0,1,2,3,4,5,6,7,8,9], 2))

    try:
        pickle_in = open("oneVone.pickle","rb")
        paraList = pickle.load(pickle_in)
    except:
        st = time.time()
        temp = Parallel(n_jobs=-2)(delayed(createClf)(globalTrain,i) for i in l)
        print(time.time()-st)
        with open('oneVone.pickle', 'wb') as f:
            pickle.dump(temp, f) 
        
    testY = globalTest[784]
    testX = np.array(globalTest.drop(columns=[784]))/255

    predictions = classify(paraList,globalTrain,testX)
    oneVonePred = findMajorityClass(predictions,l)
    print("Accuracy on One vs One Test")
    checkAccuracy(testY,oneVonePred)

    valY = globalVal[784]
    valX = np.array(globalVal.drop(columns=[784]))/255

    predictionsVal = classify(paraList,globalTrain,valX)
    oneVonePredVal = findMajorityClass(predictionsVal,l)
    print("Accuracy on One vs One Validation")
    checkAccuracy(valY,oneVonePredVal)

    # PART 2(B)

    try:
        pickle_in = open("oneVoneSVM.pickle","rb")
        clfList = pickle.load(pickle_in)
    except:
        st = time.time()
        temp = Parallel(n_jobs=-2)(delayed(createSVMs)(globalTrain,i) for i in tqdm(l))
        print(time.time()-st)
        with open('oneVoneSVM.pickle', 'wb') as f:
            pickle.dump(temp, f)
    
    testY = globalTest[784]
    testX = np.array(globalTest.drop(columns=[784]))/255
    predictionsTest = allPredictions(clfList,testX)
    oneVonePredSVC = findMajorityClass(predictionsTest,l,calculations="SVC")
    print("Accuracy on One vs One SVM Test")
    checkAccuracy(testY,oneVonePredSVC)

    valY = globalVal[784]
    valX = np.array(globalVal.drop(columns=[784]))/255
    predictionsVal = allPredictions(clfList,valX)
    oneVonePredValSVC = findMajorityClass(predictionsVal,l,calculations="SVC")
    print("Accuracy on One vs One SVM Validation Set")
    checkAccuracy(valY,oneVonePredValSVC)

    # PART 2(C)

    confusion = confusionMatrix(testY,oneVonePred)
    draw(confusion,title="Confusion Matrix for Test Set")
    confusion = confusionMatrix(valY,oneVonePredVal)
    draw(confusion,title="Confusion Matrix for Validation Set")
    confusion = confusionMatrix(testY,oneVonePredSVC)
    draw(confusion,title="Confusion Matrix for Test Set")
    confusion = confusionMatrix(valY,oneVonePredValSVC)
    draw(confusion,title="Confusion Matrix for Validation Set")

    # PART 2(D)

    trainY = globalTrain[784]
    trainX = np.array(globalTrain.drop(columns=[784]))/255

    testY = globalTest[784]
    testX = np.array(globalTest.drop(columns=[784]))/255

    try:
        pickle_in = open("StratkFold.pickle","rb")
        accList = pickle.load(pickle_in)
    except:    
        Kfold = verifyKfold(trainX,trainY,testX,testY)
        with open('StratkFold.pickle', 'wb') as f:
            pickle.dump(Kfold, f)
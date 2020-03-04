import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
import math
import random
from nltk.tokenize import TweetTokenizer,RegexpTokenizer,word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from string import punctuation
import re
import seaborn as sns
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import plot_confusion_matrix
import pickle
import contractions

def init():
    train = pd.read_csv("data/train.csv",encoding="iso-8859-1",header=None)
    test = pd.read_csv("data/test.csv",encoding="iso-8859-1",header=None)
    return train,test

# Pass data,column of text
def getVocab(train,col):
    vocab = {}
    count=0
    for i in range(train[col].shape[0]):
        x = []
        words = train[col][i].split(" ")
        # words = train[col][i].replace(';',' ').replace("."," ").replace(","," ").split(" ")
        # if " " in words:
        #     words.remove(" ")
        for j in words:
            if(j not in vocab):
                vocab[j]=count
                count = count+1
    return vocab

# 0 = Negative 4 = Positive 2 = Neutral

# Pass the column of full data,column's number and vocabulary
def findParamVocab(data,vocab,col):
    v = vocab.fromkeys(vocab,1)
    total = len(vocab)
    for i in data[col]:
        # words = i.replace(';',' ').replace("."," ").replace(","," ").split(" ")
        words = i.split(" ")
        if " " in words:
            words.remove(" ")
        total = total+len(words)
        for j in words:
            v[j] = v[j]+1
    for key in v:
        v[key] = float(v[key]/total)
    return v,total
    
def learnParam(train,vocab,col):
    m=train.shape[0]
    phi0 = (train[train[0]==0].shape[0]+1)/(m+2)
    phi4 = (train[train[0]==4].shape[0]+1)/(m+2)
    param0,t0 = findParamVocab(train[train[0]==0],vocab,col)
    param4,t4 = findParamVocab(train[train[0]==4],vocab,col)
    return phi0,phi4,param0,param4,t0,t4

# Pass the column of text
# def findClasses(data,phi0,phi1,phi2,theta0,theta2,theta4,t0,t2,t4):
def findClasses(data,phi0,phi4,theta0,theta4,t0,t4):
    pred = []
    pred0 = []
    pred4 = []
    for i in data:
        prob0 = 0
        prob2 = 0
        prob4 = 0
        # words = i.replace(';',' ').replace("."," ").replace(","," ").split(" ")
        words = i.split(" ")
        if " " in words:
            words.remove(" ")
        for j  in words:
            try:
                prob0 = prob0+(math.log(theta0[j]))
            except:
                prob0 = prob0+float(1/(t0+1))
        
            try:
                prob4 = prob4+(math.log(theta4[j]))
            except:
                prob4 = prob4+float(1/(t4+1))
                
        prob0 = prob0+(math.log(phi0))
        prob4 = prob4+(math.log(phi4))
        if(prob0>prob4):
            pred.append(0)
        else:
            pred.append(4)
        pred0.append(prob0)
        pred4.append(prob4)
    return pred,pred0,pred4

def randomGuess(data):
    choices=[0,4]
    pred=[]
    for i in data:
        pred.append(random.choice(choices))
    return pred

def majorityGuess(train,test):
    c0 = train[train[0]==0][5].unique().shape[0]
    c4 = train[train[0]==4][5].unique().shape[0]
    m=0
    if(c0>c4):
        m=0
    else:
        m=4
        
    pred=[]
    for i in test[5]:
        pred.append(m)
    return pred

def confusionMatrix(test,prediction):
    mat = np.zeros((2,2))
    index=0
    for i in test[0]:
        mat[int(i/4)][int(prediction[index]/4)] = mat[int(i/4)][int(prediction[index]/4)]+1
        index=index+1
    return mat

def checkAccuracy(Y,pred):
    check = (Y==pred)
    t=0
    f=0
    for i in check:
        if i:
            t=t+1
        else:
            f=f+1
    print(t/(t+f))

# just pass the whole data
# 81.89 = words = contractions.fix(i).replace(","," ").replace(";"," ").replace("."," ").replace("!"," ").split(" ")
# 82.45 = words = (i.replace(","," ").replace(";"," ").replace("."," ").replace("!"," ")).split(" ")
# 83.00 = words = (i.replace(";"," ").replace("."," ").replace("!"," ")).split(" ")
# 83.28 = words = (i.replace("."," ").replace("!"," ")).split(" ")
def cleanText(data,col):
    temp = []
    for i in data[col]:
        words = (i.replace("."," ").replace("!"," ")).split(" ")
        temp.append(" ".join([stemmer.stem(j) for j in words if (j not in stop and '@' not in j)]))
    data[6]=temp

def drawConfusionMatrix(matrix):
    df_cm = pd.DataFrame(matrix.astype(int), index = [i for i in "04"],columns = [i for i in "04"])
    plt.figure(figsize = (7,5))
    sns.heatmap(df_cm,cmap="Blues", annot=True,fmt="d",linewidth=1,annot_kws={"size": 16})
    sns.set(font_scale=1.4)
    plt.show()

def tfidfVectorizer(clf,X,data):
    for i in tqdm(range(0,data.shape[0]//1000)):
        clf.partial_fit(X[1000*i:1000*(i+1)].todense(),data[0][1000*i:1000*(i+1)],classes=np.array([0,4]))
    return clf

def rocCurve(data,prob0,prob4):
    y = label_binarize(data[0],classes=[0,4])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(1-y,prob0)
    roc_auc[0] = auc(fpr[0], tpr[0])
    fpr[1], tpr[1], _ = roc_curve(y,prob4)
    roc_auc[1] = auc(fpr[1], tpr[1])

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',lw=lw, label='ROC curve for Class 0(area = %0.2f)' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], color='cyan',lw=lw, label='ROC curve for Class 1(area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

if __name__=="__main__":
    train,test = init()

    #  PART A

    st = time()
    vocab = getVocab(train,5)
    print("Learning Vocabulary ",time()-st)

    st = time()
    phi0,phi4,theta0,theta4,t0,t4 = learnParam(train,vocab,5)
    print("Learning Param ",time()-st)

    st = time()
    predictionTrain,pred0,pred4 = findClasses(train[5],phi0,phi4,theta0,theta4,t0,t4)
    print("Prediction on Training ",time()-st)
    checkAccuracy(predictionTrain,train[0])

    st = time()
    test = test[test[0]==0].append(test[test[0]==4])
    predictionTest,predTest0,predTest4 = findClasses(test[5],phi0,phi4,theta0,theta4,t0,t4)
    print("Prediction on Testing ",time()-st)
    checkAccuracy(predictionTest,test[0])

    # PART B

    st = time()
    predictionRandom = randomGuess(test[5])
    print("Random Prediction ",time()-st)
    checkAccuracy(predictionRandom,test[0])

    st = time()
    predictionMajority = majorityGuess(train,test)
    print("Majority Prediction ",time()-st)
    checkAccuracy(predictionMajority,test[0])

    # PART C

    matrix = confusionMatrix(test,predictionTest)
    drawConfusionMatrix(matrix)

    # PART D

    stemmer = PorterStemmer()
    stop = set(stopwords.words('english'))
    st=time()
    cleanText(train,5)
    print("Data Cleaned ",time()-st)

    st = time()
    vocabClean = getVocab(train,6)
    print("Cleaned Vocabulary ",time()-st)

    st = time()
    Cphi0,Cphi4,Ctheta0,Ctheta4,Ct0,Ct4 = learnParam(train,vocabClean,6)
    print("Cleaned Params ",time()-st)

    st = time()
    cleanText(test,5)
    predictionTestClean,CpredTest0,CpredTest4 = findClasses(test[6],Cphi0,Cphi4,Ctheta0,Ctheta4,Ct0,Ct4)
    print("Predicted on Clean ",time()-st)

    checkAccuracy(predictionTestClean,test[0])

    matrix = confusionMatrix(test,predictionTestClean)
    drawConfusionMatrix(matrix)

    
    # PART E

    # PART F

    tfidf = TfidfVectorizer()
    clf = GaussianNB()
    multi = MultinomialNB()

    X = tfidf.fit_transform(train[6])

    try:
        gaussianModel = pickle.load(open("tfidfClf.sav", 'rb'))
    except:
        clf = tfidfVectorizer(clf,X,train) 
        filename = 'tfidfClf.sav'
        pickle.dump(clf, open(filename, 'wb'))
    
    multiModel = MultinomialNB()
    multiModel.fit(X,train[0])

    Y = tfidf.transform(test[6])
    testDense = Y.todense()

    predictionTestGaussian = gaussianModel.predict(testDense)
    predictionTestMulti = multiModel.predict(testDense)

    checkAccuracy(predictionTestGaussian,test[0])
    checkAccuracy(predictionTestMulti,test[0])


    Y = train[0]
    percentile = SelectPercentile(chi2, percentile=10)
    selectedData = percentile.fit_transform(X,Y)

    clf2 = GaussianNB()
    try:
        gaussianModel = pickle.load(open("tfidf10Clf.sav", 'rb'))
    except:
        per10 = tfidfVectorizer(clf2,selectedData,train)
        filename = 'tfidf10Clf.sav'
        pickle.dump(clf, open(filename, 'wb'))

    testDense10per = percentile.transform(testDense)
    prediction10per = clf2.predict(testDense10per)

    checkAccuracy(prediction10per,test[0])

    # PART G

    # probGaussian = loaded_model.predict_log_proba(testDense)
    # probMulti = demo.predict_log_proba(Y)
    # rocCurve(test,probGaussian[:,0],probGaussian[:,1])
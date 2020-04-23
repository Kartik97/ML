import numpy as np
import pandas as pd
from math import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import axes3d, Axes3D
import time

def init(t0=0,t1=0,t2=0):   # Column Vector of parameters being initialised
    return np.array([[t0],[t1],[t2]])

def synData(p,s):  # p=parameters s=size of set to be generated
    X1 = np.reshape(np.random.normal(3,2,size=s),(s,-1))     # 3=mean var=4
    X2 = np.reshape(np.random.normal(-1,2,size=s),(s,-1))    # -1=mean var=4
    X2 = np.append(X1,X2,axis=1)
    X = np.append(np.reshape(np.ones((s,1)),(s,-1)),X2,axis=1)
    noise = np.reshape(np.random.normal(0,sqrt(2),s),(s,-1))
    Y=np.dot(X,p)+noise
    return X,Y,noise

def error(data,out,p):           # Cost Function
    return np.sum((out-np.dot(data,p))**2)/(2*data.shape[0])

def stochasticDescent(X,Y,theta,alpha,steps,epsilon=1e-4,batch_size=1,check=1000):
    temp = np.append(X,Y,axis=1)  # Initial shuffling
    np.random.shuffle(temp)
    X = temp[:,:3]
    Y = temp[:,3:4]
    converged = False
    i=0
    m=X.shape[0]
    num_batches=int(m/batch_size)
    b=0
    prevCost = -1
    curCost = 0
    theta0 = []
    theta1 = []
    theta2=[]
    st = time.time()
    while (not converged and i<steps):
        curX = X[b*batch_size:(b+1)*batch_size,:]
        curY = Y[b*batch_size:(b+1)*batch_size,:]
        b=int((b+1)%num_batches)
        
        hyp = np.dot(curX,theta)
        grad = np.dot((curY-hyp).T,-curX).T/curX.shape[0] # Calculating gradient
        thetaNext = theta-alpha*grad
        curCost = curCost+error(curX,curY,thetaNext)

        if(i%100 == 0):        # Recording theta values
            theta0.append(thetaNext[0][0])
            theta1.append(thetaNext[1][0])
            theta2.append(thetaNext[2][0])

        if(i%check==0):                  # Comparing the cost after some set of iterations given by check
            curCost = curCost/check
            if(prevCost!=-1 and abs(prevCost-curCost)<=epsilon):
                converged = True
            prevCost = curCost
            curCost = 0
        
        if(i>steps):
            theta=thetaNext
            converged = True
        else:
            theta = thetaNext

        i=i+1
    print(i)
    return theta,theta0,theta1,theta2

def animateTheta(frame,line,t0,t1,t2):
    line.set_xdata(t0[:frame])
    line.set_ydata(t1[:frame])
    line.set_3d_properties(t2[:frame])
    return line

if __name__ == '__main__':
    initialHyp = init(3,1,2)
    testData = np.array(pd.read_csv("../ass1_data/data/q2/q2test.csv"))
    testX = np.append(np.ones((testData.shape[0],1)),testData[:,:2],axis=1)
    testY = testData[:,2:3]

    # PART A----Synthesizing Data

    X,Y,Noise=synData(init(3,1,2),1000000)

    print(X.shape)
    print(Y.shape)

    # PART B----Learning Parameter Values on the data synthesized

    alpha = 0.001
    steps = 1000000

    st=time.time()
    param,t0,t1,t2 =stochasticDescent(X,Y,init(0,0,0),alpha,steps,epsilon=1e-6,batch_size=10000,check=5000)
    print(time.time()-st)
    print(param)

    # PART C----Comparing the error between the learned and original hypothesis

    errorGen = error(testX,testY,param)
    errorOrig = error(testX,testY,initialHyp)
    print(errorGen)
    print(errorOrig)

    # PART D----Plot showing the movement of theta values untill convergence

    fig = plt.figure(0)
    ax = Axes3D(fig)
    line, = ax.plot(t0,t1,t2,color='red',lw='1')
    ax.set_xlabel("Theta0")
    ax.set_ylabel("Theta1")
    ax.set_zlabel("Theta2")
    ax.set_title("SGD Convergence")
    anim = FuncAnimation(fig, animateTheta,frames=len(t0),fargs=(line,t0,t1,t2),interval=1,repeat=True,blit=False)
    plt.show()

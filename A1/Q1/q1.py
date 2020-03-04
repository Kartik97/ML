import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import axes3d, Axes3D
import time
from math import *

# Function for normalizing the data
def normalize(X):
    m = X.shape[0]
    mean = np.sum(X,axis=0)/m
    var = np.sum((X-mean)**2,axis=0)/m
    sd = np.sqrt(var)
    X = (X-mean)/sd
    return X

# Initializing data and parameters
def init():
    X = np.array(pd.read_csv("../ass1_data/data/q1/linearX.csv",header=None))
    X = normalize(X)
    x = np.append(np.ones(X.shape),X,axis=1)
    Y = np.array(pd.read_csv("../ass1_data/data/q1/linearY.csv",header=None))
    params = np.zeros((2,1))
    return x,X,Y,params

# Cost Function
def cost(data,out,p):
    return np.sum((out-np.dot(data,p))**2)/(2*data.shape[0])

# Function for gradient Descent
def gradientDescent(X,Y,p,steps,epsilon=1e-12,alpha=1e-4):
    
    st = time.time()
    theta0 = []
    theta1 = []
    J = []

    conv = False
    i=0

    while(not conv and i<steps):
        hyp = np.dot(X,p)
        grad = np.dot((Y-hyp).T,(-X)).T/(X.shape[0])
        pNext = p-alpha*grad

        if((time.time()-st)*1000 >= 0.2 and abs(pNext-p).any()>1e-4):
            theta0.append(pNext[0][0])
            theta1.append(pNext[1][0])
            J.append(cost(X,Y,pNext))
            st=time.time()

        if(pNext[0][0] > 1e10 or pNext[1][0] > 1e10): #Checking divergence of parameters
            break

        if(isnan(pNext[0][0]) or isnan(pNext[1][0]) or isinf(pNext[0][0]) or isinf(pNext[1][0])):
            break
            conv = True
            break

        if( abs(pNext-p).all() < epsilon  or i>steps):
            pNext=p
            conv = True
        else:
            p = pNext    

        i=i+1

    return p,theta0,theta1,J

# Function for animations
def animateWireFrame(frame,line,t0,t1,J):
    line.set_xdata(t0[:frame])
    line.set_ydata(t1[:frame])
    line.set_3d_properties(J[:frame])
    return line

def animateContour(frame,line,t0,t1):
    line.set_xdata(t0[:frame])
    line.set_ydata(t1[:frame])
    return line

if __name__ == '__main__':

    # PART A----Learning the parameters

    X,X_,Y,params = init()
    epsilon = 1e-12
    steps = 100000
    alpha = 0.025
    diverged = 0
    updatedParam,t0,t1,J = gradientDescent(X,Y,params,steps,epsilon,alpha)
    print(updatedParam)
    yPred = np.dot(X,updatedParam)

    # Part (B)----Plotting the hypothesis learned

    plt.figure(0)
    ax = plt.gca()
    plt.plot(np.delete(X,0,axis=1),Y,'o',color='blue')
    plt.plot(np.delete(X,0,axis=1),yPred,color='green')
    plt.xlabel("X (Data)")
    plt.ylabel("Y (Output)")
    plt.title("Hypothesis Function Learned")
    plt.show()

    # Part (C)----Plotting the mesh and movement of theta

    if(abs(updatedParam[0][0])>1e10 or abs(updatedParam[1][0])>1e10): #Checking the divergence of parameters
        print("Parameters have diverged")
        diverged = 1

    plt.figure(1)
    k = 1 
    theta_0 = np.linspace(updatedParam[0][0]-k, updatedParam[0][0]+k, 100)
    theta_1 = np.linspace(updatedParam[1][0]-k, updatedParam[1][0]+k, 100)
    Theta_0, Theta_1 = np.meshgrid(theta_0, theta_1)

    J_mat = np.zeros(Theta_0.shape)
    for i in range(Theta_0.shape[0]):
        for j in range(Theta_1.shape[1]):
            temp_theta = np.array([[Theta_0[i][j]],[Theta_1[i][j]]])
            J_mat[i][j] = np.sum((Y-np.dot(X,temp_theta))**2)/(2*X.shape[0])

    fig1 = plt.figure(1)
    ax1 = Axes3D(fig1)
    ax1.plot_surface(Theta_0, Theta_1, J_mat,alpha=0.3, label='Cost Function')
    ax1.set_xlabel("Theta0")
    ax1.set_ylabel("Theta1")
    ax1.set_zlabel("Cost")
    ax1.set_title("Gradient Descent Convergence")
    line, = ax1.plot([],[],[],label='Gradient Descent Movement',color='red',lw='1')    

    
    anim = FuncAnimation(fig1, animateWireFrame,frames=len(J),fargs=(line,t0,t1,J),interval=200,repeat=True,blit=False)
    plt.show()

    # PART D----Contour plot
    
    fig2 = plt.figure(2)
    ax2 = plt.gca()

    cp  = ax2.contour(Theta_0,Theta_1,J_mat,10)
    line, = ax2.plot([],[],label='Gradient Descent Movement',color='red',lw='1') 
    ax2.clabel(cp, inline=True, fontsize=10)
    ax2.set_title('Contour Plot')
    ax2.set_xlabel('Theta0')
    ax2.set_ylabel('Theta1')

    anim = FuncAnimation(fig2, animateContour,frames=len(J),fargs=(line,t0,t1), interval=200,save_count=len(J),repeat=True,blit=False)
    plt.show()
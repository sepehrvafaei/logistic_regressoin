import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#using gradient descent

def logistic_regression(x,y):
    n=len(y)
    iters=100
    alpha=0.05
    x=np.c_[np.ones(n),x]
    m=len(x[0])
    precision=0.001
    theta=np.zeros(m)
    theta_current=theta
    cost_prevoius=0
    cost_current=np.dot(np.dot(x,theta)-y,np.dot(x,theta)-y)/(2*n)
    sigmoid=lambda a:1/(1+math.exp(-a))
    for i in range(iters):
        if(abs(cost_current-cost_previous)<precision):break
        theta_current -= alpha*(1/n)*np.dot(np.array(list(map(sigmoid,np.dot(x,theta))))-y,x)
        theta=theta_current
        cost_previous=cost_current
        cost_current=np.dot(np.dot(x,theta)-y,np.dot(x,theta)-y)/(2*n)
    print(theta)

rng=np.random.default_rng()
training_input=rng.integers(-5,5,size=(500,2))
training_output=rng.integers(0,2,500)
#logistic_regression(training_input,training_output)


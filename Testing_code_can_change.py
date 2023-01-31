import numpy as np
import matplotlib.pyplot as plt

def logistic_function(x, x0=0, k=0.001):
    return -200 * (1 / (1 + np.exp(-k * (x - x0))))+200
def reverse_sigmoid(x, x0=0, k=0.005):
    return 2/(1+np.exp(-x*k))
def tanh(x,k=1):
    upper = np.exp(x*k)-np.exp(-x*k)
    lower = np.exp(x*k)+np.exp(-x*k)
    return upper/lower

K=10
x=np.linspace(0,K,101)

y=(1-tanh(x,k=1/(0.3*K)))*100

fig=plt.figure()
ax0=fig.add_subplot(1,1,1)
ax0.plot(x,y)
ax0.grid()
plt.show()
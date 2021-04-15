import numpy as np
import math
import matplotlib.pyplot as plt

# [0] Expression like these are very common: f(E)=Sum_i W_i delta(E-E_i)
#     this function will evaluate f(E) in a range of values of E between
#     Emin and Emax with interval dE taking as input a vector of values
#     of W_i and E_i. Sigma is the Gaussian broadening

def broaden(W,E,sigma,Emin,Emax,dE):
    N=len(E)
    NP=int((Emax-Emin)/dE)         #number of points to evaluate 
    f=np.empty(NP)
    X=np.empty(NP)
    C=-1/(2*sigma**2)              #quantity precomputed as needed ofter
    for i in range (0,NP):
        x=Emin+i*dE
        X[i]=x
        f[i]=0.
        for j in range (0,N):
            f[i]=f[i]+W[j]*math.exp(C*(x-E[j])**2)
    f=0.39894228*f/sigma    #gaussian normalization 
    return (X,f)


coord = np.matrix([[0, 0, 0], [1, 0, 0], [0, 1.5, 0], [1, 2, 0]])  #coordinates 
w = np.array([0.2, 0.2, 0.3, 0.3])         #weights

R= np.array([])              # initialize energy of vibrational states 
W= np.array([])
N=len(w)

for i in range(N-1):
    for j in range (i+1,N):  # loop in i<j
        R=np.append(R,np.linalg.norm(coord[i]-coord[j]))
        W=np.append(W,w[i]*w[j])

(x,f)=broaden(W,R,0.1,0.8,3.0,0.03)
plt.plot(x,f)       
plt.show()
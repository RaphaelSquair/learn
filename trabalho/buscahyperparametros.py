import copy

import array
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy.core._multiarray_umath
import numpy.ma
import numpy.matlib
import numpy.random.mtrand
import pandas as pd
import scipy.ndimage
import data_opening as op
import numpy as np


def Jcusto(x,tensoes,correntes):
    [ind01out,controlout,error]=op.func(x,tensoes,correntes)

    return [ind01out,controlout,error]

t0 =10
M=1000
N = 3
cont=0
Tmin=0.001
epsilon=0.3
Xk = np.ones(N)*5
Xmin = np.ones(N)*5
Xnext = np.ones(N)*5

tensoes =pd.read_csv("1khz/tensoes.csv",sep=",")
correntes = pd.read_csv("1khz/correntes.csv",sep=",")


#Jmin = Jcusto(Xk)
#Jnext = numpy.ma.copy(Jmin)
#Jatual= numpy.ma.copy(Jnext)
T=t0
k=0
Jhist=[]
JminHist=[]
Thist =[]
#Bestgraf = []
inf_limit = 5
sup_limit = 20

while(T>Tmin):
    for i in range(M):

        if (i == 0):
            [ind01out, controlout, Jatual] = Jcusto(Xk,tensoes,correntes)
            Jmin = Jatual
            Jnext = Jmin
            controlgraf = controlout
        else:
            R = np.random.rand(N) * 20 -10
            Xnext = np.clip(Xk + epsilon * R, inf_limit, sup_limit)
            Jatual=Jnext
            [ind01out, controlout, Jnext] = Jcusto(Xnext, tensoes, correntes)


        deltaJ = Jnext-Jatual
        r = numpy.random.mtrand.rand(1)
        if(r< numpy.core._multiarray_umath.exp(-deltaJ/T)):
            Xk=Xnext
        if(Jatual<Jmin):
            Jmin=Jatual
            Xmin=Xk
            Bestgraf=ind01out
        cont = cont+1
        Jhist.append(Jatual)
        JminHist.append(Jmin)
        Thist.append(T)
    k=k+1
    T=t0*(0.9)**k
    print(T)

print(cont)
t=range(cont)
t= numpy.ma.array(t)
Jhist=numpy.ma.array(Jhist)
JminHist=numpy.ma.array(JminHist)
f,(ax1,ax2,ax3) = matplotlib.pyplot.subplots(3, 1, sharex=False)
#ax=gca()
ax1.plot(t,Jhist,color='r')
ax1.set_title('posição de X')
ax2.set_ylim((-0.1, 2))
ax2.plot(t,Jhist,color='b')
ax2.plot(t,JminHist,color='r')
ax2.set_title('Distibuição J(x)')
ax2.set_ylim((-0.2, 2))
ax3.plot(t,Thist)
ax3.set_title('Temperatura(T)')
ax3.set_ylim((0, 10))

matplotlib.pyplot.show()



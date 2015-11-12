#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from fenton1985 import *

kd = np.logspace(np.log10(0.5),1)
N = len(kd)

fig, ax = plt.subplots(nrows=2, sharex=True)

C0 = np.zeros((N))
C2 = np.zeros((N))
C4 = np.zeros((N))
for i in range(N):
    C0[i],C2[i],C4[i] = evalC(kd[i])
ax[0].semilogx(kd,C0,label='C0')
ax[0].semilogx(kd,C2,label='C2')
ax[0].semilogx(kd,C4,label='C4')
ax[0].legend(loc='best')

A11 = np.zeros((N))
A22 = np.zeros((N))
A33 = np.zeros((N))
A44 = np.zeros((N))
A55 = np.zeros((N))
A31 = np.zeros((N))
A42 = np.zeros((N))
A51 = np.zeros((N))
A53 = np.zeros((N))
for i in range(N):
    A11[i],A22[i],A33[i],A44[i],A55[i],A31[i],A42[i],A51[i],A53[i] = evalA(kd[i])
ax[1].semilogx(kd,A11,label='A11')
ax[1].semilogx(kd,A22,label='A22')
ax[1].semilogx(kd,A33,label='A33')
ax[1].semilogx(kd,A44,label='A44')
ax[1].semilogx(kd,A55,label='A55')
ax[1].semilogx(kd,A31,label='A31')
ax[1].semilogx(kd,A42,label='A42')
ax[1].semilogx(kd,A51,label='A51')
ax[1].semilogx(kd,A53,label='A53')
ax[1].legend(loc='best')


plt.show()

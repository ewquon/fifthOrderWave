#!/usr/bin/python
import sys
from fenton1985 import *
import numpy as np
from scipy.optimize import fsolve

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

makeplots = True

TOL = 1e-8

g = 9.81 # gravity [m/s]
d = 4.0 # depth [m]
twopi = 2*np.pi

#T,H = [5.66],[1.2]
#T = np.arange(1.4,6.0,0.2)
#H = np.arange(0.05,2.0,0.1)
T = np.arange(1.4,10.0,0.2)
H = np.arange(0.05,3.0,0.1)
NT = len(T)
NH = len(H)
coskx = np.zeros((NH,NT))
kx = np.zeros((NH,NT))
maxslope = np.zeros((NH,NT))
lam = np.zeros((NH,NT))
umean = np.zeros((NH,NT))
dzdt = np.zeros((NH,NT))
warnings = 0

for j in range(NT):
    for i in range(NH):

        k = calculateWavenumber(g,T[j],H[i],d)
        lam[i,j] = twopi/k
        e = k*H[i]/2

        B22,B31,B42,B44,B53,B55 = evalB(k*d)

        A = e + B31*e**3 - (B53+B55)*e**5
        B = B22*e**2 + B42*e**4
        C = -B31*e**3 + B53*e**5
        D = B44*e**4
        E = B55*e**5

        C0 = 16*D - 4*B
        C1 = A - 27*C + 125*E
        C2 = 8*B - 128*D
        C3 = 36*C - 500*E
        C4 = 128*D
        C5 = 400*E

        def coskx_eqn(x): 
            eqn = C0 + C1*x + C2*x**2 + C3*x**3 + C4*x**4 + C5*x**5 # = (curvature) / (-k)
            return eqn
        coskx[i,j] = fsolve(coskx_eqn,1)
        if coskx[i,j] > 1 or coskx[i,j] < -1: warnings += 1

        t = np.arccos(coskx[i,j])
        kx[i,j] = t
        maxslope[i,j] = np.abs(A*np.sin(t) + 2*B*np.sin(2*t) + 3*C*np.sin(3*t) +  4*D*np.sin(4*t) +  5*E*np.sin(5*t))
        umean[i,j] = lam[i,j]/T[j]
        dzdt[i,j] = maxslope[i,j] * umean[i,j]

Ncases = NH * NT
print Ncases,'cases processed'

print '  cos(kx) min/max :',np.min(coskx),np.max(coskx),'[-1,1]'
print '      kx  min/max :',np.min(kx),np.max(kx),'[0,2pi]'
print ' x/lambda min/max :',np.min(kx/twopi),np.max(kx/twopi),'[0,0.25]'
print '   dz/dx  min/max :',np.min(maxslope),np.max(maxslope)
print '   umean  min/max :',np.min(umean),np.max(umean)
print '   dz/dt  min/max :',np.min(dzdt),np.max(dzdt)

if warnings: print 'Valid roots could not be found for',warnings,'cases'
assert( warnings==0 )

if makeplots:
    XX,YY = np.meshgrid(H,T)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(XX,YY,lam.T,rstride=1,cstride=1, cmap=cm.coolwarm)
    ax.set_xlabel('wave height [m]')
    ax.set_ylabel('wave period [s]')
    ax.set_zlabel('wave length [m]')
    ax.set_title('Nonlinear wave length')
    ax.plot([.08,.3,.76,1.2,1.6,1.2],[1.86,1.41,2.32,3.1,4.38,5.66],\
            [5.41,3.36,8.97,15.06,25.01,33.57],\
            'k*',markeredgewidth=3)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(XX,YY,umean.T,rstride=1,cstride=1, cmap=cm.coolwarm)
    ax.set_xlabel('wave height [m]')
    ax.set_ylabel('wave period [s]')
    ax.set_zlabel('wave speed [m/s]')
    ax.set_title('Mean wave speed')
    ax.plot([.08,.3,.76,1.2,1.6,1.2],[1.86,1.41,2.32,3.1,4.38,5.66],\
            [2.91,2.38,3.86,4.86,5.71,5.93],\
            'k*',markeredgewidth=3)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(XX,YY,kx.T/twopi,rstride=1,cstride=1, cmap=cm.coolwarm_r)
    ax.set_xlabel('wave height [m]')
    ax.set_ylabel('wave period [s]')
    ax.set_zlabel('x/lambda')
    ax.set_zlim((0,0.25))
    ax.set_title('Location of maximum wave velocity')

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(XX,YY,maxslope.T, rstride=1,cstride=1, cmap=cm.coolwarm)
    ax.set_xlabel('wave height [m]')
    ax.set_ylabel('wave period [s]')
    ax.set_zlabel('|dz/dx|')
    ax.set_title('Maximum wave slope')

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(XX,YY,dzdt.T, rstride=1,cstride=1, cmap=cm.coolwarm)
    ax.set_xlabel('wave height [m]')
    ax.set_ylabel('wave period [s]')
    ax.set_zlabel('|dz/dt| [m/s]')
    ax.set_title('Maximum local vertical velocity')
    ax.plot([.08,.3,.76,1.2,1.6,1.2],[1.86,1.41,2.32,3.1,4.38,5.66],\
            [0.14,0.71,1.09,1.34,1.46,0.82],\
            'k*',markeredgewidth=3)

    plt.show()


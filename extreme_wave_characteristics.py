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

refT = np.array([1.86,1.41,2.32,3.1,4.38,5.66])
refH = np.array([0.08, 0.3,0.76,1.2, 1.6,1.2])
Nref = len(refT)
assert(Nref==len(refH))

#T,H = [5.66],[1.2]
T = np.arange(1.4,6.0,0.2)
H = np.arange(0.05,2.0,0.1)
#T = np.arange(1.4,10.0,0.2)
#H = np.arange(0.05,3.0,0.1)
NT = len(T)
NH = len(H)

lam = np.zeros((NH,NT))
umean = np.zeros((NH,NT))
dzdx_max = np.zeros((NH,NT))
u_max = np.zeros((NH,NT))
u_min = np.zeros((NH,NT))
w_max = np.zeros((NH,NT))
kx_wmax = np.zeros((NH,NT))
warnings = 0
#guess = np.pi/4
for j in range(NT):
    for i in range(NH):

        k = calculateWavenumber(g,T[j],H[i],d)
        lam[i,j] = twopi/k
        umean[i,j] = lam[i,j]/T[j]
        e = k*H[i]/2
        kd = k*d

        A11,A22,A33,A44,A55,A31,A42,A51,A53 = evalA(kd)
        a1 = e*A11 + e**3*A31 + e**5*A51
        a2 = e**2*A22 + e**4*A42
        a3 = e**3*A33 + e**5*A53
        a4 = e**4*A44
        a5 = e**5*A55
        
        B22,B31,B42,B44,B53,B55 = evalB(kd)
        b1 = e + B31*e**3 - (B53+B55)*e**5
        b2 = B22*e**2 + B42*e**4
        b3 = -B31*e**3 + B53*e**5
        b4 = B44*e**4
        b5 = B55*e**5
        kymin = kd - e + e**2*B22 + e**4*(B42+B44)
        kymax = kd + e + e**2*B22 + e**4*(B42+B44)

        C0,C2,C4 = evalC(kd)
        unorm = C0 * np.sqrt(g/k)

        #
        # calculate min/max downwave velocity 
        # assuming it occurs at the wave crest
        #
        u_min[i,j] = ( \
                 -  a1 * np.cosh(  kymin) \
                + 2*a2 * np.cosh(2*kymin) \
                - 3*a3 * np.cosh(3*kymin) \
                + 4*a4 * np.cosh(4*kymin) \
                - 5*a5 * np.cosh(5*kymin) ) * unorm
        u_max[i,j] = ( \
                    a1 * np.cosh(  kymax) \
                + 2*a2 * np.cosh(2*kymax) \
                + 3*a3 * np.cosh(3*kymax) \
                + 4*a4 * np.cosh(4*kymax) \
                + 5*a5 * np.cosh(5*kymax) ) * unorm

        #
        # calculate max normal velocity
        #
        def wavesurf(kx): # returns ky
            return kd \
                + e*np.cos(kx) \
                + e**2*B22*np.cos(2*kx) \
                + e**3*B31*(np.cos(kx) - np.cos(3*kx)) \
                + e**4*(B42*np.cos(2*kx) + B44*np.cos(4*kx)) \
                + e**5*(-(B53+B55)*np.cos(kx) + B53*np.cos(3*kx) + B55*np.cos(5*kx))
        def dwdx(kx): #returns d[w/unorm]/dx
            ky = wavesurf(kx)
            return   a1 * k * np.sinh(  ky) * np.cos(  kx) \
                +  4*a2 * k * np.sinh(2*ky) * np.cos(2*kx) \
                +  9*a3 * k * np.sinh(3*ky) * np.cos(3*kx) \
                + 16*a4 * k * np.sinh(4*ky) * np.cos(4*kx) \
                + 25*a5 * k * np.sinh(5*ky) * np.cos(5*kx)
        #kxmax,info,ierr,mesg = fsolve(dwdx,0,full_output=True)
        kxmax,info,ierr,mesg = fsolve(dwdx,np.pi/4,full_output=True)
        #kxmax,info,ierr,mesg = fsolve(dwdx,guess,full_output=True)
        if not ierr: print 'problem solving dw/dx eqn:',mesg
        if not isinstance(kxmax,float): kxmax = kxmax[0]
        if kxmax > 0 and kxmax < np.pi/2: guess = kxmax
        else: print 'doh'
        kx_wmax[i,j] = kxmax
        kymax = wavesurf(kxmax)
        w_max[i,j] = ( \
                a1 * np.sinh(  kymax)*np.sin(  kxmax) \
            + 2*a2 * np.sinh(2*kymax)*np.sin(2*kxmax) \
            + 3*a3 * np.sinh(3*kymax)*np.sin(3*kxmax) \
            + 4*a4 * np.sinh(4*kymax)*np.sin(4*kxmax) \
            + 5*a5 * np.sinh(5*kymax)*np.sin(5*kxmax) ) * unorm

        #
        # calculate max slope
        #
        c0 = 16*b4 - 4*b2
        c1 = b1 - 27*b3 + 125*b5
        c2 = 8*b2 - 128*b4
        c3 = 36*b3 - 500*b5
        c4 = 128*b4
        c5 = 400*b5
        def coskx_eqn(x): 
            return c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5 # = (curvature) / (-k)
        coskx,info,ierr,mesg = fsolve(coskx_eqn,1,full_output=True)
        if not ierr: print 'problem solving cos(kx) eqn:',mesg
        if coskx > 1 or coskx < -1: warnings += 1
        kx = np.arccos(coskx)
        dzdx_max[i,j] = b1*np.sin(kx) \
                    + 2*b2*np.sin(2*kx) \
                    + 3*b3*np.sin(3*kx) \
                    + 4*b4*np.sin(4*kx) \
                    + 5*b5*np.sin(5*kx)

Ncases = NH * NT
print Ncases,'cases processed'
if warnings > 0: print 'WARNINGS:',warnings

if makeplots:
    XX,YY = np.meshgrid(H,T)
    reflamb = np.zeros((Nref))
    refuavg = np.zeros((Nref))
    refdzdx = np.zeros((Nref))
    refumin = np.zeros((Nref))
    refumax = np.zeros((Nref))
    refwmax = np.zeros((Nref))

    from scipy.interpolate import interp2d
    F_lamb = interp2d(H,T,lam.T)
    F_uavg = interp2d(H,T,umean.T)
    F_dzdx = interp2d(H,T,dzdx_max.T)
    F_umin = interp2d(H,T,u_min.T)
    F_umax = interp2d(H,T,u_max.T)
    F_wmax = interp2d(H,T,w_max.T)
    # need to do this because potential repeated values in refT/refH
    # will throw a ValueError
    for i in range(Nref):
        reflamb[i] = F_lamb(refH[i],refT[i])
        refuavg[i] = F_uavg(refH[i],refT[i])
        refdzdx[i] = F_dzdx(refH[i],refT[i])
        refumin[i] = F_umin(refH[i],refT[i])
        refumax[i] = F_umax(refH[i],refT[i])
        refwmax[i] = F_wmax(refH[i],refT[i])

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(XX,YY,lam.T,rstride=1,cstride=1, cmap=cm.coolwarm)
    ax.set_xlabel('wave height [m]')
    ax.set_ylabel('wave period [s]')
    ax.set_zlabel('wave length [m]')
    ax.set_title('Nonlinear wave length')
    ax.plot(refH,refT,reflamb,'k+-',markeredgewidth=3)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(XX,YY,umean.T,rstride=1,cstride=1, cmap=cm.coolwarm)
    ax.set_xlabel('wave height [m]')
    ax.set_ylabel('wave period [s]')
    ax.set_zlabel('wave speed [m/s]')
    ax.set_title('Mean wave speed')
    ax.plot(refH,refT,refuavg,'k+-',markeredgewidth=3)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(XX,YY,dzdx_max.T, rstride=1,cstride=1, cmap=cm.coolwarm)
    ax.set_xlabel('wave height [m]')
    ax.set_ylabel('wave period [s]')
    ax.set_zlabel('dz/dx')
    ax.set_title('Maximum wave slope')
    ax.plot(refH,refT,refdzdx,'k+-',markeredgewidth=3)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(XX,YY,u_min.T, rstride=1,cstride=1, cmap=cm.coolwarm)
    ax.set_xlabel('wave height [m]')
    ax.set_ylabel('wave period [s]')
    ax.set_zlabel('u [m/s]')
    ax.set_title('Minimum local downwave velocity')
    ax.plot(refH,refT,refumin,'k+-',markeredgewidth=3)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(XX,YY,u_max.T, rstride=1,cstride=1, cmap=cm.coolwarm)
    ax.set_xlabel('wave height [m]')
    ax.set_ylabel('wave period [s]')
    ax.set_zlabel('u [m/s]')
    ax.set_title('Maximum local downwave velocity')
    ax.plot(refH,refT,refumax,'k+-',markeredgewidth=3)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(XX,YY,w_max.T, rstride=1,cstride=1, cmap=cm.coolwarm)
    ax.set_xlabel('wave height [m]')
    ax.set_ylabel('wave period [s]')
    ax.set_zlabel('w [m/s]')
    ax.set_title('Maximum local vertical velocity')
    ax.plot(refH,refT,refwmax,'k+-',markeredgewidth=3)

#    fig = plt.figure()
#    ax = fig.add_subplot(111,projection='3d')
#    ax.plot_surface(XX,YY,kx_wmax.T, rstride=1,cstride=1, cmap=cm.coolwarm)
#    ax.set_zlim((0,np.pi/2))

    plt.show()


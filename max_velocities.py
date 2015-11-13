#!/usr/bin/python
import sys
if len(sys.argv) <= 2: sys.exit('specify wave period and height')

from fenton1985 import *
import numpy as np
from scipy.optimize import fsolve

DEBUG = False
if DEBUG: import matplotlib.pyplot as plt

T = float(sys.argv[1])
H = float(sys.argv[2])
g = 9.81    # gravity [m/s]
d = 4.0     # depth [m]
twopi = 2*np.pi

# calculate k and working variables
k = calculateWavenumber(g,T,H,d)
kd = k*d
e = k*H/2
lam = twopi/k
umean = lam/T
print 'mean x-velocity=',umean

# coefficients to evaluate the potential function and wave surface profile
A11,A22,A33,A44,A55,A31,A42,A51,A53 = evalA(kd)
B22,B31,B42,B44,B53,B55 = evalB(kd)
C0,C2,C4 = evalC(kd)
unorm = C0 * np.sqrt(g/k)

#
# calculate umax assuming it occurs at the wave crest (kxmax=0)
#
kymax = kd + e + e**2*B22 + e**4*(B42+B44)
umax = (e*A11 + e**3*A31 + e**5*A51) * np.cosh(  kymax) \
        + 2*(e**2*A22 + e**4*A42)    * np.cosh(2*kymax) \
        + 3*(e**3*A33 + e**5*A53)    * np.cosh(3*kymax) \
        + 4* e**4*A44                * np.cosh(4*kymax) \
        + 5* e**5*A55                * np.cosh(5*kymax)
umax *= unorm

#
# find the maximum slope
#
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

def coskx_eqn(x):  # x = cos(kx)
    eqn = C0 + C1*x + C2*x**2 + C3*x**3 + C4*x**4 + C5*x**5 # = (curvature) / (-k)
    return eqn
#coskx fsolve(coskx_eqn,0)
coskx = fsolve(coskx_eqn,1)
if not isinstance(coskx,float): coskx = coskx[0]

t = np.arccos(coskx)
kxmax = t
maxslope = np.abs(A*np.sin(t) + 2*B*np.sin(2*t) + 3*C*np.sin(3*t) +  4*D*np.sin(4*t) +  5*E*np.sin(5*t))
print 'max slope=',maxslope
print 'max slope occurs at kx=',kxmax,'in [0,2pi]'
# print '                  at x=',kxmax/k,' expected less than',0.25*lam
# 
# kymax = wavesurf(kxmax)
# print 'max slope occurs at ky=',kymax
# print '                     y=',(kymax-kd)/k
# 
# # z-velocity at the maximum slope
# wmax_est = zvel(kxmax,kymax)
# 
# print 'estimated max z-velocity:',wmax_est,'m/s'
print '------------------------------------------'

def wavesurf(kx): # returns ky
    return kd \
        + e*np.cos(kx) \
        + e**2*B22*np.cos(2*kx) \
        + e**3*B31*(np.cos(kx) - np.cos(3*kx)) \
        + e**4*(B42*np.cos(2*kx) + B44*np.cos(4*kx)) \
        + e**5*(-(B53+B55)*np.cos(kx) + B53*np.cos(3*kx) + B55*np.cos(5*kx))
def dkydx(kx):
    return -k*( \
            e*np.sin(kx) \
            + 2*e**2*B22*np.sin(2*kx) \
            + e**3*B31*(np.sin(kx) - 3*np.sin(3*kx)) \
            + e**4*(2*B42*np.sin(2*kx) + 4*B44*np.sin(4*kx)) \
            + e**5*(-(B53+B55)*np.sin(kx) + 3*B53*np.sin(3*kx) + 5*B55*np.sin(5*kx)) \
            )
def zvel(kx,ky):
    # note the 'k' factor from the derivative is absorved into 'unorm'
    w = (e*A11 + e**3*A31 + e**5*A51) * np.sinh(  ky)*np.sin(  kx) \
        + 2*(e**2*A22 + e**4*A42)     * np.sinh(2*ky)*np.sin(2*kx) \
        + 3*(e**3*A33 + e**5*A53)     * np.sinh(3*ky)*np.sin(3*kx) \
        + 4* e**4*A44                 * np.sinh(4*ky)*np.sin(4*kx) \
        + 5* e**5*A55                 * np.sinh(5*ky)*np.sin(5*kx)
    return unorm * w

# 
# calculate maximum z-velocity without any assumptions
#
def dwdx(kx): #returns d[w/unorm]/dx
    ky = wavesurf(kx)
    dky = dkydx(kx)
    return   (e*A11 + e**3*A31 + e**5*A51) *  k  * np.sinh(  ky) * np.cos(  kx) \
        +  4*(e**2*A22 + e**4*A42)         *  k  * np.sinh(2*ky) * np.cos(2*kx) \
        +  9*(e**3*A33 + e**5*A53)         *  k  * np.sinh(3*ky) * np.cos(3*kx) \
        + 16* e**4*A44                     *  k  * np.sinh(4*ky) * np.cos(4*kx) \
        + 25* e**5*A55                     *  k  * np.sinh(5*ky) * np.cos(5*kx) \
        +    (e*A11 + e**3*A31 + e**5*A51) * dky * np.cosh(  ky) * np.sin(  kx) \
        +  4*(e**2*A22 + e**4*A42)         * dky * np.cosh(2*ky) * np.sin(2*kx) \
        +  9*(e**3*A33 + e**5*A53)         * dky * np.cosh(3*ky) * np.sin(3*kx) \
        + 16* e**4*A44                     * dky * np.cosh(4*ky) * np.sin(4*kx) \
        + 25* e**5*A55                     * dky * np.cosh(5*ky) * np.sin(5*kx)
#kxmax,info,istat,mesg = fsolve(dwdx,np.pi/4,full_output=True)
#print info
kxmax = fsolve(dwdx,np.pi/4)
if not isinstance(kxmax,float): kxmax = kxmax[0]
kymax = wavesurf(kxmax)
wmax = zvel(kxmax,kymax)

print 'vmax occurs at kx=',kxmax,'in [0,2pi]'
print '             at x=',kxmax/k,' expected less than',0.25*lam
# print 'vmax occurs at ky=',kymax
# print '             at y=',(kymax-kd)/k
print 'MAXIMUM X,Z VELOCITIES:',umean+umax,wmax,'m/s'

#------------------------------------------------------------------

if DEBUG:# {{{
    kx = np.linspace(0,4*np.pi,501)
    kzsurf = wavesurf(kx)
    wsurf = zvel(kx,kzsurf)

    fig,ax = plt.subplots(nrows=2, sharex=True)

    ax[0].plot(kx,wsurf)
    ax[0].set_xlabel('kx')
    ax[0].set_ylabel('z-velocity')

    print 'DEBUG: approx max z-vel',np.max(wsurf)

    kxmid = 0.5*(kx[1:] + kx[:-1])
    dx = (kx[1]-kx[0])/k
    dwdx_est = (wsurf[1:] - wsurf[:-1])/dx
    dwdx_calc = dwdx(kx)*unorm
    ax[1].plot(kx,dwdx_calc,label='calculated')
    ax[1].plot(kxmid,dwdx_est,'--',label='finite diff')
    ax[1].plot(kxmid,dwdx_est-0.5*(dwdx_calc[1:]+dwdx_calc[:-1]),'r:',label='error')
    ax[1].set_xlabel('kx')
    ax[1].set_ylabel('change in z-vel')
    ax[1].legend(loc='best')

    # the plotted surface profile and slope here are correct!
    slope = -   e*np.sin(kx) \
            - 2*e**2*B22*np.sin(2*kx) \
            - e**3*B31*(np.sin(kx) - 3*np.sin(3*kx)) \
            - e**4*(2*B42*np.sin(2*kx) + 4*B44*np.sin(4*kx)) \
            - e**5*(-(B53+B55)*np.sin(kx) + 3*B53*np.sin(3*kx) + 5*B55*np.sin(5*kx))
    slope_est = ((kzsurf[1:] - kzsurf[:-1])/k)/dx
    plt.figure()
    plt.plot(kx,kzsurf/k-d)
    plt.plot(kx,slope)
    plt.plot(kxmid,slope_est,'--')
    plt.xlabel('kx')
    plt.title('surface profile and slope')

    print 'DEBUG: approx max dz/dx',np.max(slope_est)

    plt.show()# }}}

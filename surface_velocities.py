#!/usr/bin/python
import sys
if len(sys.argv) <= 2: 
    sys.exit('specify wave period and height')

from fenton1985 import *
import numpy as np
import matplotlib.pyplot as plt

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
umean = lam/T # used to calculate absolute velocity

# coefficients to evaluate the potential function and wave surface profile
A11,A22,A33,A44,A55,A31,A42,A51,A53 = evalA(kd)
B22,B31,B42,B44,B53,B55 = evalB(kd)
C0,C2,C4 = evalC(kd)
unorm = C0 * np.sqrt(g/k)

def wavesurf(kx): # returns ky
    return kd \
        + e*np.cos(kx) \
        + e**2*B22*np.cos(2*kx) \
        + e**3*B31*(np.cos(kx) - np.cos(3*kx)) \
        + e**4*(B42*np.cos(2*kx) + B44*np.cos(4*kx)) \
        + e**5*(-(B53+B55)*np.cos(kx) + B53*np.cos(3*kx) + B55*np.cos(5*kx))
def dydx(kx):
    return -( \
            e*np.sin(kx) \
            + 2*e**2*B22*np.sin(2*kx) \
            + e**3*B31*(np.sin(kx) - 3*np.sin(3*kx)) \
            + e**4*(2*B42*np.sin(2*kx) + 4*B44*np.sin(4*kx)) \
            + e**5*(-(B53+B55)*np.sin(kx) + 3*B53*np.sin(3*kx) + 5*B55*np.sin(5*kx)) \
            )

def xvel(kx,ky):
    # note the 'k' factor from the derivative is absorved into 'unorm'
    dy = dydx(kx)
    u = (e*A11 + e**3*A31 + e**5*A51)          * np.cosh(  ky) * np.cos(  kx) \
        + 2*(e**2*A22 + e**4*A42)              * np.cosh(2*ky) * np.cos(2*kx) \
        + 3*(e**3*A33 + e**5*A53)              * np.cosh(3*ky) * np.cos(3*kx) \
        + 4* e**4*A44                          * np.cosh(4*ky) * np.cos(4*kx) \
        + 5* e**5*A55                          * np.cosh(5*ky) * np.cos(5*kx)
#    uderiv = (e*A11 + e**3*A31 + e**5*A51) * dy * np.sinh(  ky) * np.sin(  kx) \
#         + 2*(e**2*A22 + e**4*A42)         * dy * np.sinh(2*ky) * np.sin(2*kx) \
#         + 3*(e**3*A33 + e**5*A53)         * dy * np.sinh(3*ky) * np.sin(3*kx) \
#         + 4* e**4*A44                     * dy * np.sinh(4*ky) * np.sin(4*kx) \
#         + 5* e**5*A55                     * dy * np.sinh(5*ky) * np.sin(5*kx)
#    return unorm*u, unorm*uderiv
    return unorm*u

def zvel(kx,ky):
    # note the 'k' factor from the derivative is absorved into 'unorm'
    w = (e*A11 + e**3*A31 + e**5*A51) * np.sinh(  ky) * np.sin(  kx) \
        + 2*(e**2*A22 + e**4*A42)     * np.sinh(2*ky) * np.sin(2*kx) \
        + 3*(e**3*A33 + e**5*A53)     * np.sinh(3*ky) * np.sin(3*kx) \
        + 4* e**4*A44                 * np.sinh(4*ky) * np.sin(4*kx) \
        + 5* e**5*A55                 * np.sinh(5*ky) * np.sin(5*kx)
    return unorm * w

kx = np.linspace(0,4*np.pi,501)
kzsurf = wavesurf(kx)
surf = kzsurf/k - d

usurf = xvel(kx,kzsurf)
wsurf = zvel(kx,kzsurf)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(kx/(2*np.pi),surf-np.min(surf),'k--')#,color='0.3')
ax1.set_ylabel('wave height [m]')
#ax1.set_ylim((-0.75*H,1.25*H))
ax1.set_ylim((0,5*H))

ax2.plot(kx/(2*np.pi),usurf,'b',linewidth=2,label='u\'')
ax2.plot(kx/(2*np.pi),wsurf,'g',linewidth=2,label='w\'')
ax2.set_xlabel('x / $\lambda$')
ax2.set_ylabel('velocity [m/s]')
ax2.legend(loc='best')

fig.suptitle('Surface Velocities')

print 'umean =',umean,'m/s'
print 'approx min/max x-vel',np.min(usurf),np.max(usurf)
print 'approx min/max z-vel',np.min(wsurf),np.max(wsurf)

plt.show()

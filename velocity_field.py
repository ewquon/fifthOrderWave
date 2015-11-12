#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from fenton1985 import *

#--------
# INPUTS
T = 5.66
H = 1.2
d = 4 #15
g = 9.81
nL = 80
nH = 20
#--------

k = calculateWavenumber(g,T,H,d)
kd = k*d
lam = 2*np.pi/k
e = k*H/2
print 'wavenumber :',k,'1/m'
print 'wavelength :',lam,'m'
print 'waveheight :',H,'m','(non-dim height=%f)'%(e)

#
# setup contour mesh
#
x = np.arange(-2*lam,2*lam,lam/nL)
#y = np.arange(0,d,H/nH) #y=0 is on seabed
y = np.arange(0,d+0.75*H,H/nH) #y=0 is on seabed
X,Y = np.meshgrid(x,y)

#
# calculate free surface
#
B22,B31,B42,B44,B53,B55 = evalB(kd)
kn = e*np.cos(k*x) \
        + e**2*B22*np.cos(2*k*x) \
        + e**3*B31*(np.cos(k*x) - np.cos(3*k*x)) \
        + e**4*(B42*np.cos(2*k*x) + B44*np.cos(4*k*x)) \
        + e**5*(-(B53+B55)*np.cos(k*x) + B53*np.cos(3*k*x) + B55*np.cos(5*k*x))
ysurf = kn/k

#
# calculate velocity fields
#
C0,C2,C4 = evalC(kd)
umean = np.sqrt(g/k)*( C0 + e**2*C2 + e**4*C4 )

print 'mean horizontal fluid speed :',umean,'m/s'
#print '  calculated from lambda, T :',lam/T,'m/s  ( diff=',umean-lam/T,')'

A11,A22,A33,A44,A55,A31,A42,A51,A53 = evalA(kd)
Udelta = (e*A11 + e**3*A31 + e**5*A51) * np.cosh(  k*Y)*np.cos(  k*X) \
        + 2*(e**2*A22 + e**4*A42)      * np.cosh(2*k*Y)*np.cos(2*k*X) \
        + 3*(e**3*A33 + e**5*A53)      * np.cosh(3*k*Y)*np.cos(3*k*X) \
        + 4* e**4*A44                  * np.cosh(4*k*Y)*np.cos(4*k*X) \
        + 5* e**5*A55                  * np.cosh(5*k*Y)*np.cos(5*k*X)
Vdelta = (e*A11 + e**3*A31 + e**5*A51) * np.sinh(  k*Y)*np.sin(  k*X) \
        + 2*(e**2*A22 + e**4*A42)      * np.sinh(2*k*Y)*np.sin(2*k*X) \
        + 3*(e**3*A33 + e**5*A53)      * np.sinh(3*k*Y)*np.sin(3*k*X) \
        + 4* e**4*A44                  * np.sinh(4*k*Y)*np.sin(4*k*X) \
        + 5* e**5*A55                  * np.sinh(5*k*Y)*np.sin(5*k*X)

#U = -umean + C0*np.sqrt(g/k)*Udelta    # in wave frame, positive u is in -ve x direction
#U = umean - C0*np.sqrt(g/k)*Udelta     # in stationary frame
U = C0*np.sqrt(g/k)*Udelta             # downwave perturbation in stationary frame
V = C0*np.sqrt(g/k)*Vdelta             # downwave perturbation in stationary frame

Umag = np.sqrt( (umean+U)**2 + V**2 )

Ubed = Umag[0,:]
stdev = np.std(Ubed)
print 'velocity magnitude on seabed (min/max/stdev):',\
        np.min(Ubed),np.max(Ubed),stdev,\
        '(%f%% Umean)'%(100*stdev/umean)

#
# clip field above the free surface
#
mask = np.zeros((U.shape[0],U.shape[1]), dtype=bool)
for i in range(len(x)):
    mask[:,i] = Y[:,i]-d > ysurf[i]
U_ = np.ma.array(U, mask=mask)
V_ = np.ma.array(V, mask=mask)
Umag_ = np.ma.array(Umag, mask=mask)

print 'max x,y-velocity:', np.max(U_)+umean, np.max(V_)

#
# plot
#
fig, ax = plt.subplots(nrows=3)
def plot_overlay(ax):
    ax.plot(x,ysurf,'k',linewidth=3)
    ax.plot([x[0],x[-1]],[-0.25*lam,-0.25*lam],'k--')

hc = ax[0].contourf(X,Y-d,U_)
plt.colorbar(hc,ax=ax[0])
plot_overlay(ax[0])
ax[0].set_ylabel('u(x,y)')

hc = ax[1].contourf(X,Y-d,V_)
plt.colorbar(hc,ax=ax[1])
plot_overlay(ax[1])
ax[1].set_ylabel('v(x,y)')

hc = ax[2].contourf(X,Y-d,Umag_)
plt.colorbar(hc,ax=ax[2])
plot_overlay(ax[2])
ax[2].set_ylabel('|U|')

plt.show()


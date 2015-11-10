#!/usr/bin/python
import sys
from fenton1985 import *
import numpy as np
from scipy.optimize import fsolve

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

makeplots = True
DEBUG1 = False
DEBUG = False

TOL = 1e-8
#guesses = np.arange(0,1,.1)

g = 9.81 # gravity [m/s]
d = 4.0 # depth [m]
twopi = 2*np.pi

H = np.arange(0.05,2.05,0.05)
lam = np.arange(5,51)
if DEBUG1:
    H,lam = [1.2], [33.5676693735]
    makeplots = False
NH = len(H)
NT = len(lam)
coskx = np.zeros((NH,NL))
kx = np.zeros((NH,NL))
maxslope = np.zeros((NH,NL))

# stats
avgtries = 0.0
warnings = 0
found = 0
invalid = 0
invalid2 = 0

if DEBUG: # visually check roots# {{{
    x = np.linspace(-1,1)
    plt.figure()# }}}

for j in range(NL):
    k = twopi/lam[j]
    B22,B31,B42,B44,B53,B55 = evalB(k*d)

    def calc_cos_coefs(e):
        A = e + B31*e**3 - (B53+B55)*e**5
        B = B22*e**2 + B42*e**4
        C = -B31*e**3 + B53*e**5
        D = B44*e**4
        E = B55*e**5
        return A,B,C,D,E

    for i in range(NH):
        e = k*H[i]/2
        A,B,C,D,E = calc_cos_coefs(e)
        C0 = 16*D - 4*B
        C1 = A - 27*C + 125*E
        C2 = 8*B - 128*D
        C3 = 36*C - 500*E
        C4 = 128*D
        C5 = 400*E
        def coskx_eqn(x): 
            eqn = C0 + C1*x + C2*x**2 + C3*x**3 + C4*x**4 + C5*x**5 # = (curvature) / (-k)
            return eqn
        if DEBUG1:# {{{
            #t = k*3.25 # random test value
            t = k*30 # random test value
            print 'h=',H[i]
            print 'e=',e
            print 'lam=',lam[j]
            print 'k=',k
            print 'kx=',t
            print 'surf ', (A*np.cos(t) +   B*np.cos(2*t) +   C*np.cos(3*t) +    D*np.cos(4*t) +    E*np.cos(5*t))/k
            print 'slope',-(A*np.sin(t) + 2*B*np.sin(2*t) + 3*C*np.sin(3*t) +  4*D*np.sin(4*t) +  5*E*np.sin(5*t))
            print 'curva',-(A*np.cos(t) + 4*B*np.cos(2*t) + 9*C*np.cos(3*t) + 16*D*np.cos(4*t) + 25*E*np.cos(5*t))*k
            print 'coskx_eqn',-k*coskx_eqn(np.cos(kx))# }}}

        #coskx[i,j] = fsolve(coskx_eqn,0)
        coskx[i,j] = fsolve(coskx_eqn,1)

        t = np.arccos(coskx[i,j])
        kx[i,j] = t
        maxslope[i,j] = np.abs(A*np.sin(t) + 2*B*np.sin(2*t) + 3*C*np.sin(3*t) +  4*D*np.sin(4*t) +  5*E*np.sin(5*t))

        if DEBUG: #visually check points# {{{
            plt.plot(x,coskx_eqn(x),'0.7')
            plt.plot(coskx[i,j],0,'rx')
            plt.title('H=%f  lam=%f'%(H[i],lam[j]))
            plt.show()# }}}

#        tries = 0
#        for guess in guesses:
#            tries += 1
#            coskx[i,j], info, istat, mesg = fsolve(coskx_eqn,guess,full_output=True)
#            if np.abs(info['fvec']) < TOL:
#                if coskx[i,j] > -1 and coskx[i,j] < 1: #break
#                    t = np.arccos(coskx[i,j])
#                    surf = (A*np.cos(t) +   B*np.cos(2*t) +   C*np.cos(3*t) +    D*np.cos(4*t) +    E*np.cos(5*t))/k
#                    if surf > 0.75*H[i] or surf < -0.5*H[i]: 
#                        print 'cos(kx) in bounds, but H[i]=',H[i],' surf=',surf
#                        invalid2 += 1
#                    else: 
#                        found += 1
#                        break
#                else: invalid += 1
#        if np.abs(info['fvec']) > TOL or coskx[i,j] < -1 or coskx[i,j] > 1 \
#                or surf > 0.75*H[i] or surf < -0.5*H[i]: 
#            #print info
#            warnings += 1
#        else:
#            #kx[i,j] = np.arccos(coskx[i,j])
#            #t = kx[i,j]
#            kx[i,j] = t
#            maxslope[i,j] = np.abs(A*np.sin(t) + 2*B*np.sin(2*t) + 3*C*np.sin(3*t) +  4*D*np.sin(4*t) +  5*E*np.sin(5*t))
#            #print 'soln found in',tries,'tries'
#        avgtries += tries

Ncases = NH * NL
print Ncases,'cases processed'
#avgtries /= Ncases

print '  cos(kx) min/max :',np.min(coskx),np.max(coskx),'[-1,1]'
print '      kx  min/max :',np.min(kx),np.max(kx),'[0,2pi]'
print ' x/lambda min/max :',np.min(kx/twopi),np.max(kx/twopi),'[0,1]'
print '   dz/dx  min/max :',np.min(maxslope),np.max(maxslope)

#print 'fsolve converged after',avgtries,'tries on average'
#print found,'valid roots found'
#print invalid,'invalid roots found (outside [-1,1])'
#print invalid2,'invalid roots found (inside [-1,1], but not on surface)'
print 'Valid roots could not be found for',warnings,'cases'
assert( warnings==0 )

if makeplots:
    XX,YY = np.meshgrid(H,lam)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    #ax.plot_surface(XX,YY,coskx.T, rstride=1,cstride=1, cmap=cm.coolwarm)
    ax.plot_surface(XX,YY,kx.T/twopi,rstride=1,cstride=1, cmap=cm.coolwarm_r)
    ax.set_xlabel('waveheight [m]')
    ax.set_ylabel('wavelength [m]')
    #ax.set_zlabel('cos(kx)')
    ax.set_zlabel('x/lambda')
    ax.set_zlim((0,0.25))
    ax.set_title('Location of maximum vertical wave velocity')

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(XX,YY,maxslope.T, rstride=1,cstride=1, cmap=cm.coolwarm)
    ax.set_xlabel('waveheight [m]')
    ax.set_ylabel('wavelength [m]')
    #ax.set_zlabel('cos(kx)')
    ax.set_zlabel('|dz/dx|')
    ax.set_title('Maximum wave slope')

    plt.show()


#!/usr/bin/python
import sys
import os
import numpy as np
#from scipy.optimize import fsolve
from fenton1985 import *
from scipy import interpolate
from scipy.optimize import minimize_scalar

import matplotlib.pyplot as plt

periodsToCompare = 4
Nref = periodsToCompare*100
macro = 'setupSim.java'
surfDir = 'waterSurface'
errFile = 'errors.dat'

g = 9.81    # gravity

makeplots = True
saveplots = 'error'
verbose = False

#
# read parameters from macro
#
vartypes = ['double','int']
print 'Reading variables of type',vartypes,'from',macro,':'
with open(macro, 'r') as f:
    for line in f:
        line = line.strip()
        for vartype in vartypes:
            if line.startswith(vartype) and '=' in line:
                line = line[len(vartype)+1:].split(';')[0]
                print ' ',line
                exec(line)

#k = calculateWavenumber(g,T,H,d)
#lam = 2*np.pi/k
#print 'Calculated lambda =',lam
k = 2*np.pi/L
kd = k * d

#
# setup surface definition
#
xref = np.linspace(0,periodsToCompare*L,Nref)
B22,B31,B42,B44,B53,B55 = evalB(kd)
knorm = k*H/2
def surf(xoff=0):
    kn = knorm*np.cos(k*(xref-xoff)) \
        + knorm**2*B22*np.cos(2*k*(xref-xoff)) \
        + knorm**3*B31*(np.cos(k*(xref-xoff)) - np.cos(3*k*(xref-xoff))) \
        + knorm**4*(B42*np.cos(2*k*(xref-xoff)) + B44*np.cos(4*k*(xref-xoff))) \
        + knorm**5*( \
            -(B53+B55)*np.cos(k*(xref-xoff)) \
            + B53*np.cos(3*k*(xref-xoff)) \
            + B55*np.cos(5*k*(xref-xoff)) \
            )
    return kn/k
yref = surf()

#
# find csv files
#
csvfiles = []
for f in os.listdir(surfDir):
    if f.endswith(".csv"):
        csvfiles.append(surfDir+os.sep+f)
#print csvfiles

#
# sort csv files
#
Ntimes = len(csvfiles)
times = np.zeros((Ntimes))
i = -1
for csv in csvfiles:
    t = csv[:-4].split('_')[-1]
    #print t
    i += 1
    times[i] = float(t)
indices = [ i[0] for i in sorted(enumerate(times), key=lambda x:x[1]) ]

#
# define csv reader
#
def readCsv(fname,N=-1): # assume 1 header line
    if N < 0: 
        with open(fname,'r') as f:
            for line in f: N+=1
    x = np.zeros((N))
    y = np.zeros((N))
    with open(fname,'r') as csv:
        i = -1
        csv.readline()
        for line in csv:
            i += 1
            line = line.split(',')
            x[i] = float(line[0])
            y[i] = float(line[1])
    order = x.argsort()
    return x[order],y[order]

#
# process all csv files
#
first = True
times = times[indices]
t0 = times[0]
err = np.zeros((Ntimes))
for i in range(Ntimes):
    idx = indices[i]
    t = times[i]
    fname = csvfiles[idx]

    if first: #first read

        if verbose: print 'Processing',fname,'as first file'
        x,y = readCsv(fname)

        #
        # find the offset at (near) t=0
        #
        x -= x[0]
        fint = interpolate.interp1d(x,y)
        def diff(xoff):
            xint = np.linspace(xref[0]+xoff,xref[-1]+xoff,Nref)
            yint = fint(xint)
            e = yint - yref
            return e.dot(e) #L2 error
        result = minimize_scalar(diff,bounds=(0,x[-1]-xref[-1]),method='bounded')
        if verbose: print 'Calculating offset with optimizer'
        if verbose: print result
        if not result.success: print 'WARNING: optimizer did not converge'
        xoff0 = result.x
            
        first = False

    else:

        if verbose: print 'Processing',fname
        x,y = readCsv(fname)
        x -= x[0]
        fint = interpolate.interp1d(x,y)

    yref = surf(xoff0+U*(t-t0))
    e = fint(xref) - yref
    err[i] = e.dot(e)**0.5

    # dev: plot current and reference solutions
    if makeplots:
        plt.clf()
        plt.plot(xref,yref,'k--',linewidth=3)
        plt.plot(x,y,'b')
        if saveplots=='': plt.show()
        else: 
            plt.ylim((-0.55*H,0.55*H))
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title('wave surface')
            fname = saveplots+'_%04d.png'%(i)
            plt.savefig(fname)
            print 'Wrote',fname

#
# print final results
#
#print err
if not errFile=='':
    with open(errFile,'w') as f:
        for t,e in zip(times,err):
            f.write(' %f %g\n' % (t,e) )
    print 'Wrote',errFile

print '  max error:',err.max()
print '  cumulative error:',err.sum()

if makeplots:
    plt.clf()
    plt.semilogy(times,err)
    plt.show()


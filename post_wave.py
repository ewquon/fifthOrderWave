#!/usr/bin/python
import sys
import os
import numpy as np
from fenton1985 import *
from scipy import interpolate
from scipy.optimize import fsolve

import matplotlib.pyplot as plt

periodsToCompare = 4
Nref = periodsToCompare*100
macro = 'setupSim.java'
surfDir = 'waterSurface'
errFile = 'post.dat'

g = 9.81    # gravity
TOL = 1e-8  # tolerance for checking fsolve actually is getting to F(x)=0
NSMOO = 10  # smoothing for zero-finding, don't need to preserve amplitude
            # NOTE: if too large, can cause errors in numerical beach and/or
            #       in regions with large fluctuations

makeplots = True
showplots = False
saveplots = '' #'error'
savefinal = True

# DEBUG:
verbose = False
timing = False
checksurfplot = False

if timing: import time

if len(sys.argv) > 1: saveplots = sys.argv[1]

#
# read parameters from macro
#
vartypes = ['double','int']
print 'Reading variables of type',vartypes,'from',macro,':'
if timing: tlast = time.time()
with open(macro, 'r') as f:
    for line in f:
        if 'execute()' in line: break
        line = line.strip()
        for vartype in vartypes:
            if line.startswith(vartype) and '=' in line:
                line = line[len(vartype)+1:].split(';')[0]
                print ' ',line
                exec(line)

k = 2*np.pi/L
kd = k * d
if timing: tcurr = time.time(); print '  (done in %f s)'%(tcurr-tlast); tlast = tcurr

#
# setup surface definition
#
if timing: print 'setup surf def'
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
if timing: tcurr = time.time(); print '  (done in %f s)'%(tcurr-tlast); tlast = tcurr

#
# find csv files
#
if timing: print 'finding csv files'
csvfiles = []
for f in os.listdir(surfDir):
    if f.endswith(".csv"):
        csvfiles.append(surfDir+os.sep+f)
if timing: tcurr = time.time(); print '  (done in %f s)'%(tcurr-tlast); tlast = tcurr

#
# sort csv files
#
if timing: print 'sorting csv files'
Ntimes = len(csvfiles)
times = np.zeros((Ntimes))
i = -1
for csv in csvfiles:
    t = csv[:-4].split('_')[-1]
    i += 1
    times[i] = float(t)
indices = [ i[0] for i in sorted(enumerate(times), key=lambda x:x[1]) ]
if timing: tcurr = time.time(); print '  (done in %f s)'%(tcurr-tlast); tlast = tcurr

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
if timing: print 'processing csv files'
times = times[indices]
t0 = times[0]
err = np.zeros((Ntimes))
lam = np.zeros((Ntimes))
for itime in range(Ntimes):
    idx = indices[itime]
    t = times[itime]
    fname = csvfiles[idx]

    if itime==0: #first read

        if verbose: print 'Processing',fname,'as first file'
        x,y = readCsv(fname)
        x -= x[0]
        #fint = interpolate.interp1d(x,y)

        #
        # find the offset at (near) t=0
        #
        def inlet(xoff): # ==surf(xref=0)
            kn = knorm*np.cos(k*(-xoff)) \
                + knorm**2*B22*np.cos(2*k*(-xoff)) \
                + knorm**3*B31*(np.cos(k*(-xoff)) - np.cos(3*k*(-xoff))) \
                + knorm**4*(B42*np.cos(2*k*(-xoff)) + B44*np.cos(4*k*(-xoff))) \
                + knorm**5*( \
                    -(B53+B55)*np.cos(k*(-xoff)) \
                    + B53*np.cos(3*k*(-xoff)) \
                    + B55*np.cos(5*k*(-xoff)) \
                    )
            return kn/k - y[0]
        def inletSlope(xoff):
            ytmp = surf(xoff)
            return (ytmp[1]-ytmp[0])/(xref[1]-xref[0])
        guess = 0 #-L/2
        xoff0,info,istat,msg = fsolve(inlet,guess,full_output=True)
        #print info
        if isinstance(xoff0,np.ndarray): xoff0 = xoff0[0]
        while xoff0 < 0 or \
                (not np.sign(inletSlope(xoff0)) == np.sign(y[1]-y[0])) or \
                np.abs(info['fvec'][0]) > TOL:
            guess += L/4
            xoff0,info,istat,msg = fsolve(inlet,guess,full_output=True)
            #print info
            if verbose: print '  guess:',guess,' xoff,slope=',xoff0,inletSlope(xoff0)
            if isinstance(xoff0,np.ndarray): xoff0 = xoff0[0]
        print 'initial offset:',xoff0
            
    else:

        if verbose: print 'Processing',fname
        x,y = readCsv(fname)
        x -= x[0]
        #fint = interpolate.interp1d(x,y)

    # check for distinct x
    ux = set(x)
    if not len(ux)==len(x): # interp can return NaN
        N = len(ux)
        ynew = np.zeros((N))
        print 't=',t,': correcting non-distinct x values',len(x),'to',len(ux)
        i = -1
        ysum = 0
        dup = 0
        for i0 in range(len(x)-1):
            if not x[i0+1]==x[i0]:
                i += 1
                #print i0,i,x[i0],y[i0]
                if ysum==0:
                    ynew[i] = y[i0]
                else:
                    ynew[i] = (ysum+y[i0])/(dup+1.0)
                    #print ysum+y[i0],dup+1,'AVG',ynew[i]
                    ysum = 0
                    dup = 0
            else: # average this value
                ysum += y[i0]
                dup += 1
                #print i0,i,x[i0],y[i0],'DUP',ysum,dup
        i0 += 1
        i += 1
        assert( i==N-1 )
        #print i0,i,x[i0],y[i0]
        if ysum==0:
            ynew[i] = y[i0]
        else:
            ynew[i] = (ysum+y[i0])/(dup+1.0)
            #print ysum+y[i0],dup+1,'AVG',ynew[i]
        x = np.array(list(ux))
        y = ynew

    yref = surf(xoff0+U*(t-t0))
    #e = fint(xref) - yref
    e = np.interp(xref,x,y) - yref
    err[itime] = np.dot(e,e)**0.5

    #
    # estimate wavelength
    #
    # - first smooth signal
    N = len(y)
    ysmoo = np.zeros((N))
    for i in range(N):
        ist = max(0,i-NSMOO)
        ind = min(N,i+NSMOO)
        ysmoo[i] = np.mean(y[ist:ind])
    # - then check for zero crossings
    guesses = []
    for i in range(0,len(y)-1):
        #if y[i-1]*y[i+1] < 0: guesses.append(x[i])
        if not np.sign(ysmoo[i])==np.sign(ysmoo[i+1]): 
            x0 = x[i] - ysmoo[i]*(x[i+1]-x[i])/(ysmoo[i+1]-ysmoo[i])
            guesses.append(x0)
        if len(guesses) >= 2*periodsToCompare: break
    guesses = np.array(guesses)
    lam[itime] = 2*np.mean(np.diff(guesses))

    if checksurfplot:
        print guesses
        print 't,lambda',t,lam[itime]
        plt.plot(x,y,'ko')
        plt.plot(x,ysmoo,'b-')
        plt.plot(guesses,np.zeros(len(guesses)),'rx')
        plt.show()

    #
    # plot current and reference solutions
    #
    if makeplots:
        if saveplots or (savefinal and itime==Ntimes-1):
            plt.clf()
            plt.plot(xref,yref,'k:',linewidth=3,label='theory')
            plt.plot(x,y,'b',label='simulation')
            plt.plot(guesses,np.zeros(len(guesses)),'bo')
        
        if saveplots: #not saveplots==''
            plt.ylim((-0.55*H,0.55*H))
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title('wave surface')
            fname = saveplots+'_%04d.png'%(itime)
            plt.savefig(fname)
            print 'Wrote',fname
        elif showplots: plt.show()

# end of time loop

#
# print final results
#
if not errFile=='':
    with open(errFile,'w') as f:
        f.write(' t max_error wavelength\n')
        for t,e,l in zip(times,err,lam):
            f.write(' %f %g %f' % (t,e,l) )
            f.write('\n')
    print 'Wrote',errFile

print '  max error:',err.max()
print '  cumulative error:',err.sum()

# wavelength error
coefs = np.polyfit(times,lam,3)
p = np.poly1d(coefs)
lam_final = p(times[-1])
print '  final wavelength:',lam_final
print '  wavelength error:',(lam_final-L)/L#*100,'%'

# scale the wave profile to get a more realistic estimate 
# of the wave amplitude error
fint = interpolate.interp1d(L/lam_final*x,y)
yscaled = fint(xref)
e = yscaled - yref
print '  final wavelength-corrected error:', np.dot(e,e)**0.5

if makeplots:
    plt.plot(xref,yscaled,'r--',label='sim, corrected')
    plt.title('T=%.2f s, H=%.2f m, lam=%f m (nH=%d,nL=%d,cfl=%.4f)' \
            % (T,H,L,nH,nL,cfl) )
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('z')
    if showplots: plt.show()
    if savefinal: plt.savefig('final_surf.png')

    plt.figure()
    plt.semilogy(times,err)
    plt.xlabel('time')
    plt.ylabel('||error||')
    if showplots: plt.show()
    if savefinal: plt.savefig('final_error.png')


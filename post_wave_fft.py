#!/usr/bin/python
import sys
import os
import numpy as np
from fenton1985 import *
from scipy.interpolate import interp1d
from numpy import fft

macro = 'setupSim.java'
surfDir = 'waterSurface'

periodsToCompare = 4
g = 9.81    # gravity
NSMOO = 10  # smoothing for zero-finding, don't need to preserve amplitude

# DEBUG:
verbose = False
debug = False
makeplot = True

#
# read parameters from macro
#
vartypes = ['double','int']
print 'Reading variables of type',vartypes,'from',macro,':'
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
if verbose: print 'k=',k
kd = k * d

#
# find csv files
#
csvfiles = []
for f in os.listdir(surfDir):
    if f.endswith(".csv"):
        csvfiles.append(surfDir+os.sep+f)

#
# sort csv files
#
Ntimes = len(csvfiles)
times = np.zeros((Ntimes))
i = -1
for csv in csvfiles:
    t = csv[:-4].split('_')[-1]
    i += 1
    times[i] = float(t)
indices = [ i[0] for i in sorted(enumerate(times), key=lambda x:x[1]) ]
times = times[indices]

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
    x = x[order]
    y = y[order]
    #assert( not any( np.diff(x)==0 ) )
    if any(np.diff(x)==0):
        print 'Warning: duplicate x found in',fname
        if debug: print 'starting N=',N
        newx = np.zeros((N))
        newy = np.zeros((N))
        newN = N
        i = 0
        inew = 0
        while i < N:
            if debug: print 'checking i=',i,' ( inew=',inew,')'
            if i < N-1 and x[i+1]==x[i]:
                if debug: print '  i=',i,'== i=',i+1
                i2 = i+2
                while i2 < N and x[i2]==x[i]:
                    if debug: print '  i=',i,'== i=',i2
                    i2 += 1
                newx[inew] = x[i]
                newy[inew] = np.mean(y[i:i2])
                ndup = i2-i-1
                i += ndup+1
                newN -= ndup
                if debug: print '  repeated',ndup,'times'
                if debug: print '  new N=',newN
            else: 
                newx[inew] = x[i]
                newy[inew] = y[i]
                i+= 1
            inew += 1
        x = newx[:newN]
        y = newy[:newN]
    return x,y
#
# process all csv files
#
if len(sys.argv) <= 1: 
    plots = [times[-1]]
else:
    plots = [ float(val) for val in sys.argv[1:] ]

if makeplot: 
    import matplotlib.pyplot as plt
    fig = plt.figure()
    #fig.subplots_adjust(bottom=0.2) # defaults: bottom=0.1

for selectedTime in plots:
    itime = np.nonzero(times >= selectedTime)[0][0]
    idx = indices[itime]
    t = times[itime]
    fname = csvfiles[idx]
    if verbose: print 'Selected time',selectedTime,'- found t=',t,' (',fname,')'

    # read profile data
    x,y = readCsv(fname)
    x -= x[0]
    if makeplot:
        plt.subplot(311)
        if len(plots)==1: plt.plot(x,y,label='simulated surface')
    dx = np.mean(np.diff(x))
    if verbose: print '  avg dx =',dx

    # first smooth the signal
    Nx = len(x)
    ysmoo = np.zeros((Nx))
    for i in range(Nx):
        ist = max(0,i-NSMOO)
        ind = min(Nx,i+NSMOO)
        ysmoo[i] = np.mean(y[ist:ind])

    # then find crests
    dy = np.zeros(Nx)
    xm,ym = [],[]
#    xn,yn = [],[] # points near the min/maxima
    crests = []
    for i in range(1,Nx-1): # calculate first derivative from smoothed signal
#        dy[i] = (y[i+1]-y[i-1])/(x[i+1]-x[i-1])
        dy[i] = (ysmoo[i+1]-ysmoo[i-1])/(x[i+1]-x[i-1])
    for i in range(1,Nx-2):
        if not np.sign(dy[i+1]) == np.sign(dy[i]): #local min/maximum
#            xn += [x[i],x[i+1]]
#            yn += [y[i],y[i+1]]
            x0 = x[i] - (x[i+1]-x[i])/(dy[i+1]-dy[i])*dy[i]
            #y0 = y[i] + (y[i+1]-y[i])/(x[i+1]-x[i])*(x0-x[i])
            fint = interp1d( x[i-1:i+3], y[i-1:i+3], kind='cubic' )
            y0 = fint(x0)
#            curv1 = (y[i+1] - 2*y[i]   + y[i-1]) / np.mean( np.diff(x[i-1:i+2]) )
#            curv2 = (y[i+2] - 2*y[i+1] + y[i]  ) / np.mean( np.diff(x[i  :i+3]) )
            curv1 = (ysmoo[i+1] - 2*ysmoo[i]   + ysmoo[i-1]) / np.mean( np.diff(x[i-1:i+2]) )
            curv2 = (ysmoo[i+2] - 2*ysmoo[i+1] + ysmoo[i]  ) / np.mean( np.diff(x[i  :i+3]) )
            if verbose:
                print 'checking (%f,%f) (%f,%f) (%f,%f) (%f,%f)' \
                    % ( x[i-1],y[i-1], x[i],y[i], x[i+1],y[i+1], x[i+2],y[i+2] )
                print '  min/maximum at (%f,%f)' % (x0,y0)
                print '  curvatures',curv1,curv2
            assert( x0 >= x[i] and x0 <= x[i+1] )
            #assert( np.sign(curv1) == np.sign(curv2) )
            if not np.sign(curv1) == np.sign(curv2):
                if np.abs(curv1) > np.abs(curv2): curv = curv1
                else: curv = curv2
                if verbose: print '  using curv=',curv
            else: curv = curv1
            if curv < 0: 
                xm.append(x0)
                ym.append(y0)
                crests.append(i)

    ist = crests[0]
    ind = crests[periodsToCompare]

    if makeplot and len(plots)==1:
#        plt.plot(x[ist:ind],y[ist:ind],'k-',linewidth=3)
        plt.plot(x,ysmoo,'g--',label='smoothed signal')
        plt.plot(xm,ym,'r+',label='detected maxima')
#        plt.plot(xn,yn,'g.')

    # build signal at equally spaced intervals
    def nextpow2(Nx):
        i = np.log2(Nx)
        N = np.ceil(i)
        return int(2**N)
    Nfft = nextpow2(Nx)
    xfft = np.linspace(x[ist],x[ind-1],Nfft)
    dx = xfft[1] - xfft[0]
    fint = interp1d( x, y, kind='cubic' )
    yfft = fint( xfft )
    if makeplot:
        if len(plots)==1:
            plt.plot(xfft,yfft,'k.',label='FFT input signal')
            plt.legend(loc='best')
        else:
            plt.plot(xfft,yfft)

    # perform fft
    #F = fft.fft(y)/N
    #F = fft.fft(y[ist:ind])/N
    F = fft.fft(yfft) / Nfft
    wn = fft.fftfreq(Nfft,dx) # wavenumber
    wn *= L #normalize
    P = np.abs(F)**2 # spectral power
    if verbose: 
        print 'N =',Nx,' ( nextpow2:',Nfft,')'

    kidx = np.nonzero(wn > 5)[0][0]
    hfe = np.sum(P[kidx:Nfft/2]) * (wn[Nfft/2-1]-wn[kidx])

    print 't=',t,':  peak spatial frequency * (2pi/k), high-freq error =',wn[np.argmax(P[:Nfft/2])],hfe

    if makeplot:
        plt.subplot(312)
        #plt.plot(wn[:kidx],P[:kidx],label='t=%.2f'%(t))
        plt.plot(wn[:Nfft/2],P[:Nfft/2],label='t=%.2f'%(t))
        plt.subplot(313)
        plt.loglog(wn[:Nfft/2],P[:Nfft/2],label='t=%.2f'%(t))

if makeplot:
    plt.subplot(311)
    plt.xlabel('x')
    plt.ylabel('z(t)')

    plt.subplot(312)
    plt.xlabel('k / k_1')
    plt.ylabel('|F{z}|^2')
    plt.legend(loc='best')
    plt.xlim((0,6))
    #plt.ylim((0,0.01))

    plt.subplot(313)
    plt.xlabel('k / k_1')
    plt.ylabel('|F{z}|^2')
    plt.xlim((0.1,20))
    plt.ylim((1e-12,0.1))

    #plt.tight_layout() # not in version of matplotlib on peregrine

    plt.savefig('fft.png')

    if len(plots)==1: plt.show()


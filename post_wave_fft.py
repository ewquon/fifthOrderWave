#!/usr/bin/python
import sys
import os
import numpy as np
from fenton1985 import *
from numpy import fft

import matplotlib.pyplot as plt

macro = 'setupSim.java'
surfDir = 'waterSurface'

g = 9.81    # gravity

# DEBUG:
verbose = True
debug = False

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

for selectedTime in plots:
    itime = np.nonzero(times >= selectedTime)[0][0]
    idx = indices[itime]
    t = times[itime]
    fname = csvfiles[idx]
    if verbose: print 'Selected time',selectedTime,'- found t=',t,' (',fname,')'

    # read profile data
    x,y = readCsv(fname)
    x -= x[0]
    plt.subplot(211)
    plt.plot(x,y)
    N = len(x)
    dx = np.mean(np.diff(x))
    print '  dx =',dx

    # perform fft
    F = fft.fft(y)/N
    wn = fft.fftfreq(N,dx) # wavenumber
    P = np.abs(F)**2 # spectral power

    # plot result
    plt.subplot(212)
    plt.plot(wn[:N/2]/k,P[:N/2],label='t=%f'%(t))

plt.subplot(211)
plt.xlabel('x')
plt.ylabel('y')
plt.subplot(212)
plt.xlabel('k / k_1')
plt.ylabel('|F{y}|^2')
plt.legend(loc='best')

plt.show()

#!/usr/local/bin/python
import numpy as np
from scipy import interpolate
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

exactFile = 'exact_ss0.dat'
starData = 'star_ss0.dat'

Nref = 0
with open(exactFile,'r') as f:
    for line in f: Nref += 1
xref = np.zeros((Nref))
yref = np.zeros((Nref))
with open(exactFile,'r') as f:
    i = -1
    for line in f:
        i += 1
        linedata = line.split()
        xref[i] = float(linedata[0])
        yref[i] = float(linedata[1])
print 'Read',Nref,'reference points from',exactFile
print '  xref start/end',xref[0],xref[-1]

N = 0
with open(starData,'r') as f:
    for line in f: N += 1
x = np.zeros((N))
y = np.zeros((N))
with open(starData,'r') as f:
    i = -1
    for line in f:
        i += 1
        linedata = line.split()
        x[i] = float(linedata[0])
        y[i] = float(linedata[1])
print 'Read',N,'points from',starData
print '  x range :',np.min(x),np.max(x)
print '  y range :',np.min(y),np.max(y)
x -= x[0]
print '  shifted x start/end',x[0],x[-1]

# find offset...
fint = interpolate.interp1d(x,y)
def err(off):
    xint = np.linspace(off,xref[-1]+off,Nref)
    yint = fint(xint)
    e = yint - yref
    return e.dot(e) #L2 error
#res = minimize_scalar(err)
res = minimize_scalar(err,bounds=(0,x[-1]-xref[-1]),method='bounded')
#print res
offset = res.x
error = res.fun**0.5
print res.message
print 'L2 error:',error

# plot comparison
xoff = xref+offset
plt.plot(x,y,'b-',label='Star')
plt.plot(xoff,fint(xoff),'b+',label='Star (interp)')
plt.plot(xoff,yref,'ks',label='Theory')
plt.legend(loc='best')
plt.show()


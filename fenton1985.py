#!/usr/bin/python
import sys
import numpy as np

#def evalA(kd): #not used# {{{
#    A11 = 1./np.sinh(kd)
#    A22 = 3.*S**2/(2.*(1.-S)**2)
#    denA31 = (8.*np.sinh(kd*(1.-S)**3))
#    A31 = (-4. - 20.*S + 10.*S**2 - 13.*S**3)/denA31
#    A33 = (-2.*S**2 + 11.*S**3)/denA31
#    A42 = (12.*S - 14.*S**2 - 264.*S**3 - 45.*S**4 - 13.*S**5)/(24.*(1.-S)**5)
#    A44 = (10.*S**3 - 174.*S**4 + 291.*S**5 + 278.*S**6)/(48.*(3.+2.*S)*(1.-S)**5)
#    A51 = (-1184. + 32.*S + 13232.*S**2 + 21712.*S**3 + 20940.*S**4 + 12554.*S**5 - 500*S**6 \
#        - 3341.*S**7 - 670.*S**8)/(64.*np.sinh(kd*(3.+2.*S)*(4.+S)*(1.-S)**6))
#    A53 = (4.*S + 105.*S**2 + 198.*S**3 - 1376.*S**4 - 1302.*S**5 - 117.*S**6 + 58.*S**7) \
#        / (32.*np.sinh(kd*(3.+2.*S)*(1.-S)**6))
#    A55 = (-6.*S**3 + 272.*S**4 - 1552.*S**5 + 852.*S**6 + 2029.*S**7 + 430.*S**8) \
#        / (64.*np.sinh(kd*(3.+2.*S)*(4.+S)*(1.-S)**6))
#    return A11,A22,A33,A44,A55,A31,A42,A51,A53# }}}
#def evalE(kd): #not used# {{{
#    E2 = np.tanh(kd) * (2 + 2.*S + 5.*S**2)/(4.*(1.-S)**2)
#    E4 = np.tanh(kd) * (8. + 12.*S - 152.*S**2 - 308.*S**3 - 42.*S**4 + 77.*S**5)/(32.*(1.-S)**5)
#    return E2,E4# }}}

# coefficients from Fenton 1985, Table 1# {{{
# B coefficients -- should be correct, free surface profiles are qualitatively correct
def evalB(kd):
    S = 1./np.cosh(2*kd)
    B22 = 1./np.tanh(kd)*(1.+2.*S)/(2.*(1.-S))
    B31 = -3.*(1. + 3.*S + 3.*S**2 + 2.*S**3)/(8.*(1.-S)**3)
    B42 = 1./np.tanh(kd)*(6. - 26.*S - 182.*S**2 - 204.*S**3 - 25.*S**4 + 26.*S**5)/(6.*(3.+2.*S)*(1.-S)**4)
    B44 = 1./np.tanh(kd)*(24. + 92.*S + 122.*S**2 + 66.*S**3 + 67.*S**4 + 34.*S**5)/(24.*(3.+2.*S)*(1.-S)**4)
    B53 = 9.*(132. + 17.*S - 2216.*S**2 - 5897.*S**3 - 6292.*S**4 - 2687.*S**5 + 194.*S**6 \
        + 467.*S**7 + 82.*S**8)/(128.*(3.+2.*S)*(4.+S)*(1.-S)**6)
    B55 = 5.*(300. + 1579.*S + 3176.*S**2 + 2949.*S**3 + 1188.*S**4 + 675.*S**5 + 1326.*S**6 \
        + 827.*S**7 + 130.*S**8)/(384.*(3.+2.*S)*(4.+S)*(1.-S)**6)
    return B22,B31,B42,B44,B53,B55

# C coefficients -- should be correct, are approximately equal to wavelengths calculated by Star
def evalC(kd):
    S = 1./np.cosh(2*kd)
    C0 = np.tanh(kd)**0.5
    C2 = C0 * (2. + 7.*S**2)/(4.*(1.-S)**2)
    C4 = C0 * (4. + 32.*S - 116.*S**2 - 400.*S**3 - 71.*S**4 + 146.*S**5)/(32.*(1.-S)**5)
    return C0,C2,C4

# D coefficients
def evalD(kd):
    S = 1./np.cosh(2*kd)
    D2 = -np.tanh(kd)**-0.5/2.
    D4 =  np.tanh(kd)**-0.5 * (2. + 4.*S + S**2 + 2.*S**3)/(8.*(1.-S)**3)
    return D2,D4# }}}

def calculateWavenumber(g,T,H,d,guess=None):
    from scipy.optimize import fsolve
    if guess < 0:
        guess = 4*np.pi**2/g/T**2 # deep water approximation, use as a starting guess
    def eqn23_L2(k): # TODO: handle mean current speed not 0
	C0,C2,C4 = evalC(k*d)
	F = -2*np.pi/T/(g*k)**0.5 + C0 + (k*H/2)**2*C2 + (k*H/2)**4*C4
	#return F*F
	return F
    k = fsolve(eqn23_L2,guess)
    if isinstance(k,np.ndarray): k = k[0] # depending on version, may return array or scalar

    return k


###############################################################################
###############################################################################
###############################################################################

if __name__ == '__main__':

    #-----------------------
    # hard-coded parameters
    g = 9.81 # m/s^2
    d = 4.0     # m, depth
    #-----------------------
    
    output = ''
    verbose = True
    if len(sys.argv) < 3:
        # need to at least specify the sea state in terms of T and H
        print '\nUSAGE:\n'
        print ' - calculate wavelength (lambda) and mean wave speed (U)'
        print '   with option to display plot or save surface profile'
        print '   coordinates to file\n'
        print '  ',sys.argv[0]+' [T] [H]'
        print '  ',sys.argv[0]+' [T] [H] plot\t\t(requires matplotlib)'
        print '  ',sys.argv[0]+' [T] [H] [filename]\n'
        print ' - abbreviated output of lambda and U\n'
        print '  ',sys.argv[0]+' [T] [H] short\n'
        sys.exit()
    else:
        T = float(sys.argv[1])
        H = float(sys.argv[2])
    if len(sys.argv) > 3: 
        try:
            lam = float(sys.argv[3])
            if len(sys.argv) > 4: output = sys.argv[4]
        except ValueError:
            output = sys.argv[3]
    if not output=='' and \
            (output=='short' or output[:3].lower()=='lam' or output[0].lower()=='u'): verbose = False
    
    if verbose:
        print 'Input period            :',T,'s'
        print 'Input wave height       :',H,'m'
    
    #
    # calculate wave number
    #
    try:
        # if lambda is specified
        k = 2*np.pi/lam
        if verbose: print 'Input wave length       :',lam,'m'
    except NameError: 
        # wavelength not specified, calculate it from T
        # requires scipy.optimize
        kdeep = 4*np.pi**2/g/T**2 # deep water approximation, use as a starting guess
        lam_deep = 2*np.pi/kdeep
        k = calculateWavenumber(g,T,H,d,kdeep)
        lam = 2*np.pi/k
        Ur = H/d*(lam/d)**2 #Ursell number

        if verbose:
            print 'Approximate wave length :',lam_deep,'m  (in deep water)'
            print 'Calculated wave length  :',lam,'m  (diff=%f%%, Ur=%f)' % \
                    ( 100*(lam-lam_deep)/lam_deep, Ur )
    
    e = k*H/2 #dimensionless wave height
    kd = k*d
    
    # mean horizontal fluid speed (eqn.13)
    C0,C2,C4 = evalC(kd)
    unorm = C0 + e**2*C2 + e**4*C4
    umean = unorm*(g/k)**0.5
    if verbose: print 'Mean wave speed         :',umean,'m/s'
    
    # volume flux under wave (eqn.15)
    #D2,D4 = evalD(kd)
    #Qnorm = unorm*kd + e**2*D2 + e**4*D4
    #print 'Volume flux under the wave:',Qnorm*(g/k**3)**0.5,'m^2/s'

    if output=='': sys.exit()
    
    if output=='short':
        print lam, umean
        sys.exit()
    elif output[:3].lower()=='lam':
        print lam
        sys.exit()
    elif output[0].lower()=='u':
        print umean
        sys.exit()
    
    #
    # calculate additional quantities for plotting
    #

    x = np.linspace(0,2*lam,200)
    
    # free surface profile (eqn.14)
    B22,B31,B42,B44,B53,B55 = evalB(kd)
    kn = kd + e*np.cos(k*x) \
            + e**2*B22*np.cos(2*k*x) \
            + e**3*B31*(np.cos(k*x) - np.cos(3*k*x)) \
            + e**4*(B42*np.cos(2*k*x) + B44*np.cos(4*k*x)) \
            + e**5*(-(B53+B55)*np.cos(k*x) + B53*np.cos(3*k*x) + B55*np.cos(5*k*x))
    y = kn/k-d
    
    if output=='plot':
        import matplotlib.pyplot as plt
        plt.plot(x,y)
        xranges=[x[0],x[-1]]
        plt.xlim(xranges)
        plt.xlabel('x')
        plt.ylabel('free surface height')
        plt.title('lambda=%f m (k=%f 1/m):  T=%f s,  H=%f m'%(lam,k,T,H))
        plt.show()
    
    elif not output=='':
        with open(output,'w') as f:
            for xi,yi in zip(x,y):
                f.write(' %f %f\n'%(xi,yi))
        print 'Wrote',output
    

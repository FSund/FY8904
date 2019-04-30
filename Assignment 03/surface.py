from scipy.special import jv
from cmath import pi, exp
from math import factorial
import numpy as np


def bessel(order, argument):
    '''
    Bessel function of the first kind of real order and complex argument
    '''
    return jv(order, argument)


class Surface:
    def __init__(self, a, xi0):
        self.a = a
        self.xi0 = xi0


class DoubleCosine(Surface):
    '''
    doubly-period cosine profile with period a
    '''

    def __init__(self, a, xi0):
        super().__init__(a, xi0)

    def ihat(self, gamma, G):
        h1 = G.h1
        h2 = G.h2

        # from assignment (I think this is the correct form)
        # value = (-1j)**h1*bessel(h1, gamma*xi0/2)*(-1j)**h2*bessel(h2, gamma*xi0/2)
        value = (-1j)**(h1 + h2)*bessel(h1, gamma*self.xi0/2) * \
            bessel(h2, gamma*self.xi0/2)

        # from paper (notice change from -i to -1)
        # value = (-1)**(h1 + h2)*bessel(h1, gamma*xi0/2)*bessel(h2, gamma*xi0/2)

        return value


class TruncatedCone(Surface):
    def __init__(self, a, xi0, rho_t, rho_b):
        assert(rho_b > rho_t)
        super().__init__(a, xi0)
        self.rho_t = rho_t
        self.rho_b = rho_b

    def ihat(self, gamma, G):
        '''
        Equation (59)
        '''
        # kronecker delta
        if G.h1 == 0 and G.h2 == 0:
            first = 1
        else:
            first = 0

        if G.h1 == 0 and G.h2 == 0:
            besselTerm = 1/2
        else:
            term = G.abs()*self.rho_t
            besselTerm = bessel(1, term)/term

        second = 2*pi*self.rho_t**2/self.a**2*(exp(-(1j)*gamma*self.xi0) - 1+0j)*besselTerm
        third = 2*pi*(self.rho_b - self.rho_t)/self.a**2*self.theSum(gamma, G)

        return first + second + third

    def theSum(self, gamma, G):
        '''
        The sum and integral in eq. (59)
        '''
        n = 1
        out = 0
        Gabs = G.abs()
        nMax = 20
        while True:
            first = (-(1j)*gamma*self.xi0)**n/factorial(n)
            second = self.theIntegral(Gabs, n)
            diff = first*second
            out += diff

            if abs(diff/out) < 0.01 or n >= nMax:
                # reckon it's converged if diff is less than 1%
                break

            n += 1

        # print("Used {} terms to converge".format(n))

        return out

    def theIntegral(self, Gabs, n):
        def f(u):
            # the integrand
            term = self.rho_b - (self.rho_b - self.rho_t)*u
            return term*bessel(0, Gabs*term)*u**n

        x = np.linspace(0, 1, 100)  # 100 steps???
        y = f(x)

        return np.trapz(y, x)


class TruncatedCosine(Surface):
    def __init__(self, a, xi0, rho0):
        super().__init__(a, xi0)
        self.rho0 = rho0

    def ihat(self, gamma, G):
        raise RuntimeError("truncatedCosine() not implemented yet")

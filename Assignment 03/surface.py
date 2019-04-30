from scipy.special import jv
from cmath import pi, exp, cos
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
        if G.h1 == 0 and G.h2 == 0:
            first = 1  # kronecker delta
            besselTerm = 1/2
        else:
            first = 0  # kronecker delta
            term = G.abs()*self.rho_t
            besselTerm = bessel(1, term)/term

        second = 2*pi*self.rho_t**2/self.a**2*(exp(-(1j)*gamma*self.xi0) - 1)*besselTerm
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
        while True:  # sum over n
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
        assert(rho0 < a/2)
        super().__init__(a, xi0)
        self.rho0 = rho0

    def ihat(self, gamma, G):
        '''
        Equation (61)
        '''
        if G.h1 == 0 and G.h2 == 0:
            first = 1  # kronecker delta
        else:
            first = 0  # kronecker delta

        second = 2*pi/self.a**2*self.theSum(gamma, G)

        return first + second

    def theSum(self, gamma, G):
        '''
        The sum and integral in eq. (61)
        '''
        n = 1
        out = 0
        Gabs = G.abs()
        nMax = 20
        while True:  # sum over n
            first = (-(1j)*gamma)**n/factorial(n)
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
        def f(x, n):
            # the integrand
            return x*bessel(0, Gabs*x)*self.xi(x)**n

        x = np.linspace(0, self.rho0, 100)
        y = f(x, n)

        return np.trapz(y, x)

    # def firstIntegral(self, gamma, G):
    #     def f(x_i, Gabs):
    #         phi = np.linspace(-pi, pi, 100)
    #         y = (exp(-(1j)*gamma*self.xi(x_i)) - 1)*exp(-(1j)*Gabs*x_i*cos(phi))
    #         return np.trapz(y, phi)

    #     x = np.linspace(0, self.rho0, 100)
    #     y = np.zeros(len(x))
    #     Gabs = G.abs()
    #     for i in range(len(x)):
    #         y[i] = f(x[i], Gabs)

    #     return np.trapz(y, x)

    def xi(self, x):
        # if x > self.rho0:
        #     return 0
        # else:
        #     return self.xi0*cos(pi*x/(2*self.rho0))

        # vectorized version
        out = self.xi0*np.cos(pi*x/(2*self.rho0))
        out[x > self.rho0] = 0  # truncate
        return out

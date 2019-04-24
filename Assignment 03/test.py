import numpy as np
from scipy.special import jv
from math import sqrt, pi


c = 1
c2 = c*c


def xi(x):
    return 0  # flat surface


def bessel(order, value):
    # bessel functions of the first kind of real order and complex argument
    return jv(order, value)


def ihat(gamma, h1, h2, xi):
    # doubly-period cosine profile with period a
    return (-1j)**h1*bessel(h1, gamma*xi/2)*(-1j)**h2*bessel(h2, gamma*xi/2)


class LatticeSite:
    def __init__(self, h1=0, h2=0):
        self.h1 = h1
        self.h2 = h2


class G(LatticeSite):
    def __init__(self, h1=0, h2=0, a=0):
        self.h1 = 2*pi/a*h1
        self.h2 = 2*pi/a*h2


def alpha0(lam, omega):
    omega2 = omega*omega
    k = 2*pi/lam
    k2 = k*k
    if k**2 < omega2/c2:
        return sqrt(omega2/c2 - k2)
    else:
        return 1j*(k2 - omega2/c2)


# def alpha0(omeag, theta, c):
#     k = 2*pi/
#     return 5

def M(pvec, qvec):
    return 1  # Dirichlet


def N(pvec, qvec):
    return 1  # Dirichlet


def RHS():
    return -ihat(gamma=alpha0(k, omega), G=K - k, xi=xi0)*N(K, k)


def LHS(Kprime):
    return ihat(gamma=-alpha0(Kprime, omega), G=K - Kprime, xi=xi0)*M(K, Kprime)


print(bessel(1, 2))
print(ihat(gamma=1, G=LatticeSite()))


# properties of incident wave vector k
theta0 = 0  # polar angle of incidence (vary this?)
phi0 = 0  # azimuthal angle of incidence (0 or 45 in the article)


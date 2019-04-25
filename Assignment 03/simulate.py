import numpy as np
from scipy.special import jv
from math import sqrt, pi, sin, cos
from time import sleep


# length unit is lambda
a = 3.5  # lattice dimension
xi0 = 0.3  # surface amplitude


class X3Vector:
    # a general 2d vector (in the x3-plane)
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.magnitude = sqrt(self.x**2 + self.y**2)

    def __add__(self, other):
        return X3Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return X3Vector(self.x - other.x, self.y - other.y)


class LatticeSite(X3Vector):
    # reciprocal lattice site (mostly G or G')
    def __init__(self, h1=0, h2=0):
        x = h1*2*pi/a
        y = h2*2*pi/a
        super().__init__(x, y)

        self.h1 = h1
        self.h2 = h2

    def __add__(self, other):
        return LatticeSite(self.h1 + other.h1, self.h2 + other.h2)

    def __sub__(self, other):
        return LatticeSite(self.h1 - other.h1, self.h2 - other.h2)

# class Reciprocal(LatticeSite):
#     def __init__(self, h1=0, h2=0):
#         self.h1 = 2*pi/a*h1
#         self.h2 = 2*pi/a*h2


class IncidentWave(X3Vector):
    # the part of the incident wave, \vec k, that is parallel to the x3-plane
    # aka. \vec k_\|
    def __init__(self, theta0, phi0):
        x = 2*pi*sin(theta0)*cos(phi0)
        y = 2*pi*sin(theta0)*sin(phi0)
        super().__init__(x, y)

        self.theta0 = theta0
        self.phi0 = phi0
        # self.magnitude = (2*pi)*sin(theta0)*sqrt((cos2(phi0) + sin2(phi0)))  # for validation


def sin2(x):
    return sin(x)**2


def cos2(x):
    return cos(x)**2


def xi(x):
    return 0  # flat surface


def bessel(order, value):
    # bessel functions of the first kind of real order and complex argument
    return jv(order, value)


def ihat(gamma, vec, xi0):
    h1 = vec.h1
    h2 = vec.h2
    # doubly-period cosine profile with period a
    return (-1j)**h1*bessel(h1, gamma*xi0/2)*(-1j)**h2*bessel(h2, gamma*xi0/2)


def alpha0(k):
    k2 = k.magnitude**2
    two_pi_squared = 4*pi*pi
    if k2 < two_pi_squared:
        return sqrt(two_pi_squared - k2)
    else:
        return 1j*(k2 - two_pi_squared)


def boundary_M(pvec, qvec):
    return 1  # Dirichlet


def boundary_N(pvec, qvec):
    return 1  # Dirichlet


def RHS(k, K, G):
    return -ihat(alpha0(k), G, xi0=xi0)*boundary_N(K, k)


def LHS(k, K, Kprime, G, Gprime):
    return ihat(-alpha0(Kprime), G-Gprime, xi0=xi0)*boundary_M(K, Kprime)


# print(bessel(1, 2))
# print(ihat(gamma=1, G=LatticeSite()))


# properties of incident wave vector k
theta0 = 0.5  # polar angle of incidence (vary this?)
phi0 = 0  # azimuthal angle of incidence (0 or 45 in the article)
k = IncidentWave(theta0, phi0)
print(k.magnitude)

G = LatticeSite(0, 0)
print(G.magnitude)
Gprime = LatticeSite(1, 2)
print((G + Gprime).h1)
print((G - Gprime).h2)

K = k + G
print(type(K))
print(K.magnitude)
Kprime = k + Gprime

print("alpha0(k):")
print(alpha0(k))
print("alpha0(Kprime):")
print(alpha0(Kprime))

print(RHS(k, K, G))
print(LHS(k, K, Kprime, G, Gprime))

H = 3
Hs = range(-H, H+1)  # +1 for endpoint
n = len(Hs)
N = n**2
A = np.zeros([N, N], dtype=np.complex_)
b = np.zeros(N, dtype=np.complex_)
print(n)
print(np.shape(A))

for i in range(N):
    # vary K and G in the outer loop
    h1 = Hs[i//n]
    h2 = Hs[i % n]
    # print("h1 = %d, h2 = %d" % (h1, h2))
    G = LatticeSite(h1, h2)
    K = k + G

    b[i] = RHS(k, K, G)
    for j in range(N):
        # vary Kprime and Gprime in the inner loop
        h1 = Hs[j//n]
        h2 = Hs[j % n]
        # print("h'1 = %d, h'2 = %d" % (h1, h2))
        Gprime = LatticeSite(h1, h2)
        Kprime = k + G

        A[i, j] = LHS(k, K, Kprime, G, Gprime)

print(A)
print(b)

x = np.linalg.solve(A, b)
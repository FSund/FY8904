import numpy as np
from scipy.special import jv
from math import sqrt, pi, sin, cos
from time import sleep
import numba

# np.seterr(all='raise')


class X3Vector:
    # a general 2d vector (in the x3-plane)
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self._abs = None
        self._abs2 = None

    def abs2(self):
        if not self._abs2:
            self._abs2 = self.x**2 + self.y**2
        return self._abs2

    def abs(self):
        if not self._abs:
            if not self._abs2:
                self.abs2()  # calculates self._abs2
            self._abs = sqrt(self._abs2)
        return self._abs

    def __add__(self, other):
        return X3Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return X3Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return self.x*other.x + self.y*other.y


class LatticeSite(X3Vector):
    '''
    reciprocal lattice site (mostly G or G')
    '''
    a = None  # static, lattice dimensions

    def __init__(self, h1=0, h2=0):
        if not self.a:
            raise RuntimeError(
                "LatticeSite has not been initialized! "
                "Please set a value for a before creating any instances.")

        L = 2*pi/self.a
        x = h1*L
        y = h2*L
        super().__init__(x, y)

        self.h1 = h1
        self.h2 = h2

    def __add__(self, other):
        return LatticeSite(self.h1 + other.h1, self.h2 + other.h2)

    def __sub__(self, other):
        return LatticeSite(self.h1 - other.h1, self.h2 - other.h2)


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


@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

# def xi(x):
#     return 0  # flat surface


def bessel(order, argument):
    '''
    Bessel function of the first kind of real order and complex argument
    '''
    return jv(order, argument)


def neumann(pvec, qvec):
    return (4*pi*pi - pvec*qvec)/(alpha0(qvec))


def alpha0(k):
    k2 = k.abs2()
    two_pi_squared = 4*pi*pi
    if k2 < two_pi_squared:  # propagating
        return sqrt(two_pi_squared - k2)
    else:  # evanescent
        return 1j*sqrt(k2 - two_pi_squared)


class Simulator:
    def __init__(self, a, xi0, dirichlet=True):
        LatticeSite.a = a
        self.xi0 = xi0
        self.dirichlet = dirichlet

    def boundary_M(self, pvec, qvec):
        if self.dirichlet:
            return 1  # Dirichlet
        else:
            return neumann(pvec, qvec)

    def boundary_N(self, pvec, kvec):
        if self.dirichlet:
            return 1  # Dirichlet
        else:
            return -neumann(pvec, kvec)

    def ihat(self, gamma, G):
        '''
        doubly-period cosine profile with period a
        '''
        h1 = G.h1
        h2 = G.h2

        # from assignment (I think this is the correct form)
        # value = (-1j)**h1*bessel(h1, gamma*xi0/2)*(-1j)**h2*bessel(h2, gamma*xi0/2)
        value = (-1j)**(h1 + h2)*bessel(h1, gamma*self.xi0/2)*bessel(h2, gamma*self.xi0/2)

        # from paper (notice change from -i to -1)
        # value = (-1)**(h1 + h2)*bessel(h1, gamma*xi0/2)*bessel(h2, gamma*xi0/2)

        return value

    def RHS(self, k, K, G):
        return -self.ihat(alpha0(k), G)*self.boundary_N(K, k)

    def LHS(self, k, K, Kprime, G, Gprime):
        return self.ihat(-alpha0(Kprime), G-Gprime)*self.boundary_M(K, Kprime)

    def simulate(self, theta0, phi0, H):
        # incident wave vector k
        k = IncidentWave(theta0, phi0)
        # print("theta0 = {}".format(theta0))
        # print("phi0 = {}".format(phi0))
        # print("|k| = {}".format(k.magnitude))

        # simulation settings
        Hs = range(-H, H+1)  # +1 to include endpoint
        n = len(Hs)
        N = n**2
        # print("n = %d" % n)
        # print("N = %d" % N)
        # if self.dirichlet:
        #     print("Dirichlet")
        # else:
        #     print("Neumann")

        # linear equation
        A = np.zeros([N, N], dtype=np.complex_)
        b = np.zeros(N, dtype=np.complex_)

        self.h = []
        # n_propagating = 0
        # Gmax = 0
        # Gmin = np.inf
        for i in range(N):
            # vary K and G in the outer loop
            h1 = Hs[i//n]
            h2 = Hs[i % n]

            G = LatticeSite(h1, h2)
            K = k + G
            # if G.magnitude > Gmax:
            #     Gmax = G.magnitude
            # if G.magnitude < Gmin:
            #     Gmin = G.magnitude

            b[i] = self.RHS(k, K, G)
            for j in range(N):
                # vary Kprime and Gprime in the inner loop
                # h1 = Hs[j//n]
                # h2 = Hs[j % n]
                h1, h2 = [Hs[k] for k in divmod(j, n)]

                Gprime = LatticeSite(h1, h2)
                Kprime = k + Gprime

                A[i, j] = self.LHS(k, K, Kprime, G, Gprime)

                # if Kprime.magnitude < 2*pi:
                #     n_propagating += 1
                if i == 0:  # only do this the first time
                    self.h.append([h1, h2])

        self.h = np.array(self.h)
        # print("%% propagating (roughly) = {}".format(n_propagating/N**2*100))
        # print("Gmax = {}".format(Gmax))
        # print("Gmax = {} omega/c".format(Gmax/(2*pi)))
        # print("Gmin = {}".format(Gmin))

        x = np.linalg.solve(A, b)
        self.r = x
        self.A = A
        self.b = b

        # find conservation of energy (eq. 43)
        conservation = complex()
        if True:
            for i in range(N):
                h1 = Hs[i//n]
                h2 = Hs[i % n]

                G = LatticeSite(h1, h2)
                K = k + G
                if K.abs2() < 4*pi*pi:
                    r2 = abs2(x[i])
                    alpha0_k = alpha0(k)
                    if (abs2(alpha0_k) > 0):
                        e = alpha0(K)/alpha0_k*r2
                        conservation += e
        else:
            conservation = None

        # calculate reflectivity
        if False:
            for i in range(N):
                h1 = Hs[i//n]
                h2 = Hs[i % n]
                G = LatticeSite(h1, h2)
                if G.abs() < 1e-10:
                    print("G({}, {}) = ({}, {})".format(h1, h2, G.x, G.y))
                    r2 = np.abs(x[i])**2
                    e = alpha0(k)/alpha0(k)*r2
                    print("Reflectivity R = {}".format(e))
                    print("(R = {})".format(r2))
        reflectivity = abs2(x[round(N/2)])

        return x, conservation, reflectivity

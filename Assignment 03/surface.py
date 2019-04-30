from scipy.special import jv


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
        super().__init__(a, xi0)
        self.rho_t = rho_t
        self.rho_b = rho_b

    def ihat(gamma, G):
        raise RuntimeError("truncatedCosine() not implemented yet")


class TruncatedCosine(Surface):
    def __init__(self, a, xi0, rho0):
        super().__init__(a, xi0)
        self.rho0 = rho0

    def ihat(self, gamma, G):
        raise RuntimeError("truncatedCosine() not implemented yet")

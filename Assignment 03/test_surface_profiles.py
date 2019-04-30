from math import pi
from surface import DoubleCosine, TruncatedCone, TruncatedCosine
from simulator import alpha0, LatticeSite, X3Vector, IncidentWave, Simulator

a = 3.5  # 0.5 or 3.5 for task 4
H = 9
h2 = 0
phi0 = 0
xi0 = 0.1  # find optimal value for this in task 4
dirichlet = False

n_its = 1000
# theta = np.linspace(-pi/2, pi/2, n_its+2)
# theta = theta[1:-1]  # remove both endpoints
theta0 = pi/4

# sim = Simulator(a, xi0, dirichlet)
# r = sim.task4(theta0, H)

rho_b = (a/2)*0.8
rho_t = rho_b*0.8
# profile = TruncatedCone(a, xi0, rho_t, rho_b)
rho0 = (a/2)*0.8
profile = TruncatedCosine(a, xi0, rho0)

LatticeSite.a = a
G = LatticeSite(0, 0)
K = IncidentWave(theta0=pi/4, phi0=0) + G
gamma = alpha0(K)
print(profile.ihat(gamma, G))

sim = Simulator(a, xi0, surface=profile)
x, U, e_m, m = sim.task4(theta0, H)
print(U)

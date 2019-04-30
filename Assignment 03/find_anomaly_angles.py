from math import pi, atan, sqrt, asin, sin, cos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from simulator import IncidentWave, LatticeSite, X3Vector


def find_anomaly_angle(k_hat, h1, h2):
    b = X3Vector(h1/a, h2/a)
    term = (k_hat*b)**2 - b*b + 1
    if term < 0:
        return np.nan, np.nan
    else:
        term = sqrt(term)

    sintheta0_plus = -(k_hat*b) + term
    sintheta0_minus = -(k_hat*b) - term

    if np.abs(sintheta0_plus) <= 1:
        theta0_plus = asin(sintheta0_plus)
    else:
        theta0_plus = np.nan

    if np.abs(sintheta0_minus) <= 1:
        theta0_minus = asin(sintheta0_minus)
    else:
        theta0_minus = np.nan

    return theta0_plus, theta0_minus


def find_anomaly_angle2(phi0, h1, h2, a):
    first = -2/a*(h1*cos(phi0) + h1*sin(phi0))
    second = (1/a*(h1*cos(phi0) + h2*sin(phi0)))**2 - (1/a**2*(h1**2 + h2**2) -1)
    if second < 0:
        return np.nan, np.nan
    plus = first + sqrt(second)
    minus = first - sqrt(second)
    if abs(plus) > 1:
        plus = np.nan
    else:
        plus = asin(plus)
    if abs(minus) > 1:
        minus = np.nan
    else:
        minus = asin(minus)

    return plus, minus


def find_anomaly_angle3(phi0, h1, h2, a):
    a = 1
    b = 4/a*(h1*cos(phi0) + h2*sin(phi0))
    c = 1/(a**2)*(h1**2 + h2**2) - 1
    root = b**2 - 4*a*c
    if root < 0:
        return np.nan, np.nan

    plus = (-b + sqrt(root))/(2*a)
    minus = (-b - sqrt(root))/(2*a)

    if abs(plus) > 1:
        plus = np.nan
    else:
        plus = asin(plus)

    if abs(minus) > 1:
        minus = np.nan
    else:
        minus = asin(minus)

    return plus, minus


def find_anomaly_angle4(phi0, h1, h2, a):
    k_hat = X3Vector(cos(phi0), sin(phi0))
    G = X3Vector(h1/a, h2/a)

    a = 1
    b = 2*(k_hat*G)
    c = G*G - 1

    root = b**2 - 4*a*c
    if root < 0:
        return np.nan, np.nan

    plus = (-b + sqrt(root))/(2*a)
    minus = (-b - sqrt(root))/(2*a)

    if abs(plus) > 1:
        plus = np.nan
    else:
        plus = asin(plus)

    if abs(minus) > 1:
        minus = np.nan
    else:
        minus = asin(minus)

    return plus, minus


phi0 = pi/4
k_hat = X3Vector(cos(phi0), sin(phi0))
H = 9
a = 3.5
LatticeSite.a = a

Hs = range(-H, H+1)
N = len(Hs)**2
theta0 = np.zeros(N)
theta1 = np.zeros(N)
i = 0
for h1 in Hs:
    for h2 in Hs:
        # theta0[i], theta1[i] = find_anomaly_angle(k_hat, h1, h2)
        # theta0[i], theta1[i] = find_anomaly_angle2(phi0, h1, h2, a)
        # theta0[i], theta1[i] = find_anomaly_angle3(phi0, h1, h2, a)
        theta0[i], theta1[i] = find_anomaly_angle4(phi0, h1, h2, a)
        i += 1

# print(theta0)
# print(theta1)

theta0 = theta0[~np.isnan(theta0)]
theta0 = theta0/(2*pi)*360
theta0 = np.sort(theta0)

theta1 = theta1[~np.isnan(theta1)]
theta1 = theta1/(2*pi)*360
theta1 = np.sort(theta1)

# print(theta0)
# print(theta1)

# fig, ax = plt.subplots()
# ax.plot(theta0, 'o')
# ax.plot(theta1, 'o')
# ax.legend()

fig, ax = plt.subplots()
for theta in [theta0, theta1]:
    for x in theta:
        ax.axvline(x)
ax.set_xlim([0, 90])
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(2))

plt.show()

theta0 = np.concatenate((theta0, theta1))
theta0 = theta0[theta0>0]
# np.save("anomaly_angles_degrees", theta0)
np.save("anomaly_angles_degrees_45", theta0)

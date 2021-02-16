import sympy
from sympy import Function, dsolve, Eq, Derivative, sin, cos, symbols, sqrt, pi, Integer, log, I, re, lambdify, simplify
from sympy.utilities.lambdify import lambdastr

import math
from sympy.abc import x

import numpy as np
theta = np.linspace(0, 2*np.pi, 720)

def dxdt(theta, t = 0.5, alpha = np.sqrt(3) / 2):
    return -alpha*(np.cos(theta) + np.sin(theta) - (t**2 + t)*(2*alpha**2 - 1)*np.sin(theta)**2*np.cos(theta)*np.sqrt(np.tan(theta)**4 + 1))

import matplotlib.pyplot as plt

for t in np.linspace(0., 1., 100):
    plt.plot(theta, np.abs(dxdt(theta, t)))
    plt.show()

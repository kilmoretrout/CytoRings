import numpy as np
import matplotlib.pyplot as plt
from func import solve_harmonic_series, compute_harmonic
from scipy.interpolate import interp1d

# get a perturbed version of a cone
t = np.linspace(0., 1., 100)
r = 1.01 - t

theta = np.linspace(0., 2*np.pi, 72)

X = []
X_theta = []

for k in range(len(t)):
    x = []
    y = []
    
    x_ = r[k]*np.cos(theta)
    y_ = r[k]*np.sin(theta)

    for j in range(100):
        x.extend(list(x_ + np.random.normal(0, 0.05, 72)))
        y.extend(list(y_ + np.random.normal(0, 0.05, 72)))

    X_ = np.vstack([np.array(x, dtype = np.float32), np.array(y, dtype = np.float32)]).T
    theta_ = np.arctan2(X_[:,1], X_[:,0])
    
    X.append(X_)
    X_theta.append(theta_)

print('created arrays...')

co = []
xy = []
xy_real = []

for k in range(len(X)):
    xy_ = X[k]
    theta_ = X_theta[k]

    ix = list(np.random.choice(range(len(xy_)), 144, replace = False))

    x_co = solve_harmonic_series(xy_[ix,0], theta_[ix])
    y_co = solve_harmonic_series(xy_[ix,1], theta_[ix])

    xy_real.append(xy_[ix])
    co.append(np.hstack([x_co, y_co]))

    x_new = compute_harmonic(theta, x_co)
    y_new = compute_harmonic(theta, y_co)

    xy.append(np.array([x_new, y_new]).T)

co = np.array(co).T

f = interp1d(t, co)








    

    



    

    

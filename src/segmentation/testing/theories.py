import os
import sys
import itertools

import os
import logging, argparse
import itertools

import platform
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.integrate import odeint

# define the differential equation for case (1)
def model(t, phi, alpha0, beta0):
    dphi_dt = -np.sin(2*phi)*(alpha0**2 * (1-t) + beta0**2 * (t - 1))
    dphi_dt = dphi_dt / (2 * (1 - t**2) * (alpha0**2 * np.sin(phi)**2 + beta0**2 * np.cos(phi)**2))

    return dphi_dt

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--experiment", default = "1")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    # our usual segmentation
    theta = np.linspace(0., 2*np.pi, 72)

    if int(args.experiment) == 1:
        # case (1)
        # 2 terms (x = cos, y = sin, no centroid or centroid = (0, 0) in x, y)
        t = np.linspace(0., 0.85, 100)

        alpha0 = np.random.uniform(1., 10.)
        beta0 = np.random.uniform(1., 10.)

        alpha = alpha0 - alpha0*t
        beta = beta0 - beta0*t

        logging.info('alpha: {:.4f}, beta: {:.4f}'.format(alpha0, beta0))

        X = []

        for theta_ in theta:
            sol = odeint(model, theta_, t, args = (alpha0, beta0))

            X.append(sol[:,0])


        theta = np.array(X)
        plt.imshow(theta)
        plt.show()

        X = []
        X_dot = []

        for i in range(len(t)):
            X_ = []
            X_dot_ = []

            for j in range(theta.shape[0]):
                X_.append([alpha[i]*np.cos(theta[j,i]), beta[i]*np.sin(theta[j,i]), t[i]])
                X_dot_.append(-alpha0*np.cos(theta[j,i]) * -alpha[i]*np.sin(theta[j,i]) + -beta0*np.sin(theta[j,i])*beta[i]*np.cos(theta[j,i]))


            X_dot.append(np.array(X_dot_))
            X.append(np.array(X_))

        X_dot = np.array(X_dot)
        X = np.array(X)

        fig = plt.figure()
        ax = fig.add_subplot(211, projection='3d')

        for ix in range(len(theta)):
            ax.plot(X[:, ix, 0], X[:, ix, 1], X[:, ix, 2], color='k')

        ax1 = fig.add_subplot(212)
        im = ax1.imshow(X_dot)
        fig.colorbar(im, ax1)

        plt.show()

        """
        quant1 = alpha0*(np.sin(theta)**2) + beta0*(np.cos(theta)**2)
        quant4 = -np.sin(2*theta) / (2*(alpha0**2 * np.sin(theta)**2 + beta0**2 * np.cos(theta)**2))

        plt.plot(quant4)
        plt.show()

        X = []
        X_dot = []

        X_quant2 = []
        X_quant3 = []

        for ix in range(len(t)):
            X_ = np.array([alpha[ix]*np.cos(theta), beta[ix]*np.sin(theta), np.ones(theta.shape)*t[ix]]).T
            X.append(X_)

            X_dot.append(np.multiply(-alpha0*np.cos(theta), -alpha[ix]*np.sin(theta)) + np.multiply(-beta0*np.sin(theta),
                                                            beta[ix]*np.cos(theta)))

            X_quant2.append(-1*(alpha0**2*(1 - t[ix]) + beta0**2*(t[ix] - 1))*np.sin(2*theta) / 2.)
            X_quant3.append((1 - t[ix])**2 * (alpha0**2 * np.sin(theta)**2 + beta0**2 * np.cos(theta)**2))

        X_quant2 = np.array(X_quant2)
        X_quant3 = np.array(X_quant3)

        fig, axes = plt.subplots(ncols = 2)
        axes[0].imshow(X_quant2)
        axes[1].imshow(X_quant3)
        plt.show()

        plt.imshow((X_quant2 / X_quant3)[:50,:])
        plt.colorbar()
        plt.show()

        X_dot = np.array(X_dot)
        X = np.array(X)

        fig = plt.figure()
        ax = fig.add_subplot(211, projection='3d')

        for ix in range(len(theta)):
            ax.plot(X[:,ix,0], X[:,ix,1], X[:,ix,2], color = 'k')

        ax1 = fig.add_subplot(212)
        ax1.imshow(X_dot)
        plt.show()
        """

if __name__ == '__main__':
    main()
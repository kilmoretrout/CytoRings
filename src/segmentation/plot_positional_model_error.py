import os
import sys
import itertools

import os
import logging, argparse
import itertools

import platform
import numpy as np
import h5py
import matplotlib.pyplot as plt

from sklearn.neighbors import KDTree
from scipy.signal import savgol_filter

from scipy import optimize
import cv2

import tifffile
import pandas as pd

from scipy.optimize import fmin, fminbound, fmin_l_bfgs_b
from scipy.integrate import quad

import random
from pyefd import elliptic_fourier_descriptors, normalize_efd, reconstruct_contour, calculate_dc_coefficients
from statsmodels.graphics import tsaplots

from scipy.stats import anderson

def co_error(r, c, cox, coy):
    correct_x = np.array([c[0], r, 0., 0., 0., 0., 0.])
    correct_y = np.array([c[1], 0., 0., 0., r, 0., 0.])

    error_x = cox - correct_x
    error_y = coy - correct_y

    return error_x, error_y

def co_errors(radii, centroids, co):
    error_x = []
    error_y = []
    
    for k in range(len(radii)):
        r = radii[k]
        c = centroids[k]

        cox = co[k,0,:]
        coy = co[k,1,:]

        ex, ey = co_error(r, c, cox, coy)

        error_x.append(ex)
        error_y.append(ey)

    error_x = np.array(error_x)
    error_y = np.array(error_y)

    return error_x, error_y

def correct(radii, centroids):
    cx = []
    cy = []

    for k in range(len(radii)):
        r = radii[k]
        c = centroids[k]

        cx.append(np.array([c[0], r, 0., 0., 0., 0., 0.]))
        cy.append(np.array([c[1], 0., 0., 0., r, 0., 0.]))

    return np.array(cx), np.array(cy)
        

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--ofile", default = "noise_metrics.csv")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.mkdir(args.odir)
            logging.debug('root: made output directory {0}'.format(args.odir))
        else:
            os.system('rm -rf {0}'.format(os.path.join(args.odir, '*')))

    return args

def compute_harmonic(theta, co, N = 3):
    A = []

    for t in theta:
        _ = []
        _.append(1.)
        
        for ix in range(1, N + 1):
            _.append(np.cos(t*ix))

        for ix in range(1, N + 1):
            _.append(np.sin(t*ix))


        #print _
        A.append(np.array(_))

    A = np.array(A)

    return A.dot(co)

def differentiator(f, N = 7, m = 2):
    if N == 5 and m == 2:
        ret = []
        for k in range(2, len(f) - 2):
            ret.append((2*(f[k + 1] - f[k - 1]) + f[k + 2] - f[k - 2]) / 8)

        return ret

    if N == 7 and m == 2:
        ret = []
        for k in range(3, len(f) - 3):
            ret.append((5 * (f[k + 1] - f[k - 1]) + 4 * (f[k + 2] - f[k - 2]) + (f[k + 3] - f[k - 3])) / (32))

        return ret

    if N == 11 and m == 2:
        ret = []
        for k in range(5, len(f) - 5):
            ret.append((42 * (f[k + 1] - f[k - 1]) + 48 * (f[k + 2] - f[k - 2]) + 27 * (f[k + 3] - f[k - 3]) + 8 * (
                        f[k + 4] - f[k - 4]) + (f[k + 5] - f[k - 5])) / (512))

        return ret
    if N == 7 and m == 4:
        ret = []
        for k in range(3, len(f) - 3):
            ret.append((39*(f[k + 1] - f[k - 1]) + 12*(f[k + 2] - f[k - 2]) - 5*(f[k + 3] - f[k - 3])) / (96))

        return ret
    elif N == 11 and m == 4:
        ret = []
        for k in range(5, len(f) - 5):
            ret.append((322*(f[k + 1] - f[k - 1]) + 256*(f[k + 2] - f[k - 2]) + 39*(f[k + 3] - f[k - 3]) - 32*(f[k + 4] - f[k - 4]) - 11*(f[k + 5] - f[k-5])) / (1536))

        return ret

def analytic_speed(radii, centroids, randomize = False):
    theta = np.linspace(0., 2*np.pi, 72)

    d = []

    for k in range(len(radii) - 1):

        x0 = radii[k] * np.cos(theta) + centroids[k][0]
        y0 = radii[k] * np.sin(theta) + centroids[k][1]

        x1 = radii[k + 1] * np.cos(theta) + centroids[k + 1][0]
        y1 = radii[k + 1] * np.sin(theta) + centroids[k+ 1][1]

        d.append(np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2))

    return np.array(d).T / 2.7

def analytic_position_with_noise(radii, centroids):
    theta = np.linspace(0., 2 * np.pi, 72)

    xy = []

    for k in range(len(radii)):
        x0 = radii[k] * np.cos(theta) + centroids[k][0] + np.random.normal(0, 1, (len(theta), ))
        y0 = radii[k] * np.sin(theta) + centroids[k][1] + np.random.normal(0, 1, (len(theta), ))

        xy.append(np.array([x0, y0]).T)


    return np.array(xy)


import math
import scipy.stats as st  # for pvalue


# Finds runs in data: counts and creates a list of them
# TODO: There has to be a more pythonic way to do this...
def getRuns(l):
    runsList = []
    tmpList = []
    for i in l:
        if len(tmpList) == 0:
            tmpList.append(i)
        elif i == tmpList[len(tmpList) - 1]:
            tmpList.append(i)
        elif i != tmpList[len(tmpList) - 1]:
            runsList.append(tmpList)
            tmpList = [i]
    runsList.append(tmpList)

    return len(runsList), runsList


# define the WW runs test described above
def WW_runs_test(R, n1, n2, n):
    # compute the standard error of R if the null (random) is true
    seR = math.sqrt(((2 * n1 * n2) * (2 * n1 * n2 - n)) / ((n ** 2) * (n - 1)))

    # compute the expected value of R if the null is true
    muR = ((2 * n1 * n2) / n) + 1

    # test statistic: R vs muR
    z = (R - muR) / seR

    return z


def solve_harmonic_series(x, N=3):
    theta = np.linspace(0., 2 * np.pi, len(x))

    A = []

    for t in theta:
        _ = []
        _.append(1.)

        for ix in range(1, N + 1):
            _.append(np.cos(t * ix))

        for ix in range(1, N + 1):
            _.append(np.sin(t * ix))

        # print _
        A.append(np.array(_))

    A = np.array(A)

    B = x.T

    co = np.linalg.lstsq(A, B)[0]

    return A.dot(co), co

import random
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import anderson

f_result = dict()
freqx = np.fft.fftshift(np.fft.fftfreq(100, 1))  # q(n, d=1.0)
freqy = np.fft.fftshift(np.fft.fftfreq(100, 1))
fX, fY = np.meshgrid(freqx, freqy)

from scipy.ndimage import gaussian_filter
odir = 'images'

def append(result, data, speed_real, noise_var, param, key, smooth = False):
    global f_result

    if not smooth:
        names = ['differentiator_{0}_{1}', 'diff', 'grad']
    else:
        names = ['smooth_differentiator_{0}_{1}', 'smooth_diff', 'smooth_grad']

    N, m = param

    for k in range(len(data)):
        datum = data[k]

        noise = (np.mean(speed_real) - datum[:, 5:-8])
        result['method'].append(names[k].format(N, m))
        result['anderson_statistic'].append(anderson(noise.flatten())[0])
        L = np.zeros(noise.shape)
        L[np.where(noise > 0)] = 1

        """
        gaussian = np.random.normal(np.mean(noise), np.std(noise), size = noise.shape)

        X, Y = np.meshgrid(np.array(range(noise.shape[1]))*2.7, range(72))

        fig, axes = plt.subplots(nrows = 3)

        im = axes[0].pcolormesh(X, Y, noise / np.mean(speed_real) * 100)
        fig.colorbar(im, ax = axes[0])
        im = axes[1].pcolormesh(X, Y, gaussian_filter(noise, sigma = 3) / np.mean(speed_real) * 100)
        fig.colorbar(im, ax=axes[1])
        im = axes[2].pcolormesh(X, Y, gaussian_filter(noise, sigma = 5) / np.mean(speed_real) * 100)
        fig.colorbar(im, ax=axes[2])

        plt.savefig(os.path.join(odir, '{1}_{0}.png'.format(names[k].format(N,m), key)))
        plt.close()
        """
        ps = np.fft.fftshift(np.fft.fft2(noise, s = (100, 100)))

        freqx = np.fft.fftshift(np.fft.fftfreq(100, 1))  # q(n, d=1.0)
        freqy = np.fft.fftshift(np.fft.fftfreq(100, 1))

        if not names[k].format(N,m) in f_result.keys():
            f_result[names[k].format(N,m)] = [np.abs(ps)]
        else:
            f_result[names[k].format(N, m)].append(np.abs(ps))

        p = []

        L = list(L.flatten())

        # Gather info
        numRuns, listOfRuns = getRuns(L)  # Grab streaks in the data

        # Define parameters
        R = numRuns  # number of runs
        n1 = sum(L)  # number of 1's
        n2 = len(L) - n1  # number of 0's
        n = n1 + n2  # should equal len(L)

        # Run the test
        ww_z = WW_runs_test(R, n1, n2, n)

        # test the pvalue
        # p_values_one = st.norm.sf(abs(ww_z))  # one-sided
        p = st.norm.sf(abs(ww_z)) * 2  # twosided

        result['runs_p'].append(p)
        result['mean'].append(np.mean(noise.flatten()))
        result['var'].append(np.var(noise.flatten()))
        result['noise_var'].append(noise_var)
        result['ns'].append(np.mean(np.abs(noise / np.mean(speed_real))))

    return result

def main():
    args = parse_args()

    #ifile = h5py.File(args.ifile, 'r')

    #keys = list(ifile.keys())[:100]
    #random.shuffle(keys)

    keys = ['{0:03d}'.format(k) for k in range(100)]

    names = ['alpha0', 'alpha1', 'alpha2', 'alpha3', 'beta1', 'beta2', 'beta3']
    theta = np.linspace(0., 2*np.pi, 72)

    params = [(5,2), (7,2), (11,2), (7,4), (11,4)]
    result = dict()
    result['method'] = list()
    result['noise_var'] = list()
    result['anderson_statistic'] = list()
    result['mean'] = list()
    result['var'] = list()
    result['runs_p'] = list()
    result['ns'] = list()

    for key in keys:
        if args.idir != "None":
            X = np.load(os.path.join(os.path.join(args.idir, 'info'), '{0}.npz'.format(key)), allow_pickle=True)
            radii = X['radii']
            centroids = X['centroids']
            noise_var = X['noise_var']
        for param in params:
            #xy = ifile[key]['xy']
            print(key)
            xy_noisy = analytic_position_with_noise(radii, centroids)

            print(xy_noisy.shape)

            new_poly = []

            cos_x = []
            cos_y = []

            for x in xy_noisy:
                xs, cox = solve_harmonic_series(x[:, 0])
                ys, coy = solve_harmonic_series(x[:, 1])

                cos_x.append(cox)
                cos_y.append(coy)

                xy_smooth = np.array([xs, ys]).T

                new_poly.append(xy_smooth)

            new_poly = np.array(new_poly, dtype=np.float32)

            dx = []
            dy = []

            dx_diff = []
            dy_diff = []

            dx_gradient = []
            dy_gradient = []

            N, m = param

            for k in range(72):
                dx.append(differentiator(xy_noisy[:,k,0], N, m))
                dy.append(differentiator(xy_noisy[:,k,1], N, m))

                dx_diff.append(np.diff(xy_noisy[:,k,0]))
                dy_diff.append(np.diff(xy_noisy[:,k,1]))

                dx_gradient.append(np.gradient(xy_noisy[:,k,0]))
                dy_gradient.append(np.gradient(xy_noisy[:,k,1]))

            dx = np.vstack(dx)
            dy = np.vstack(dy)

            dx_diff = np.vstack(dx_diff)
            dy_diff = np.vstack(dy_diff)

            dx_gradient = np.vstack(dx_gradient)
            dy_gradient = np.vstack(dy_gradient)

            speed = np.sqrt(dx**2 + dy**2) / 2.7
            speed_diff = np.sqrt(dx_diff**2 + dy_diff**2) / 2.7
            speed_gradient = np.sqrt(dx_gradient**2 + dy_gradient**2) / 2.7

            if args.idir != "None":
                speed_real = analytic_speed(radii, centroids)
                if np.var(speed_real) < 0.000001:
                    speed_real[::] = np.mean(speed_real)
            else:
                speed_real = np.ones(speed_diff.shape)*np.median(speed)

            result = append(result, [speed, speed_diff, speed_gradient], speed_real, noise_var, param, key)

            dx = []
            dy = []

            dx_diff = []
            dy_diff = []

            dx_gradient = []
            dy_gradient = []

            for i in range(72):
                dx.append(differentiator(new_poly[:, i, 0], N, m))
                dy.append(differentiator(new_poly[:, i, 1], N, m))

                dx_diff.append(np.diff(new_poly[:, i, 0]))
                dy_diff.append(np.diff(new_poly[:, i, 1]))

                dx_gradient.append(np.gradient(new_poly[:, i, 0]))
                dy_gradient.append(np.gradient(new_poly[:, i, 1]))

            dx = np.vstack(dx)
            dy = np.vstack(dy)

            dx_diff = np.vstack(dx_diff)
            dy_diff = np.vstack(dy_diff)

            dx_gradient = np.vstack(dx_gradient)
            dy_gradient = np.vstack(dy_gradient)

            speed2 = np.sqrt(dx ** 2 + dy ** 2) / 2.7
            speed_diff = np.sqrt(dx_diff ** 2 + dy_diff ** 2) / 2.7
            speed_gradient = np.sqrt(dx_gradient ** 2 + dy_gradient ** 2) / 2.7

            result = append(result, [speed2, speed_diff, speed_gradient], speed_real, noise_var, param, key, smooth = True)


    df = pd.DataFrame(result)
    df.to_csv(args.ofile, index = False)
    df_ = df.set_index('method')

    fig, axes = plt.subplots(nrows = 2, ncols = 7)
    methods = ['differentiator_{0}_{1}'.format(u[0], u[1]) for u in params] + ['diff', 'grad']
    methods_smooth = ['smooth_differentiator_{0}_{1}'.format(u[0], u[1]) for u in params] + ['smooth_diff', 'smooth_grad']

    print(methods)

    for k in range(len(methods)):
        im = axes[0, k].pcolormesh(fX, fY, np.mean(np.array(f_result[methods[k]]), axis = 0))

        axes[0,k].set_title(methods[k])

    for k in range(len(methods_smooth)):
        im = axes[1, k].pcolormesh(fX, fY, np.mean(np.array(f_result[methods_smooth[k]]), axis=0))

        axes[1, k].set_title(methods_smooth[k].replace('smooth_', ''))

    plt.tight_layout()
    plt.show()



"""         if args.idir != "None":
                speed_real = analytic_speed(radii, centroids)
                if np.var(speed_real) < 0.000001:
                    speed_real[::] = np.mean(speed_real)
            else:
                speed_real = np.ones(speed_diff.shape)*np.median(speed)

            qqplot((speed[:,10:] - np.mean(speed_real)).flatten(), line='s')
            plt.show()

            plt.hist((speed[:,10:] - np.mean(speed_real)).flatten(), bins = 25)
            plt.show()


            fig, axes = plt.subplots(nrows = 4, ncols = 3)
            im = axes[0,0].imshow(speed)
            fig.colorbar(im, ax = axes[0,0])
            im = axes[0,1].imshow(speed_diff)
            fig.colorbar(im, ax=axes[0,1])
            im = axes[0,2].imshow(speed_gradient)
            fig.colorbar(im, ax=axes[0,2])

            k = np.random.choice(range(72))

            print(k)
            axes[1, 0].plot(speed[k, :])
            axes[1, 1].plot(speed_diff[k, :])
            axes[1, 2].plot(speed_gradient[k, :])

            axes[1, 0].plot(speed_real[k, :])
            axes[1, 1].plot(speed_real[k, :])
            axes[1, 2].plot(speed_real[k, :])"""
        
        
        


        
            

if __name__ == '__main__':
    main()
            

            

        


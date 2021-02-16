import os
import sys
import itertools

import os
import logging, argparse
import itertools

import platform
from func import PointCloud, differentiator
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
import numpy as np

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--shape", default = "cone")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    pc = PointCloud(Nk = 1, N = 72)

    if args.ifile != "None":
        pc.load_file(args.ifile)
    elif args.shape == "elliptical_cone":
        pc.elliptical_cone()
    elif args.shape == "cone":
        pc.cone(n_frames = 75, noise_sigma = 0)

    t = np.linspace(0., 1., 100)
    r = 125. - 100 * t
    y_displacement = np.array(range(len(t))) * (r[0] - r[1]) / 1.5

    co = np.zeros((100, 2, 3))
    co[:,0,0] = 0.
    co[:,0,1] = r

    co[:,1,0] = -y_displacement
    co[:,1,2] = r

    pc.specify_derivatives(t, co)

    for k in range(pc.xy.shape[0]):
        plt.plot(pc.xy[k,:,0], pc.xy[k,:,1], color = 'k')

    plt.show()

    pc.solve_phi()
    pc.plot()


















if __name__ == '__main__':
    main()
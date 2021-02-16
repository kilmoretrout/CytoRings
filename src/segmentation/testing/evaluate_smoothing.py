import os
import sys
import itertools

import os
import logging, argparse
import itertools

import platform
from func import PointCloud
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline
import numpy as np
import itertools

from mpi4py import MPI
import pandas as pd

# configure MPI
comm = MPI.COMM_WORLD

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--odir", default = "None")

    parser.add_argument("--shape", default = "cone")

    args = parser.parse_args()

    if comm.rank == 0:
        if args.odir != "None":
            if not os.path.exists(args.odir):
                os.mkdir(args.odir)
                logging.debug('root: made output directory {0}'.format(args.odir))
            else:
                os.system('rm -rf {0}'.format(os.path.join(args.odir, '*')))

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

flatten = lambda t: [item for sublist in t for item in sublist]

def main():
    args = parse_args()

    N = 1024

    # here we are going to make some cones with various noise levels
    # smooth with various smoothing factors with a univariate spline
    # and see what error in speed looks like
    noises = [0.001, 0.01, 0.05, 0.1, 0.5, 1., 2., 4., 16.]
    factors = list(np.linspace(0.1, N, 20))

    todo = list(itertools.product(noises, factors))

    speed_errors = np.zeros((72*len(noises), 100*len(factors)))
    speed_phi_errors = np.zeros((72*len(noises), 100*len(factors)))

    ret = dict()
    ret['noise'] = []
    ret['smoothing_factor'] = []
    ret['mae'] = []
    ret['var'] = []
    ret['mae_phi'] = []
    ret['var_phi'] = []
    ret['speed_error_phi'] = []
    ret['speed_error'] = []

    for ix in range(comm.rank, len(todo), comm.size):
        noise, factor = todo[ix]
        logging.info('working on {0} of {1}'.format(ix + 1, len(todo)))

        if args.shape == 'cone':
            pc = PointCloud(Nk = 1, N = N)
            pc.cone(noise, n_frames = 100)

        pc.solve_axial_slices(factor)
        pc.solve_phi()

        speed, speed_phi = pc.compute_speeds()

        speed_error = speed - pc.real_speed
        speed_phi_error = speed_phi - pc.real_speed

        mae = np.mean(np.abs(speed_error))
        var = np.var(speed_error)
        mae_phi = np.mean(np.abs(speed_phi_error))
        var_phi = np.var(speed_phi_error)

        ret['noise'].append(noise)
        ret['smoothing_factor'].append(factor)
        ret['mae'].append(mae)
        ret['var'].append(var)
        ret['mae_phi'].append(mae_phi)
        ret['var_phi'].append(var_phi)
        ret['speed_error'].append(speed_error)
        ret['speed_error_phi'].append(speed_phi_error)

        n_index = noises.index(noise)
        f_index = factors.index(factor)

        #speed_errors[72*n_index:72*(n_index + 1), 100*f_index:100*(f_index + 1)] = speed_error
        #speed_phi_errors[72 * n_index:72 * (n_index + 1), 100 * f_index:100 * (f_index + 1)] = speed_phi_error

    result = comm.gather(ret, root = 0)

    if comm.rank == 0:
        final = {}

        for k in result[0].keys():
            final[k] = np.array(flatten([d[k] for d in result]))

        ind = np.lexsort((final['noise'], final['smoothing_factor']))

        for k in final.keys():
            final[k] = final[k][ind]

        np.savez(os.path.join(args.odir, 'errors.npz'), noise = final['noise'], smoothing_factor = final['smoothing_factor'],
                 speed_error = final['speed_error'], speed_error_phi = final['speed_error_phi'])
        del final['speed_error']
        del final['speed_error_phi']

        df = pd.DataFrame(final)
        df.to_csv(os.path.join(args.odir, 'metrics.csv'), index = False)
        logging.info('0: wrote csv and npz to {0}'.format(args.odir))









if __name__ == '__main__':
    main()
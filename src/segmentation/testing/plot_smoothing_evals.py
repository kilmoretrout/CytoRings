import os
import sys
import itertools

import os
import logging, argparse
import itertools

import platform
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--odir", default = "None")

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

def main():
    args = parse_args()

    df = pd.read_csv(os.path.join(args.idir, 'metrics.csv'), index_col = False)
    X = np.load(os.path.join(args.idir, 'errors.npz'))

    noises = X['noise']
    factors = X['smoothing_factor']
    err = X['speed_error']
    err_phi = X['speed_error_phi']

    unique_noises = list(set(noises))
    unique_factors = list(set(factors))
    N_factors = len(unique_factors)

    mae = df['mae']
    var = df['var']
    mae_phi = df['mae_phi']
    var_phi = df['var_phi']

    speed_errors = np.zeros((err.shape[0] * err.shape[1], err.shape[0] * err.shape[2]))
    speed_phi_errors = np.zeros((err_phi.shape[0] * err_phi.shape[1], err_phi.shape[0] * err_phi.shape[2]))

    for k in range(len(err)):
        """
        fig, axes = plt.subplots(ncols = 2)

        im = axes[0].imshow(err[k])
        fig.colorbar(im, ax = axes[0])

        im = axes[1].imshow(err_phi[k])
        fig.colorbar(im, ax=axes[0])

        plt.show()
        """
        print((k // N_factors)*err.shape[1],((k // N_factors) + 1)*err.shape[1],
        (k % N_factors)*err.shape[2],((k % N_factors) + 1)*err.shape[2])

        speed_errors[(k // N_factors)*err.shape[1]:((k // N_factors) + 1)*err.shape[1],
        (k % N_factors)*err.shape[2]:((k % N_factors) + 1)*err.shape[2]] = err[k]
        speed_phi_errors[(k // N_factors)*err.shape[1]:((k // N_factors) + 1)*err.shape[1],
        (k % N_factors)*err.shape[2]:((k % N_factors) + 1)*err.shape[2]] = err_phi[k]

    norm = mpl.colors.Normalize(vmin=np.min(unique_noises), vmax=np.max(unique_noises))
    cmap = cm.hot

    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=5, nrows=4, figure=fig)

    ax_mae = fig.add_subplot(spec[0,:3])
    ax_var = fig.add_subplot(spec[1,:3])
    ax_mae_phi = fig.add_subplot(spec[2,:3])
    ax_var_phi = fig.add_subplot(spec[3,:3])

    for u in unique_noises:
        ix = list(np.where(noises == u)[0])
        ax_mae.plot(factors[ix], mae[ix], color = m.to_rgba(u))
        ax_var.plot(factors[ix], var[ix], color=m.to_rgba(u))
        ax_mae_phi.plot(factors[ix], mae_phi[ix], color=m.to_rgba(u))
        ax_var_phi.plot(factors[ix], var_phi[ix], color=m.to_rgba(u))

    ax_speed = fig.add_subplot(spec[:2,3:])
    ax_speed_phi = fig.add_subplot(spec[2:,3:])

    ax_speed.imshow(speed_errors[err.shape[1]:err.shape[1]*4,:err.shape[2]*4])
    ax_speed_phi.imshow(speed_phi_errors[err.shape[1]:err.shape[1]*4,:err.shape[2]*4])

    plt.show()

if __name__ == '__main__':
    main()
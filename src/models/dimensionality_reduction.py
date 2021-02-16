import os
import sys
import itertools

import os
import logging, argparse
import itertools

import platform
import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

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

    colors = ['r', 'b', 'g', 'k', 'y']

    X = []

    cases = os.listdir(args.idir)
    cases_files = dict()

    for case in cases:
        idir = os.path.join(args.idir, case)

        cases_files[case] = [os.path.join(idir, u) for u in os.listdir(idir)]

    case_ix = dict()

    ix = 0

    for case in cases:
        ifiles = cases_files[case]

        for ifile in ifiles:
            F = np.load(ifile)['F']

            X.append(F)

            if case not in case_ix.keys():
                case_ix[case] = []

            case_ix[case].extend(list(range(ix, ix + F.shape[0])))
            ix += F.shape[0]

    X = np.vstack(X)

    pca = PCA(2)
    Y = pca.fit_transform(X)

    cases = list(np.random.choice(cases, 5, replace=False))
    for case in cases:
        indices = case_ix[case]
        ix = cases.index(case)

        print(colors[ix])

        plt.scatter(Y[indices,0], Y[indices,1], c = colors[ix], label = case)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
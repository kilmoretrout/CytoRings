import os
import sys
import itertools

import os
import logging, argparse
import itertools

import platform
import h5py
from scipy.signal import hilbert
import numpy as np

def find_optimal_set(s, candidates, min_size = 1, max_size = 7, to_use = 15):
    coms = []

    for j in range(min_size, max_size + 1):
        coms.extend(list(itertools.combinations(range(to_use), j)))

    rs = []

    for com in coms:
        _ = np.zeros(s.shape)

        for j in com:
            _ += candidates[j]

        rs.append(np.sqrt(np.mean((_ - s)**2)) / (np.max(s) - np.min(s)))

    return coms[np.argmin(rs)], np.min(rs)

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default = "cyto_data_xy_v0.1.hdf5")
    parser.add_argument("--imfs", default = "IMFs_v0.1.hdf5")

    parser.add_argument("--key", default = "speed_phi")
    parser.add_argument("--trim", default = "5")

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

    ifile = h5py.File(args.ifile, 'r')
    imfs = h5py.File(args.imfs, 'r')

    cases = ['Control', 'Act-2', 'Pfn-1', 'Cyk-1', 'Ect-2', 'Plst-1', 'Nop-1']

    case_numbers = []
    for case in cases:
        numbers = list(imfs[case].keys())

        case_numbers.extend([(case, u) for u in numbers])

    for case, number in case_numbers:
        IMFs = np.array(imfs[case][number]['IMFs'])
        xyzs = np.array(imfs[case][number]['xyzs'])
        period = np.array(imfs[case][number]['period'])

        speed = np.array(ifile[case][number]['speed_phi'])[:,int(args.trim):-int(args.trim)]

        print(IMFs.shape, speed.shape)
        oset, rmse = find_optimal_set(speed, IMFs, to_use = IMFs.shape[0])

        print(rmse)
        print(case, number, len(oset))


if __name__ == '__main__':
    main()
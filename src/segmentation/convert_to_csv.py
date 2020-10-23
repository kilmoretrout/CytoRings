import os
import sys
import itertools

import os
import logging, argparse
import itertools
import h5py

import platform
import numpy as np
import pandas as pd

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--ofile", default = "None")
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

    keys = list(ifile.keys())

    for key in keys:
        result = dict()
        result['frame'] = []
        result['index'] = []
        result['x'] = []
        result['y'] = []

        poly = np.array(ifile[key]['xy'])

        for k in range(poly.shape[0]):
            result['index'].extend(list(range(72)))
            result['frame'].extend([k for u in range(72)])
            result['x'].extend(poly[k,:,0])
            result['y'].extend(poly[k,:,1])

        df = pd.DataFrame(result)
        df.to_csv(os.path.join(args.odir, '{0}.csv'.format(key)), index = False)


if __name__ == '__main__':
    main()
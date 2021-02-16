import os
import sys
import itertools

import os
import logging, argparse
import itertools

import platform

CMD_PATH = '/home/kilgoretrout/cytosim/bin/report'

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

    odir = os.path.abspath(args.odir)

    idirs = [os.path.join(args.idir, u) for u in os.listdir(args.idir)]
    idirs = [u for u in idirs if os.path.isdir(u)]

    for idir in idirs:
        print(idir)
        os.chdir(idir)
        os.system('{0} fiber:positon:anillin_filament > {1}'.format(CMD_PATH, os.path.join(odir, '{0}.csv'.format(idir.split('/')[-1]))))


if __name__ == '__main__':
    main()
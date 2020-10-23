import os
import sys
import itertools

import os
import logging, argparse
import itertools

import platform
import numpy as np
import matplotlib.pyplot as plt

from tifffile import imsave

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()

    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default = "None")
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

from scipy.ndimage import gaussian_filter

def main():
    args = parse_args()

    ifile = open(args.ifile, 'r')
    line = ifile.readline()
    line = ifile.readline()
    line = line.replace('\n', '')

    frame = 0

    t = [0]
    xy = []
    uv = []
    length = []

    xy_ = []
    uv_ = []
    length_ = []
    reading = False

    ix = 0

    while True:
        print(line)
        if '% time' in line:
            t.append(float(line.split(' ')[-1]))
            print(t[-1])
        elif ('endToEnd' in line) and ('posX' in line):
            reading = True

        line = ifile.readline()

        if '\n' in line:
            line = line.replace('\n', '')
        else:
            break

        # we expect this line to be coordinates, etc.
        if reading:
            if not '% end' in line:
                line = list(map(float, [u for u in line.split(' ') if u != '']))

                xy_.append([line[3], line[4]])
                uv_.append([line[5], line[6]])
                length_.append([line[2], line[7]])
            else:
                reading = False

                xy.append(np.array(xy_, dtype = np.float32))
                uv.append(np.array(uv_, dtype = np.float32))
                length.append(np.array(length_, dtype = np.float32))

                xy_ = []
                uv_ = []
                length_ = []

                if len(xy[-1]) > 0:
                    plt.scatter(xy[-1][:,0], xy[-1][:,1])
                    plt.xlim(-15, 15)
                    plt.ylim(-15, 15)
                    plt.savefig('frames/{0:06d}.png'.format(ix))
                    plt.close()

                    ix += 1

    for k in range(len(xy)):
        if len(xy[k]) > 0:
            vox = np.zeros((15000, 15000), dtype = np.int16)

            xbins = np.linspace(-15, 15, 15001)
            ybins = np.linspace(-15, 15, 15001)
            print(xbins[1] - xbins[0])

            x_ = np.digitize(xy[k][:,0], xbins)
            y_ = np.digitize(xy[k][:,1], ybins)

            for j in range(len(x_)):
                vox[15000 - y_[j], x_[j]] += 1

            imsave('{0:03d}.tiff'.format(k), vox)


if __name__ == '__main__':
    main()
import os
import sys
import itertools

import os
import logging, argparse
import itertools

import platform
import h5py
import numpy as np
import copy

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from subprocess import Popen, PIPE

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--odir", default = "seg_viz")

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

    ifile = h5py.File(args.ifile, 'r+')
    cases = sorted(list(ifile.keys()))

    for case in cases:
        reps = list(ifile[case].keys())

        for rep in reps:
            logging.info('working on case {0}, rep {1}'.format(case, rep))

            if 'xy_phi' in list(ifile[case][rep].keys()):
                xy = np.array(ifile[case][rep]['xy'])
                stack = np.array(ifile[case][rep]['stack'])
                xy_phi = np.array(ifile[case][rep]['xy_phi'])
                speed = np.array(ifile[case][rep]['speed'])
                speed_phi = np.array(ifile[case][rep]['speed_phi'])
            else:
                continue

            # make a movie
            cmd = "ffmpeg -y -f rawvideo -loglevel panic -vcodec rawvideo -s {1}x{2} -pix_fmt rgb24 -r 4 -i - -an -vcodec libx264 -pix_fmt yuv420p {0}".format(
                os.path.join(args.odir, '{0}_{1}.mp4'.format(case, rep)), 1400, 900).split(' ')
            p = Popen(cmd, stdin=PIPE)

            for frame in range(stack.shape[0]):
                xy = xy_phi[frame]

                xs = xy[:, 0]
                ys = xy[:, 1]

                fig = plt.figure(figsize=(14, 9))
                ax = plt.subplot(221)

                ax.set_title('Frame {0}'.format(frame))
                ax.imshow(stack[frame], cmap='gray')
                ax.plot(xs, ys)

                ax = plt.subplot(222)

                for k in range(72):
                    ax.plot(xy_phi[:frame, k, 0], xy_phi[:frame, k, 1], color = 'k')

                ax = plt.subplot(223)

                im = copy.copy(speed)
                im[:,frame:] = np.nan

                ax.imshow(im)
                ax.set_title('speed')

                ax = plt.subplot(224)

                im = copy.copy(speed_phi)
                im[:, frame:] = np.nan

                ax.imshow(im)
                ax.set_title('speed_phi')

                canvas = FigureCanvas(fig)
                canvas.draw()

                buf = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(900, 1400, 3)
                p.stdin.write(buf.tostring())

                plt.close()

            p.stdin.close()
            p.wait()






if __name__ == '__main__':
    main()
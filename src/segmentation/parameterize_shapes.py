import os
import sys
import itertools

import os
import logging, argparse
import itertools

import platform
import h5py
import numpy as np

from func import PointCloud
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from subprocess import Popen, PIPE

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--odir", default = "cyto_data/movies")

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

    pc = PointCloud(N = 72, Nk = 3)

    for case in cases:
        reps = list(ifile[case].keys())

        for rep in reps:
            logging.info('working on case {0}, rep {1}'.format(case, rep))

            if 'xy_phi' in list(ifile[case][rep].keys()):
                continue

            xy = np.array(ifile[case][rep]['xy'])
            stack = np.array(ifile[case][rep]['stack'])

            pc.load_array(xy)
            pc.solve_axial_slices()
            pc.solve_phi()


            try:
                # make a movie
                cmd = "ffmpeg -y -f rawvideo -loglevel panic -vcodec rawvideo -s {1}x{2} -pix_fmt rgb24 -r 4 -i - -an -vcodec libx264 -pix_fmt yuv420p {0}".format(
                    os.path.join(args.odir, '{0}_{1}.mp4'.format(case, rep)), 800, 600).split(' ')
                p = Popen(cmd, stdin=PIPE)

                for frame in range(stack.shape[0]):
                    xy = pc.xy[frame]

                    xs = xy[:, 0]
                    ys = xy[:, 1]

                    fig = plt.figure(figsize=(8, 6))
                    ax = plt.subplot(111)

                    ax.set_title('Frame {0}'.format(frame))
                    ax.imshow(stack[frame], cmap='gray')
                    ax.plot(xs, ys)

                    canvas = FigureCanvas(fig)
                    canvas.draw()

                    buf = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(600, 800, 3)
                    p.stdin.write(buf.tostring())

                    plt.close()

                p.stdin.close()
                p.wait()

                speed, speed_phi = pc.compute_speeds()

                fig, axes = plt.subplots(ncols = 2)
                im1 = axes[0].imshow(speed)
                im2 = axes[1].imshow(speed_phi)

                fig.colorbar(im1, ax = axes[0])
                fig.colorbar(im2, ax = axes[1])

                plt.savefig(os.path.join(args.odir, '{0}_{1}.png'.format(case, rep)), dpi = 100)
                plt.close()

                ifile.create_dataset('{0}/{1}/speed'.format(case, rep), data = speed)
                ifile.create_dataset('{0}/{1}/speed_phi'.format(case, rep), data = speed_phi)
                ifile.create_dataset('{0}/{1}/xy_smooth'.format(case, rep), data = pc.xy)
                ifile.create_dataset('{0}/{1}/xy_phi'.format(case, rep), data = pc.xy_phi)

                ifile.flush()
            except:
                continue

    ifile.close()

if __name__ == '__main__':
    main()
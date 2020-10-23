#!/usr/bin/env python
"""
Author: Dylan Ray

Segments, smooths segmentations, and saves.  Gets smooth, spatially and temporally, estimates of the cytokinetic ring
in microscopy data of single-celled C. Elegans embryos.

Runs like:
python2 extract_optnet.py --verbose --idir some_folder
"""

import numpy as np

from tifffile import imread, imsave

from data3d import Data3d
import logging
import argparse
import sys
import os
import h5py

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

from subprocess import Popen, PIPE

def get_number(string):
    ret = ''
    count = 0

    for s in string:
        if s.isdigit():
            ret += s
            count += 1

        if count == 3:
            break

    return ret

if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser(
        description="Takes a folder and segments all the TIF stacks within it using Florians's routines.")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbosity settings.  Will print progress to the console.")
    parser.add_argument("--idir", default="cyto_ring_data", help="Folder to segment.")
    parser.add_argument("--ofile", default="data_v7.0.hdf5", help="Save as HDF5 database.")
    parser.add_argument("--normalize", action="store_true",
                        help="Routine that was originally in the workflow wher we normalize row wise in each image.  Doesn't seem to help too much...")
    parser.add_argument("--movie_dir", default="movies", help="Directory for visualization movies.")

    parser.add_argument("--plot", action = "store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if os.path.exists(args.movie_dir):
        os.system('rm {0}/*'.format(args.movie_dir))
    else:
        os.mkdir(args.movie_dir)

    # get a list of files
    ifiles = os.listdir(args.idir)

    # we want the TIFs
    stacks = [u for u in ifiles if u.split('.')[-1] == 'tif' or u.split('.')[-1] == 'tiff']

    # identify each TIF stack by a single number from its file name
    numbers = [get_number(u) for u in stacks]
    stacks = [x for y, x in sorted(zip(numbers, stacks))]

    # initialize the output file
    ofile = h5py.File(args.ofile, 'w')

    # counter for progress output
    ix = 0

    n_files = len(stacks)

    # legacy data structure (ignore for now)
    result = dict()
    result['number'] = list()
    result['file_name'] = list()

    # keep track to avoid repeats
    numbers_done = []

    for ifile in stacks:
        number = get_number(ifile)

        logging.debug('0: working on file {0} of {1}'.format(ix + 1, n_files))

        try:
            stack = imread(os.path.join(args.idir, ifile))
        except:
            logging.debug('0: couldnt read {0}'.format(ifile))
            continue

        # normalize the stack
        stack = stack.astype(np.float32) / np.max(stack.astype(np.float32))

        print(stack.shape)

        if len(stack.shape) != 3:
            logging.debug('0: mismatched shape! for file {0}'.format(ifile))
            continue

        # Florian's routine
        # Based on:
        """
        A 2d version of the optimal net surface problem.
        Relevant publication: [Wu & Chen 2002]
        """
        data = Data3d(stack, pixelsize=(1., 1.), silent=True)
        data.set_seg_params(num_columns = 72, K = 250, max_delta_k = 3)

        ring = data.init_object("ring")
        max_rs = (stack[0].shape[1] / 2,
                  stack[0].shape[1] / 2)
        min_rs = (5, 5)

        cx = stack.shape[2] / 2
        cy = stack.shape[1] / 2
        data.add_object_at(ring, min_rs, max_rs, frame=0, seed=(cx, cy), segment_it=True)
        data.track(ring, seed_frame = 0, target_frames = range(0, len(data.images)), recenter_iterations = 2)

        poly = []

        for frame in range(0, len(data.images)):
            xy = np.array(data.get_result_polygone(ring, frame))

            poly.append(xy)

        poly = np.array(poly)

        if args.plot:
            # make a movie
            cmd = "ffmpeg -y -f rawvideo -loglevel panic -vcodec rawvideo -s {1}x{2} -pix_fmt rgb24 -r 4 -i - -an -vcodec libx264 -pix_fmt yuv420p {0}".format(
                os.path.join(args.movie_dir, '{0}.mp4'.format(number)), 800, 600).split(' ')
            p = Popen(cmd, stdin=PIPE)

            for frame in range(0, len(data.images)):
                xy = poly[frame]

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

        # save and move on
        if number in numbers_done:
            pass
        else:
            ofile.create_dataset('{0}/xy'.format(number), data = poly)

            result['number'].append(number)

        result['file_name'].append(ifile)

        numbers_done.append(number)
        ix += 1

    ofile.close()











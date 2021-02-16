import os
import sys
import itertools

import os
import logging, argparse
import itertools

import platform
import random
import numpy as np

from tifffile import imsave, imread
import matplotlib.pyplot as plt
import cv2

target_size = (280, 288)

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--idir_synth", default = "cyto_data/synthetic")
    parser.add_argument("--idir", default = "cyto_data/cyto_ring_data")

    parser.add_argument("--odir", default = "cyto_data/CycleGAN_data/")

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

    synthetic_dirs = [os.path.join(args.idir_synth, u) for u in os.listdir(args.idir_synth)]
    ims = [os.path.join(args.idir, u) for u in os.listdir(args.idir)]

    random.shuffle(ims)

    i = int(np.round(len(ims) * 0.9))
    os.mkdir(os.path.join(args.odir, 'trainB'))
    ix = 0

    for im in ims[:i]:
        try:
            stack = imread(im)
        except:
            logging.debug('0: couldnt read {0}'.format(im))
            continue

        stack = stack.astype(np.float32) / np.max(stack)
        stack = (stack * 255).astype(np.uint8)

        if (stack.shape[1] > 256) and (stack.shape[2] > 256):
            # randomly crop to target size
            for k in range(stack.shape[0]):
                Nx = stack.shape[2] // 2
                Ny = stack.shape[1] // 2
                im = stack[k][Ny - 128:Ny + 128, Nx - 128:Nx + 128]

                cv2.imwrite(os.path.join(os.path.join(args.odir, 'trainB'), '{0:06d}.png'.format(ix)), im)

                ix += 1

    os.mkdir(os.path.join(args.odir, 'testB'))
    ix = 0

    for im in ims[i:]:
        try:
            stack = imread(im)
        except:
            logging.debug('0: couldnt read {0}'.format(im))
            continue

        stack = stack.astype(np.float32) / np.max(stack)
        stack = (stack * 255).astype(np.uint8)

        if (stack.shape[1] > 256) and (stack.shape[2] > 256):
            # randomly crop to target size
            for k in range(stack.shape[0]):
                Nx = stack.shape[2] // 2
                Ny = stack.shape[1] // 2
                im = stack[k][Ny - 128:Ny + 128, Nx - 128:Nx + 128]

                cv2.imwrite(os.path.join(os.path.join(args.odir, 'testB'), '{0:06d}.png'.format(ix)), im)

                ix += 1


    synthetic_ims = []

    for dir in synthetic_dirs:
        synthetic_ims.extend([os.path.join(os.path.join(dir, 'images'), u) for u in os.listdir(os.path.join(dir, 'images'))])

    random.shuffle(synthetic_ims)

    # take 90 % for training
    os.mkdir(os.path.join(args.odir, 'trainA'))
    i = int(np.round(len(synthetic_ims) * 0.9))

    ix = 0

    for im in synthetic_ims[:i]:
        im = cv2.imread(im)

        Nx = im.shape[1] // 2
        Ny = im.shape[0] // 2
        im = im[Ny - 128:Ny + 128, Nx - 128:Nx + 128]

        im = im.astype(np.float32) / np.max(im)
        im = (im * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(os.path.join(args.odir, 'trainA'), '{0:06d}.png'.format(ix)), im)
        ix += 1

    # take 90 % for training
    os.mkdir(os.path.join(args.odir, 'testA'))
    ix = 0

    for im in synthetic_ims[i:]:
        im = cv2.imread(im)
        Nx = im.shape[1] // 2
        Ny = im.shape[0] // 2
        im = im[Ny - 128:Ny + 128, Nx - 128:Nx + 128]

        im = im.astype(np.float32) / np.max(im)
        im = (im * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(os.path.join(args.odir, 'testA'), '{0:06d}.png'.format(ix)), im)
        ix += 1



if __name__ == '__main__':
    main()
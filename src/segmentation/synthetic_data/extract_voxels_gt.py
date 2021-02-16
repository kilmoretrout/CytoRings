import os
import logging, argparse
import numpy as np

from scipy.signal import convolve2d

from tifffile import imsave, imread

# distance away from coverslip (micro-meters), sigma_x (nm), sigma_y (nm), translation (micro-meters)
import torch
from skimage.filters import gaussian

from scipy.stats import norm
from scipy.interpolate import interp1d, interp2d
import cv2

def strideConv(arr, arr2, s):
    return convolve2d(arr, arr2[::-1, ::-1], mode='valid')[::s, ::s]

def get_gauss(sx, sy):
    N = int(np.ceil(sy * 2))
    if N % 2 == 0:
        N -= 1

    x, y = np.meshgrid(np.linspace(-2*sy, 2*sy, N), np.linspace(-2*sy, 2*sy, N))

    Z = np.exp(-0.5 * (x ** 2 / sx**2 + y ** 2 / sy**2)) * (1. / (2 * np.pi * sx * sy))

    return Z

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()

    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")

    parser.add_argument("--blur_file", default = "src/segmentation/synthetic_data/psfshift.csv")
    parser.add_argument("--filter", default = "src/segmentation/synthetic_data/confocalPSF_werner_XZplane.tif")

    parser.add_argument("--idir", default = "csvs_daniel")
    parser.add_argument("--odir", default = "cyto_data/synthetic")

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

    return args

from scipy.ndimage import gaussian_filter

def main():
    args = parse_args()

    psf = np.loadtxt(args.blur_file, delimiter = ',')
    sigma_x = interp1d(psf[:,0], psf[:,1], kind = 'cubic') # micrometers -> nm
    sigma_y = interp1d(psf[:,0], psf[:,2], kind = 'cubic') # micrometers -> nm
    translation = interp1d(psf[:,0], psf[:,3], kind = 'cubic') # micrometers -> micrometers

    psf = imread(args.filter)

    ifiles = os.listdir(args.idir)

    device = torch.device("cuda")
    module = torch.nn.Conv2d(
        in_channels=1, out_channels=1, kernel_size=52, padding=0, bias=False, stride=52
    )
    module.weight.data = torch.full_like(module.weight.data, 1.0)
    module.to(device)

    for ifile in ifiles:
        print('working on {0}'.format(ifile))

        odir = os.path.join(args.odir, ifile.split('.')[0])

        if not os.path.exists(odir):
            os.system('mkdir -p {0}'.format(odir))
            os.system('mkdir -p {0}'.format(os.path.join(odir, 'images')))
        else:
            continue

        ifile = open(os.path.join(args.idir, ifile), 'r')
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

        print('gathering coordinates...')
        while True:
            if '% time' in line:
                t.append(float(line.split(' ')[-1]))
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

        print('generating voxels...')



        counter = 0

        N = np.sum([len(u) for u in xy])
        xyz = np.zeros((N, 3))

        ix = 0

        for k in range(len(xy)):
            print(len(xy[k]), t[k])
            if len(xy[k]) > 0:
                # pixel coordinate
                xyz[ix:ix+len(xy[k]),:2] = xy[k]
                # time in seconds
                xyz[ix:ix+len(xy[k]),2] = t[k]

                ix += len(xy[k])

                vox = np.zeros((20000, 20000), dtype = np.float32)

                xbins = np.linspace(-20, 20, 20001)
                ybins = np.linspace(-20, 20, 20001)

                xy_ = xy[k]

                for j in range(len(xy_)):
                    y = xy_[j][1] + 15 # distance from the cover-slip

                    sx = sigma_x(y)
                    sy = sigma_y(y)
                    trans = translation(y)

                    xy_[j][1] = xy_[j][1] + trans
                    Z = get_gauss(sx, sy)

                    ix = np.digitize(xy_[j][0], xbins)
                    iy = 20000 - np.digitize(xy_[j][1], ybins)

                    N = (Z.shape[0] - 1) // 2

                    vox[iy - N:iy + N + 1, ix - N:ix + N + 1] += Z

                vox = vox[2500:-2500,2500:-2500].reshape(1, 1, 15000, 15000)

                vox = torch.FloatTensor(vox).to(device)

                with torch.no_grad():
                    vox = module(vox).detach().cpu().numpy()

                vox = vox.reshape(vox.shape[-2], vox.shape[-1])
                vox = convolve2d(vox, psf, mode = 'same')

                vox /= np.max(vox)

                vox_stretched = []

                x = list(range(0, vox.shape[0], 10))
                x_ = list(range(max(x)))

                for k in range(vox.shape[1]):
                    y = [vox[:,k][u] for u in x]
                    f = interp1d(x, y)

                    vox_stretched.append(np.array(f(x_)))

                vox_stretched = np.vstack(vox_stretched).T
                vox_stretched = np.round(255*vox_stretched).astype(np.uint8)

                cv2.imwrite(os.path.join(os.path.join(odir, 'images'), '{0:04d}.png'.format(counter)), vox_stretched)

                counter += 1

        np.savetxt(os.path.join(odir, 'points.txt'), xyz)

if __name__ == '__main__':
    main()

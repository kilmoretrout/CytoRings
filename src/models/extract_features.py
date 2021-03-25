import os
import sys
import itertools

import os
import logging, argparse
import itertools

import platform
import h5py
import numpy as np
from scipy.signal import periodogram, welch, spectrogram
from scipy.interpolate import UnivariateSpline, interp1d


def poly_area(xy):
    x = xy[:, 0]
    y = xy[:, 1]

    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--odir", default = "None")

    parser.add_argument("--trim", default = "5")
    parser.add_argument("--features", default = "fourier_spectra")

    parser.add_argument("--key", default = "speed_phi")

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
    cases = list(ifile.keys())

    trim = int(args.trim)

    X = []
    F = []

    periods = np.linspace(8., 120., 100)

    for case in cases:
        odir = os.path.join(args.odir, case)
        os.mkdir(odir)

        for number in list(ifile[case].keys()):
            if args.key in ifile[case][number].keys():
                speed = np.array(ifile[case][number][args.key], dtype = np.float32)
                xy = np.array(ifile[case][number]['xy'], dtype = np.float32)

                speed = speed[:, trim : -trim]
                xy = xy[trim : -trim, :, :]

                area = np.array([poly_area(xy[k]) for k in range(xy.shape[0])])
                t = np.array(range(speed.shape[1])) * 2.7

                if args.features == "fourier_spectra":
                    f = []
                    x = []

                    for s in speed:
                        x_ = np.abs(np.fft.fft(s, 100))**2
                        f_ = np.fft.fftfreq(100, 2.7)

                        x.append(x_[f_ > 0])
                        f.append(f_[f_ > 0])

                    feature_names = ['{} s'.format(int(np.round(u ** -1))) for u in f[-1]]

                    np.savez(os.path.join(odir, '{}.npz'.format(number)),
                             x = np.array(x, dtype = np.float32),
                             f = np.array(f, dtype = np.float32), feature_names = np.array(feature_names, dtype = str))
                elif args.features == "periodogram":
                    f = []
                    x = []

                    for s in speed:
                        f_, x_ = periodogram(s, 1.0 / 2.7)
                        p = f_ ** -1
                        x_ = interp1d(p, x_)

                        x.append(x_(periods))
                        f.append(periods)

                    feature_names = ['{} s'.format(int(np.round(u))) for u in f[-1]]

                    np.savez(os.path.join(odir, '{}.npz'.format(number)),
                             x=np.array(x, dtype=np.float32),
                             f=np.array(f, dtype=np.float32), feature_names = np.array(feature_names, dtype = str))
                elif args.features == "simple":
                    x = []

                    max_area_diff = np.max(np.abs(np.diff(area)))
                    area_var = np.var(np.diff(area))

                    for s in speed:
                        x_ = []

                        x_.append(np.mean(s))
                        x_.append(np.max(s))
                        x_.append(np.min(s))
                        x_.append(np.var(s))
                        x_.append(float(len(s)))

                        # get the number of inflection points
                        # and max and mins
                        f_s = UnivariateSpline(t, s, s = 0.000001, k = 4)

                        max_mins = f_s.derivative(1).roots()

                        # get the number of inflection points
                        # and max and mins
                        f_s = UnivariateSpline(t, s, s=0.000001, k = 5)

                        infs = f_s.derivative(2).roots()

                        x_.append(len(max_mins))
                        x_.append(len(infs))

                        x.append(x_)

                    feature_names = ['mean', 'max', 'min', 'var', 'length', 'max_mins', 'inf points']

                    np.savez(os.path.join(odir, '{}.npz'.format(number)),
                             x=np.array(x, dtype=np.float32), feature_names = np.array(feature_names, dtype = str))

                # power spectral density using Welch's method
                elif args.features == "welch":
                    f = []
                    x = []

                    for s in speed:
                        f_, x_ = welch(s, 1.0 / 2.7)
                        p = f_ ** -1
                        x_ = interp1d(p, x_)

                        x.append(x_(periods))
                        f.append(periods)

                    feature_names = ['{} s'.format(int(np.round(u))) for u in f[-1]]

                    np.savez(os.path.join(odir, '{}.npz'.format(number)),
                             x=np.array(x, dtype=np.float32),
                             f=np.array(f, dtype=np.float32), feature_names = np.array(feature_names, dtype = str))

                # spectrogram
                elif args.features == "spectrogram":
                    f = []
                    x = []

                    for s in speed:
                        # check docs
                        f_, x_ = spectrogram(s, 1.0 / 2.7)
                        p = f_ ** -1
                        x_ = interp1d(p, x_)

                        x.append(x_(periods))
                        f.append(p)

                    np.savez(os.path.join(odir, '{}.npz'.format(number)),
                             x=np.array(x, dtype=np.float32),
                             f=np.array(f, dtype=np.float32))



if __name__ == '__main__':
    main()
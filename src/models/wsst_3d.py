import numpy as np
import h5py

from ns_3d import NetSurf3d

import os
import sys
import itertools

import logging, argparse
from ssqueezepy import ssq_cwt, issq_cwt, ssq_stft

from mpi4py import MPI

from mpi4py.MPI import ANY_SOURCE
import matplotlib.pyplot as plt

def make_matrix(rows, cols):
    n = rows*cols
    M = np.zeros((n,n))
    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            # Two inner diagonals
            if c > 0: M[i-1,i] = M[i,i-1] = 1
            # Two outer diagonals
            if r > 0: M[i-cols,i] = M[i,i-cols] = 1

    return M

def get_optimal_surface(sst, K = 250, max_delta_k = 1):
    X, Y = np.meshgrid(range(0, sst.shape[2], 1), range(0, sst.shape[1], 1))

    columns = []

    for i in range(len(X)):
        for j in range(X.shape[1]):
            columns.append(np.array([X[i,j], Y[i,j]]))

    columns = np.array(columns)
    A = make_matrix(len(X), X.shape[1])

    neighbors_of = list()

    for j in range(len(A)):
        _ = []

        for k in range(A.shape[1]):
            if A[j,k] == 1:
                _.append(k)

        neighbors_of.append(_)

    ns = NetSurf3d(columns, None, neighbors_of, K = K, max_delta_k = max_delta_k)
    ns.apply_to(sst)

    xyz = []

    for k in range(len(columns)):
        x, y, z = ns.get_surface_point(k)

        xyz.append(np.array([x, int(np.round(y)), int(np.round(z))]))

    return np.array(xyz)

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--ofile", default = "None")

    parser.add_argument("--K", default="250")
    parser.add_argument("--max_delta_k", default="4")
    parser.add_argument("--n_pixels", default="8")
    parser.add_argument("--n_surfaces", default="10")

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
    # configure MPI
    comm = MPI.COMM_WORLD

    args = parse_args()

    K = int(args.K)
    max_delta_k = int(args.max_delta_k)
    n_pixels = int(args.n_pixels)

    ifile = h5py.File(args.ifile, 'r')
    cases = list(ifile.keys())

    if comm.rank == 0:
        ofile = h5py.File(args.ofile, 'w')

    trim = int(args.trim)

    # gather the treatment-replicate pairs (every process gets one)
    case_numbers = []

    for case in cases:
        numbers = ifile[case].keys()

        for number in numbers:
            case_numbers.append((case, number))

    if comm.rank != 0:
        for ix in range(comm.rank - 1, len(case_numbers), comm.size - 1):
            case, number = case_numbers[ix]
            print('{0}: working on case {1}, {2}'.format(comm.rank, case, number))

            if 'speed_phi' in ifile[case][number].keys():
                speed = np.array(ifile[case][number]['speed_phi'], dtype = np.float32)
                speed = speed[:, trim : -trim]

                t = np.array(range(speed.shape[1])) * 2.7

                W = []

                for ix in range(speed.shape[0]):
                    Tx, ssq_freqs, Wx, scales, w = ssq_cwt(speed[ix])
                    period = (np.array(ssq_freqs) ** -1) * 2.7

                    W.append(Tx)

                W_abs = np.abs(np.array(W)).transpose(1, 0, 2)
                W = np.array(W).transpose(1, 0, 2)

                IMFs = []
                xyzs = []

                for ix in range(int(args.n_surfaces)):
                    xyz = get_optimal_surface(W_abs, K = K, max_delta_k = max_delta_k)
                    xyzs.append(xyz)

                    xyz_mask = []
                    xyz_del = []

                    for k in range(xyz.shape[0]):
                        for j in range(-n_pixels, n_pixels + 1):
                            xyz_mask.append((xyz[k,2] + j, xyz[k,1], xyz[k,0]))

                        for j in range(-4, 4):
                            xyz_del.append((xyz[k, 2] + j, xyz[k, 1], xyz[k, 0]))


                    xyz_mask = np.array(xyz_mask, dtype = np.int64)
                    xyz_del = np.array(xyz_del, dtype=np.int64)

                    W_ = np.zeros(W.shape, dtype = np.complex64)
                    W_[list(xyz_mask[:,0]), list(xyz_mask[:,1]), list(xyz_mask[:,2])] = W[list(xyz_mask[:,0]), list(xyz_mask[:,1]), list(xyz_mask[:,2])]

                    W_abs[list(xyz_del[:,0]), list(xyz_del[:,1]), list(xyz_del[:,2])] = 0.

                    IMF = np.zeros(speed.shape, dtype = np.float32)
                    for k in range(72):
                        IMF[k,:] = issq_cwt(W_[:,k,:], 'morlet')

                    IMFs.append(IMF)

                comm.send([case, number, IMFs, xyzs, period], dest = 0)
            else:
                comm.send([case, number, None, None, None], dest = 0)


    else:
        done = 0

        while done != len(case_numbers):
            case, number, IMFs, xyzs, period = comm.recv(source=ANY_SOURCE)

            if period is not None:
                print('0: writing results for {0}, {1}'.format(case, number))

                ofile.create_dataset('{0}/{1}/xyzs'.format(case, number), data = np.array(xyzs, dtype = np.int64))
                ofile.create_dataset('{0}/{1}/IMFs'.format(case, number), data = np.array(IMFs, dtype = np.float32))
                ofile.create_dataset('{0}/{1}/period'.format(case, number), data = period)

            done += 1

        ofile.close()

if __name__ == '__main__':
    main()
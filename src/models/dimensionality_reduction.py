import os
import sys
import itertools

import os
import logging, argparse
import itertools

import platform
import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import random

CASES = ['Control', 'Act-2', 'Pfn-1', 'Cyk-1', 'Ect-2', 'Plst-1', 'Nop-1']
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

def get_set(idir):
    models = [os.path.join(idir, u) for u in CASES]
    npzs = dict()

    for model in models:
        npzs_ = [os.path.join(model, u) for u in os.listdir(model)]
        random.shuffle(npzs_)

        npzs[model] = npzs_

    return npzs

def get_train_val(idir, N = 1):
    train_dataset = get_set(idir)
    val_dataset = dict()

    models = sorted(list(train_dataset.keys()))
    # leave N cells out for validation
    for model in models:
        ix = np.random.choice(range(len(train_dataset[model])), N)

        val_dataset[model] = [train_dataset[model][u] for u in ix]
        train_dataset[model] = [train_dataset[model][u] for u in range(len(train_dataset[model])) if u not in ix]

    train_x = []
    train_y = []

    val_x = []
    val_y = []

    for model in models:
        y_ = models.index(model)

        train_x.append(np.vstack([np.load(u)['x'] for u in train_dataset[model]]))
        train_y.append(np.repeat(y_, (train_x[-1].shape[0], )))

        val_x.append(np.vstack([np.load(u)['x'] for u in val_dataset[model]]))
        val_y.append(np.repeat(y_, (val_x[-1].shape[0],)))

    feature_names = list(np.load(train_dataset[models[0]][0])['feature_names'])

    train_x = np.vstack(train_x)
    val_x = np.vstack(val_x)

    train_y = np.hstack(train_y)
    val_y = np.hstack(val_y)

    # normalize
    mu = np.mean(train_x, axis = 0)
    std = np.std(train_x, axis = 0)

    #train_x = (train_x - np.mean(train_x, axis = 0)) / np.std(train_x, axis = 0)
    #val_x = (val_x - mu) / std

    return train_x, train_y, val_x, val_y, models, feature_names




def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--odir", default = "None")

    parser.add_argument("--type", default = "pca")
    parser.add_argument("--mean", action = "store_true")

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

    data = get_set(args.idir)

    if args.type == 'pca':
        pca = PCA(2)

        indices = dict()

        ix = 0

        X = []
        for model in data.keys():
            indices[model] = []

            for npz in data[model]:
                x = np.load(npz)['x']
                print(x.shape)

                if args.mean:
                    X.append(np.mean(x, axis = 0).reshape(1, x.shape[1]))
                    indices[model].append(ix)

                    ix += 1
                else:
                    X.append(x)
                    indices[model].extend(list(range(ix, ix + x.shape[0])))

                    ix += x.shape[0]

        X = np.vstack(X)
        Y = pca.fit_transform(X)

        models = list(data.keys())

        for ix in range(len(models)):
            ix_ = indices[models[ix]]

            plt.scatter(Y[ix_, 0], Y[ix_, 1], c = COLORS[ix], label = CASES[ix])

        plt.legend()
        plt.show()


        return


if __name__ == '__main__':
    main()
import os
import sys
import itertools

import os
import logging, argparse
import itertools

import platform
from sklearn.ensemble import RandomForestClassifier
from dimensionality_reduction import get_set, get_train_val

from sklearn.metrics import confusion_matrix
import numpy as np

import sys, os
import numpy as np
import random
from glob import glob

import matplotlib.pyplot as plt
from itertools import product
from matplotlib.gridspec import GridSpec

class ConfusionMatrixDisplay:
    """Confusion Matrix visualization.
    It is recommend to use :func:`~sklearn.metrics.plot_confusion_matrix` to
    create a :class:`ConfusionMatrixDisplay`. All parameters are stored as
    attributes.
    Read more in the :ref:`User Guide <visualizations>`.
    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix.
    display_labels : ndarray of shape (n_classes,)
        Display labels for plot.
    Attributes
    ----------
    im_ : matplotlib AxesImage
        Image representing the confusion matrix.
    text_ : ndarray of shape (n_classes, n_classes), dtype=matplotlib Text, \
            or None
        Array of matplotlib axes. `None` if `include_values` is false.
    ax_ : matplotlib Axes
        Axes with confusion matrix.
    figure_ : matplotlib Figure
        Figure containing the confusion matrix.
    """
    def __init__(self, confusion_matrix, display_labels):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    def plot(self, include_values=True, cmap='Blues',
             xticks_rotation='horizontal', values_format=None, ax=None):
        """Plot visualization.
        Parameters
        ----------
        include_values : bool, default=True
            Includes values in confusion matrix.
        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.
        xticks_rotation : {'vertical', 'horizontal'} or float, \
                         default='horizontal'
            Rotation of xtick labels.
        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`,
            the format specification is '.2g'.
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        Returns
        -------
        display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
        """
        #check_matplotlib_support("ConfusionMatrixDisplay.plot")
        #import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        self.text_ = None

        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)
            if values_format is None:
                values_format = '.2g'

            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0
            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min
                self.text_[i, j] = ax.text(j, i,
                                           format(cm[i, j], values_format),
                                           ha="center", va="center",
                                           color=color)

        fig.colorbar(self.im_, ax=ax)
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=self.display_labels,
               yticklabels=self.display_labels,
               ylabel="True label",
               xlabel="Predicted label")

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--odir", default = "None")

    parser.add_argument("--type", default = "rf")

    # parameters for CV
    parser.add_argument("--n_cv", default = "5")
    parser.add_argument("--N", default = "2")

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

def main():
    args = parse_args()

    tag = args.idir.split('/')[-1].split('.')[0]

    if args.type == 'rf':
        accs = []
        cms = []

        for ix in range(int(args.n_cv)):
            train_x, train_y, val_x, val_y, models, feature_labels = get_train_val(args.idir, N = int(args.N))

            model = RandomForestClassifier()
            model.fit(train_x, train_y)

            val_predictions = model.predict(val_x)

            accs.append(model.score(val_x, val_y))
            cm = np.array(confusion_matrix(val_y, val_predictions), dtype = np.float32)

            cms.append(confusion_matrix(val_y, val_predictions))

        print('got mean accuracy of {0}'.format(np.mean(accs)))

        cm = np.sum(np.array(cms, dtype = np.float32), axis = 0)
        print(cm)

        for k in range(cm.shape[0]):
            cm[k, :] /= np.sum(cm[k, :])

        plt.rc('font', family='Helvetica', size=12)
        plt.rcParams.update({'figure.autolayout': True})
        fig = plt.figure(figsize=(14, 8), dpi=100)

        gs = GridSpec(1, 4)
        ax1 = fig.add_subplot(gs[0,:2])
        ax2 = fig.add_subplot(gs[0,2:])

        importances = model.feature_importances_

        try:
            ax1.scatter([int(u.split(' ')[0]) for u in feature_labels], importances)
        except:
            ax1.scatter(feature_labels, importances)
        ax1.set_xlabel('period (s)')
        ax1.set_ylabel('feature importance')

        ax2.set_title('average confusion matrix (n_cv = 5, validation cells = 2)')
        cm_display = ConfusionMatrixDisplay(cm, [u.split('/')[-1] for u in models])
        cm_display.plot(ax = ax2)
        plt.savefig(os.path.join(args.odir, 'fi_{}.png'.format(tag)), dpi = 100)




if __name__ == '__main__':
    main()
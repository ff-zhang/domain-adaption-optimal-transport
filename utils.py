import numpy as np
import scipy
import ot

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
import torchvision
import torchvision.transforms as v2

import itertools
import matplotlib.pyplot as plt
from typing import Union, Callable


def load_dataset(roots: list[str], augment: bool = False) -> Dataset:
    assert roots is not None and len(roots) > 0

    if not augment:
        return ConcatDataset(datasets=[torchvision.datasets.ImageFolder(root=path) for path in roots])

    augment_image = v2.Compose([
        v2.Resize(size=(256, 256)),
        v2.RandomCrop(size=(224, 224), padding=4, padding_mode='reflect'),  # Random Cropping
        v2.RandomHorizontalFlip(p=0.5),  # Random flipping
        v2.RandomRotation(degrees=(-20, 20)),  # Random rotation
        v2.ColorJitter(brightness=.1, hue=.05),  # Random colour jitter
        v2.ToTensor(),
    ])

    return ConcatDataset(
        datasets=[torchvision.datasets.ImageFolder(root=path, transform=augment_image) for path in roots]
    )


def cond_bincount(dataset: torchvision.datasets) -> dict[int, np.array]:
    X, y = dataset.data.numpy(), dataset.targets.numpy()

    cond_bins = dict()
    for v in dataset.class_to_idx.values():
        cond_bins[v] = np.apply_along_axis(np.bincount, axis=0, arr=X[y==v], minlength=256)

    return cond_bins


def plot_histogram(histogram: dict[int, np.array], save: bool = True, path: str = 'img/tmp.png') -> None:
    # Plot the 14th column of the given image histogram.
    fig, axs = plt.subplots(histogram.shape[1], sharex='all', figsize=(10, 60))
    for i in range(histogram.shape[1]):
        axs[i].bar([n for n in range(256)], histogram[:, i, 14], log=True)
    plt.tight_layout()
    plt.show() if not save else plt.savefig(path)


def best_fit_conditional(histogram: dict[int, np.array]) -> list:
    best = {k: [[[] for _ in range(28)] for _ in range(28)] for k in histogram.keys()}

    X = [n for n in range(256)]
    for k, v in histogram.items():
        for i, j in itertools.product(*map(lambda n: range(n), v.shape[1:])):
                log_y = np.log(histogram[k][:, i, j])
                log_y[np.logical_or(np.isnan(log_y), np.isinf(log_y))] = 0.

                best[k][i][j] = scipy.interpolate.splrep(X, log_y, s=96)

    return best


def plot_spline(histogram: np.array, spline: list[tuple], save: bool = True, path: str = 'img/tmp.png') -> None:
    # Plot the 14th column of the given image histogram and each pixel's associated spline.
    fig, axs = plt.subplots(histogram.shape[1], sharex='all', figsize=(10, 60))
    X = [n for n in range(256)]
    for i in range(histogram.shape[1]):
        axs[i].bar(X, histogram[:, i, 14], log=True, color='blue')
        axs[i].plot(X, scipy.interpolate.BSpline(*spline[i][14])(X), color='red')
    plt.tight_layout()
    plt.show() if not save else plt.savefig(path)


def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(28, 28), cmap=plt.cm.binary, vmin=None, vmax=None):
    """Images should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
                      col_start: col_start + digit_dimensions[1]] = cur_image
        cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    return cax


def plot_accuracy(acc: dict[str, list[float]], title: str = None) -> None:
    t = 0
    for k, v in acc.items():
        if len(v) == 0:
            continue
        t = len(v)
        plt.plot([i + 1 for i in range(t)], v, label=k.capitalize(), marker='o')

    plt.xlabel('Epochs')
    plt.xticks([i + 1 for i in range(t)])
    plt.xlim(left=1, right=t)

    plt.ylabel('Accuracy')
    plt.legend()
    if title is not None:
        plt.title(title)

    plt.show()


def mle_estimator(data, targets):
    targets = np.eye(10)[targets]
    num_class = np.sum(targets, axis=0)

    theta_mle = np.dot(data.T, targets) / num_class
    pi_mle = num_class / data.shape[0]

    return theta_mle, pi_mle


def plot_mle(images, **kwargs):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)

    plot_images(images, ax, **kwargs)

    fig.patch.set_visible(False)
    ax.patch.set_visible(False)

    plt.show()


def minmax(img):
    return np.clip(img, 0, 1)


def fit_transports(Xs: np.array, ys: np.array = None, Xt: np.array = None,  yt: np.array = None,
                   function: Union[str, list[str]] = 'all') -> list[ot.da.BaseTransport]:
    out = []

    if function == 'all' or 'emd' in function:
        print("Fitting EMD transport function.")
        ot_emd = ot.da.EMDTransport()
        ot_emd.fit(Xs, ys, Xt, yt)
        out.append(ot_emd)

    if function == 'all' or 'sinkhorn' in function:
        print("Fitting sinkhorn transport function")
        ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-1)
        ot_sinkhorn.fit(Xs, ys, Xt, yt)
        out.append(ot_sinkhorn)

    if function == 'all' or 'linear' in function:
        print("Fitting mapping (linear) transport function.")
        ot_mapping_linear = ot.da.MappingTransport(
            mu=1e0, eta=1e-8, bias=True, max_iter=20, verbose=True)
        ot_mapping_linear.fit(Xs, ys, Xt, yt)
        out.append(ot_mapping_linear)

    if function == 'all' or 'gaussian' in function:
        print("Fitting mapping (Gaussian) transport function.")
        ot_mapping_gaussian = ot.da.MappingTransport(
            mu=1e0, eta=1e-2, sigma=1, bias=False, max_iter=10, verbose=True)
        ot_mapping_gaussian.fit(Xs, ys, Xt, yt)
        out.append(ot_mapping_gaussian)

    return out


if __name__ == '__main__':
    # Fix the random seed
    torch.manual_seed(64)

    train_mnist = torchvision.datasets.MNIST('./data', train=True, download=True)
    test_mnist = torchvision.datasets.MNIST('./data', train=False, download=True)

    train_emnist = torchvision.datasets.EMNIST('./data', 'digits', train=True, download=True)
    test_emnist = torchvision.datasets.EMNIST('./data', 'digits', train=False, download=True)

    hist_mnist = cond_bincount(train_mnist)
    hist_emnist = cond_bincount(train_emnist)

    # plot_histogram(hist_mnist[0], save=True, path='img/hist-mnist-0.png')
    # plot_histogram(hist_emnist[0], save=True, path='img/hist-emnist-0.png')

    spline_mnist = best_fit_conditional(hist_mnist)
    spline_emnist = best_fit_conditional(hist_emnist)

    plot_spline(hist_mnist[0], spline_mnist[0], True, 'img/spline-mnist-0.png')

    print('hello world!')

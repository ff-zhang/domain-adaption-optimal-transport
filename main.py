import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

import ot
import ot.plot
import geomloss

from model import LinearRegression, FullyConnectedNetwork, test_model, train_model
from utils import plot_accuracy, plot_images


def sample_normal(mean: np.array, cov: np.array, num: int,
                  gen: np.random.Generator = np.random.default_rng()) -> np.array:

    # TODO: check size and dimension of mean, cov are correct
    return gen.multivariate_normal(mean, cov, num, check_valid='raise')


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


def toy_example():
    mu_1, mu_2 = np.array([1, -1]), np.array([-2, 0])
    sigma_1, sigma_2 = np.array([[2, 1], [1, 2]]), np.array([[2, 1], [1, 2]])

    mu_3, mu_4 = np.array([-4, 0]), np.array([3, 0])
    sigma_3, sigma_4 = np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])

    n = 40

    d1 = sample_normal(mu_1, sigma_1, n)
    d2 = sample_normal(mu_2, sigma_2, n)
    d3 = sample_normal(mu_3, sigma_3, n)
    d4 = sample_normal(mu_4, sigma_4, n)

    model = LinearRegression(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), 0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # distribution of points
    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    M = ot.dist(d1, d3)
    G0 = ot.emd(a, b, M)

    ot.plot.plot2D_samples_mat(d1, d3, G0, c=[.5, .5, 1])
    plt.scatter(d1[:, 0], d1[:, 1])
    plt.scatter(d3[:, 0], d3[:, 1])

    # distribution of points
    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    M = ot.dist(d2, d4)
    G0 = ot.emd(a, b, M)

    ot.plot.plot2D_samples_mat(d2, d4, G0, c=[.5, .5, 1])
    plt.scatter(d2[:, 0], d2[:, 1])
    plt.scatter(d4[:, 0], d4[:, 1])

    plt.show()


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,)), ]
    )

    # Load training and testing datasets.
    train_mnist = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )

    test_mnist = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )
    train_emnist = torchvision.datasets.EMNIST(
        './data', 'digits', train=True, download=True, transform=transform
    )
    test_emnist = torchvision.datasets.EMNIST(
        './data', 'digits', train=False, download=True, transform=transform
    )

    # Visualize MNIST and EMNIST datasets.
    arr_mnist = train_mnist.data.numpy().reshape(train_mnist.data.shape[0], -1)
    theta_mnist, pi_mnist = mle_estimator(arr_mnist, train_mnist.targets)
    plot_mle(theta_mnist.T)

    arr_emnist = train_emnist.data.numpy().reshape(train_emnist.data.shape[0], -1)
    theta_emnist, pi_emnist = mle_estimator(arr_emnist, train_emnist.targets)
    plot_mle(theta_emnist.T)

    # Create loaders.
    dl_train_mnist = torch.utils.data.DataLoader(train_mnist, batch_size=128, shuffle=True)
    dl_test_mnist = torch.utils.data.DataLoader(test_mnist, batch_size=64, shuffle=False)
    dl_train_emnist = torch.utils.data.DataLoader(train_emnist, batch_size=128, shuffle=True)
    dl_test_emnist = torch.utils.data.DataLoader(test_emnist, batch_size=64, shuffle=False)

    # Train model.
    model = FullyConnectedNetwork()
    history = train_model(model, dl_train_mnist, dl_test_mnist, epochs=20)
    plot_accuracy(history)

    # Load (optimal) trained model.
    model = torch.load('models/FullyConnectedNetwork/epoch-17.pt')
    acc = test_model(model, dl_test_emnist)

    Xs = np.reshape(arr_emnist[0, :], (28 * 28, 1))

    best = [np.inf, -1]
    for n in range(theta_mnist.shape[1]):
        Xt = np.reshape(theta_mnist[:, n], (28 * 28, 1))

        # EMDTransport
        ot_emd = ot.da.EMDTransport()
        ot_emd.fit(Xs=Xs, Xt=Xt)
        transp_Xs_emd = ot_emd.transform(Xs=Xs)
        Image_emd = minmax(np.reshape(transp_Xs_emd, (28, 28)))

        # SinkhornTransport
        ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-1)
        ot_sinkhorn.fit(Xs=Xs, Xt=Xt)
        transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=Xs)
        Image_sinkhorn = minmax(np.reshape(transp_Xs_sinkhorn, (28, 28)))

        # MappingTransport (linear)
        ot_mapping_linear = ot.da.MappingTransport(
            mu=1e0, eta=1e-8, bias=True, max_iter=20, verbose=True)
        ot_mapping_linear.fit(Xs=Xs, Xt=Xt)
        X1tl = ot_mapping_linear.transform(Xs=Xs)
        Image_mapping_linear = minmax(np.reshape(X1tl, (28, 28)))

        # MappingTransport (Gaussian)
        ot_mapping_gaussian = ot.da.MappingTransport(
            mu=1e0, eta=1e-2, sigma=1, bias=False, max_iter=10, verbose=True)
        ot_mapping_gaussian.fit(Xs=Xs, Xt=Xt)
        X1tn = ot_mapping_gaussian.transform(Xs=Xs)  # use the estimated mapping
        Image_mapping_gaussian = minmax(np.reshape(X1tn, (28, 28)))

        # Visualize optimized transport functions.
        plt.figure(2, figsize=(10, 5))

        plt.subplot(2, 3, 1)
        plt.imshow(np.reshape(Xs, (28, 28)))
        plt.axis('off')
        plt.title('Im. 1')

        plt.subplot(2, 3, 4)
        plt.imshow(np.reshape(Xt, (28, 28)))
        plt.axis('off')
        plt.title('Im. 2')

        plt.subplot(2, 3, 2)
        plt.imshow(Image_emd)
        plt.axis('off')
        plt.title('EmdTransport')

        plt.subplot(2, 3, 5)
        plt.imshow(Image_sinkhorn)
        plt.axis('off')
        plt.title('SinkhornTransport')

        plt.subplot(2, 3, 3)
        plt.imshow(Image_mapping_linear)
        plt.axis('off')
        plt.title('MappingTransport (linear)')

        plt.subplot(2, 3, 6)
        plt.imshow(Image_mapping_gaussian)
        plt.axis('off')
        plt.title('MappingTransport (Gaussian)')
        plt.tight_layout()

        plt.show()

    """ geomloss is too slow to be used """

    # tensor_mnist = torch.from_numpy(arr_mnist).type(torch.FloatTensor)
    # tensor_emnist = torch.from_numpy(arr_emnist[: arr_mnist.shape[0], :]).type(torch.FloatTensor)

    # Loss = geomloss.SamplesLoss('sinkhorn', p=2, blur=0.05, scaling=0.8)
    # Wass_xy = Loss(tensor_emnist, tensor_mnist)

    """ ot is too slow to be used """

    # ot_emd = ot.da.EMDTransport()
    # ot_emd.fit(Xs=arr_emnist, Xt=arr_mnist)

    # ot_sinkhorn = ot.da.SinkhornTransport(reg_e=.01)
    # ot_sinkhorn.fit(Xs=arr_emnist, Xt=arr_mnist)
    #
    # ot_emd_laplace = ot.da.EMDLaplaceTransport(reg_lap=100, reg_src=1)
    # ot_emd_laplace.fit(Xs=arr_emnist, Xt=arr_mnist)

    print('hello world!')

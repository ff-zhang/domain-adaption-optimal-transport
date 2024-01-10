import pickle

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from model import FullyConnectedNetwork, test_model, train_model
from utils import fit_transports, minmax, mle_estimator, plot_accuracy, plot_mle


def visualize_image_transport(Xs: np.array, Xt: np.array) -> None:
    Xs = np.reshape(Xs, newshape=(1, Xs.shape[0])) / (np.sum(Xs) * 10000)
    Xt = np.reshape(Xt, newshape=(1, Xt.shape[0])) / (np.sum(Xt) * 10000)

    ot_emd, ot_sinkhorn, ot_mapping_linear, ot_mapping_gaussian = fit_transports(Xs, Xt)

    # Images of output from transport functions.
    transp_Xs_emd = ot_emd.transform(Xs=Xs)
    Image_emd = minmax(np.reshape(transp_Xs_emd, (28, 28)))

    transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=Xs)
    Image_sinkhorn = minmax(np.reshape(transp_Xs_sinkhorn, (28, 28)))

    X1tl = ot_mapping_linear.transform(Xs=Xs)
    Image_mapping_linear = minmax(np.reshape(X1tl, (28, 28)))

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

    # Load the (optimal) trained model.
    model = torch.load('models/FullyConnectedNetwork/epoch-17.pt')
    acc = test_model(model, dl_test_emnist)

    for n in range(theta_mnist.shape[1]):
        visualize_image_transport(arr_emnist[0, :], theta_mnist[:, n])

    # Normalize both the source and targets datasets.
    Xs = arr_emnist / (10000 * np.sum(arr_emnist, axis=1, keepdims=True))
    ys = train_emnist.targets.numpy()

    theta_mnist = np.reshape(theta_mnist, newshape=(theta_mnist.shape[1], theta_mnist.shape[0]))
    Xt = theta_mnist / (10000 * np.sum(theta_mnist, axis=0, keepdims=True))
    yt = np.array([i for i in range(10)])

    ot_sinkhorn = fit_transports(Xs, ys, Xt, yt, function=['sinkhorn'])
    with open('models/ot_sinkhorn.pkl', 'wb') as f:
        pickle.dump(ot_sinkhorn, f)

    print('hello world!')

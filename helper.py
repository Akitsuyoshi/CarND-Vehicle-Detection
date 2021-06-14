import matplotlib.pyplot as plt
import numpy as np


def plot_images(imgs, columns=2, figsize=(25, 10)):
    plt.figure(figsize=figsize)
    for i, img in enumerate(imgs):
        plt.subplot(len(imgs) / columns + 1, columns, i + 1)
        plt.imshow(img)


def color_hist(img, nbins=32, bins_range=(0, 256)):
    rhist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    bins_edges = rhist[1]
    bin_centers = (bins_edges[1:] + bins_edges[0: len(bins_edges) - 1]) / 2

    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))

    return rhist, ghist, bhist, bin_centers, hist_features

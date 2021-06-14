import matplotlib.pyplot as plt
import numpy as np
import cv2


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


def bin_spatial(img, color_space='RGB', size=(32, 32)):
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_img = np.copy(img)

    return cv2.resize(feature_img, size).ravel()

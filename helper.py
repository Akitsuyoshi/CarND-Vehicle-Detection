import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.image as mpimg


def plot_images(imgs, titles, columns=2, figsize=(25, 10)):
    plt.figure(figsize=figsize)
    for i, img in enumerate(imgs):
        plt.subplot(len(imgs) / columns + 1, columns, i + 1)
        if len(titles) > 0:
            plt.title(titles[i])
        plt.imshow(img)


def color_hist(img, nbins=32, bins_range=(0, 256)):
    rhist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    bins_edges = rhist[1]
    bin_centers = (bins_edges[1:] + bins_edges[0: len(bins_edges) - 1]) / 2

    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))

    return rhist, ghist, bhist, bin_centers, hist_features


def convert_image_color(img, color_space):
    feature_img = np.copy(img)
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

    return feature_img


def bin_spatial(img, color_space='RGB', size=(32, 32)):
    feature_img = convert_image_color(img, color_space=color_space)
    return cv2.resize(feature_img, size).ravel()


def get_hog_features(img, orient, pix_per_cell, cell_per_block, visalise=True):
    return hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block),
               block_norm='L2-Hys',
               transform_sqrt=False,
               visualise=visalise,
               feature_vector=True,
               )


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256)):
    features = []
    for img_path in imgs:
        img = mpimg.imread(img_path)
        feature_img = convert_image_color(img, color_space)

        spatial_features = bin_spatial(feature_img, size=spatial_size)
        _, _, _, _, hist_features = color_hist(feature_img, nbins=hist_bins,
                                               bins_range=hist_range)
        hog_features = get_hog_features(
            feature_img[:, :, 0],
            orient=9, pix_per_cell=8,
            cell_per_block=2,
            visalise=False
        )
        features.append(np.concatenate(
            (spatial_features, hist_features, hog_features)
        ))

    return features

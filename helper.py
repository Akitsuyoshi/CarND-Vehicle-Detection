import matplotlib.pyplot as plt


def plot_images(imgs, columns=2, figsize=(15, 5)):
    plt.figure(figsize=figsize)
    for i, img in enumerate(imgs):
        plt.subplot(len(imgs) / columns + 1, columns, i + 1)
        plt.imshow(img)

#!/usr/bin/env python

from colorsys import hsv_to_rgb
from math import sqrt, ceil
import matplotlib.colors
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
import os
from PIL import Image
from scipy.cluster.vq import kmeans, whiten
import sys


def background(size, color):
    color = [int(c * 255) for c in color]
    bg = np.array([[color]])
    bg = np.repeat(bg, size, 0)
    bg = np.repeat(bg, size, 1)
    return bg


def rgb(red, green, blue):
    return (red / 255.0, green / 255.0, blue / 255.0)


def get_stdev(array):
    return np.std(array, axis=0)


def dist(a, b):
    (xa, ya, za) = a
    (xb, yb, zb) = b
    return sqrt((xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2)


def save_file(fig, size, filename):
    bgcolor = (1.0, 1.0, 1.0)
    dpi = 100.0
    fig.set_size_inches(size / dpi, size / dpi)
    plt.savefig(filename, facecolor=bgcolor, dpi=dpi)
    print "Analysis saved as", filename


def prep_figure(size):
    bgcolor = (1.0, 1.0, 1.0)
    fig = plt.figure(facecolor=(0.5, 0.5, 0.5), linewidth=0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    bg = background(size, bgcolor)
    ax.imshow(bg.astype('uint8'), origin='lower')
    return fig, ax


def analyze(filename, num_means, rounds):
    """
    Returns a tuple of two objects:
      * A list of the means in the form [(h, s, v), ...].  Each of the
        (h, s, v) values are in the range [0, 1].
      * A list of the same length containing the number of pixels that
        are closest to the mean at the same index in the first list.
    """

    # open the image
    current_dir = os.path.dirname(os.path.realpath(__file__))
    img = Image.open(os.path.join(current_dir, filename))

    # load pixels into array
    flat_img = np.asarray(img)

    # convert from rgb to hsv (all values in range [0, 1])
    flat_img = np.apply_along_axis(
        lambda a: (a[0] / 255.0, a[1] / 255.0, a[2] / 255.0), 2, flat_img)
    flat_img = matplotlib.colors.rgb_to_hsv(flat_img)

    # reshape to an Nx3 array
    img = np.reshape(flat_img, (len(flat_img) * len(flat_img[0]), 3))

    # perform k-means clustering
    stdev = get_stdev(img)
    whitened = whiten(img)
    means, _ = kmeans(whitened, num_means, iter=rounds)
    unwhitened = means * stdev

    unwhitened = map(tuple, unwhitened)
    unwhitened.sort()

    # count the number of pixels that are closest to each centroid
    match_counts = [0] * len(unwhitened)
    for i, row in enumerate(flat_img):
        for a in row:
            distances = [dist(a, b) for b in unwhitened]
            min_index = distances.index(min(distances))
            match_counts[min_index] += 1

    return unwhitened, match_counts


def draw_color_patches(means, match_counts, size, ax):
    # draw rectangles for each centroid
    max_count = max(match_counts)
    num_rows = int(ceil(sqrt(len(means))))
    width = size / num_rows
    height = size / num_rows
    for i, mean in enumerate(means):
        rgb_mean = hsv_to_rgb(*mean)
        rgb_mean = map(lambda x: x * 256.0, rgb_mean)
        x_coord = width * (i % num_rows)
        y_coord = height * ((num_rows - 1) - (i // num_rows))

        # make the rectangle length proportional to the number of pixels
        # that are closest to this centroid
        count = match_counts[i]
        count_ratio = count / float(max_count)
        adjusted_size = (width * 0.9) * count_ratio

        rect_coords = (x_coord, y_coord + (width * 0.1))
        ax.add_patch(Rectangle(rect_coords, adjusted_size, adjusted_size,
                               facecolor=rgb(*rgb_mean), edgecolor="none"))

        # add h,s,v label in ranges [0, 360], [0, 100], [0, 100]
        adjusted_hsv = (mean[0] * 360.0, mean[1] * 100.0, mean[2] * 100.0)
        ax.text(x_coord, y_coord, ",".join("%d" % int(x) for x in adjusted_hsv))


def parse_options():
    parser = OptionParser()
    parser.add_option('-k', '--kmeans', type='int', default=36,
                      help='number of means for clustering [default: %default]')
    parser.add_option('-r', '--rounds', type='int', default=10,
                      help='number of clustering rounds; higher values increase accuracy [default: %default]')
    parser.add_option('-s', '--size', type='int', default=1000,
                      help='Output image size in pixels (NxN) [default: %default]')
    parser.add_option('-f', '--filename', default=None,
                      help='Output file')

    options, args = parser.parse_args()

    if options.kmeans <= 0:
        print >>sys.stderr, "--kmeans must have a positive value"
        sys.exit(1)
    if options.rounds <= 0:
        print >>sys.stderr, "--rounds must have a positive value"
        sys.exit(1)
    if options.size <= 0:
        print >>sys.stderr, "--size must have a positive value"
        sys.exit(1)
    if len(args) > 1:
        print >>sys.stderr, "Expected one argument, but got %d" % len(args)
        sys.exit(1)

    return options, args


def create_outfile_name(filename):
    dot_index = filename.rindex('.')
    filename = filename[:dot_index] + "-analysis" + filename[dot_index:]
    current_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(current_dir, filename)


def main():
    options, args = parse_options()
    means, match_counts = analyze(args[0], options.kmeans, options.rounds)

    figure, ax = prep_figure(options.size)
    draw_color_patches(means, match_counts, options.size, ax)

    if options.filename is None:
        outfile = create_outfile_name(args[0])
    else:
        outfile = options.filename

    save_file(figure, options.size, outfile)


if __name__ == "__main__":
    main()

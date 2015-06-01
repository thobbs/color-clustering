color-clustering
================
A python script to analyze the colors in an image through K-means clustering.
The script creates an output file displaying the main colors used with sizes
proportional to their use and HSV values for each color.

The HSV values in the output use the following ranges:

* Hue: [0, 360]
* Saturation: [0, 100]
* Value: [0, 100]

Example Output
--------------
This is an analysis of J.M.W. Turner's "Ulysses deriding Polyphemus"

.. image:: https://raw.github.com/thobbs/color-clustering/master/turner_analysis.png

Usage
-----
``./color_clustering.py <imagefile> [--kmeans=32] [--rounds=10] [--size=1000] [--file]``

You can use the ``--help`` option for more details.

Dependencies
------------
* matplotlib
* numpy
* scipy
* PIL

License
-------
Apache 2.0 License

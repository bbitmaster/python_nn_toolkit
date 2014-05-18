# This file is adapted from http://g.sweyla.com/blog/2012/mnist-numpy/

import os, struct
from array import array as pyarray
from numpy import append, array, float_, zeros

import urllib2
import StringIO
import gzip

def read(digits, dataset = "training", path = "data", download=True):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """
    try:
        if dataset is "training":
            fname_img = os.path.join(path, 'train-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
        elif dataset is "testing":
            fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
        else:
            raise ValueError, "dataset must be 'testing' or 'training'"

        flbl = open(fname_lbl, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        lbl = pyarray("b", flbl.read())
        flbl.close()

        fimg = open(fname_img, 'rb')
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = pyarray("B", fimg.read())
        fimg.close()
        
    except Exception as ex:
        print ex
        print "Downloading files..."
        path = "http://yann.lecun.com/exdb/mnist/"
        if dataset is "training":
            fname_img = path + 'train-images-idx3-ubyte.gz'
            fname_lbl = path + 'train-labels-idx1-ubyte.gz'
        elif dataset is "testing":
            fname_img = path + 't10k-images-idx3-ubyte.gz'
            fname_lbl = path + 't10k-labels-idx1-ubyte.gz'
        else:
            raise ValueError, "dataset must be 'testing' or 'training'"

        response = urllib2.urlopen(fname_img)
        compressedFile = StringIO.StringIO(response.read())
        decompressedFile = gzip.GzipFile(fileobj=compressedFile)
        magic_nr, size, rows, cols = struct.unpack(">IIII", decompressedFile.read(16))
        img = pyarray("b", decompressedFile.read())
        
        response = urllib2.urlopen(fname_lbl)
        compressedFile = StringIO.StringIO(response.read())
        decompressedFile = gzip.GzipFile(fileobj=compressedFile)
        magic_nr, size = struct.unpack(">II", decompressedFile.read(8))
        lbl = pyarray("b", decompressedFile.read())
    ind = [ k for k in xrange(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows * cols), dtype=float_)
    labels = zeros((N, 10), dtype=float_)
    for i in xrange(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows * cols))
        labels[i,lbl[ind[i]]] = 1.0

    return images, labels

# Base scientific imports
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection # Necessary for time coloration on particle tracks.

# Numpy Imports
from numpy.linalg import norm
from numpy.linalg import det
from numpy.linalg import inv

# Scipy imports
from scipy.ndimage.filters import convolve
from scipy.optimize import fmin
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

# skimage imports
from skimage.feature import blob_log
from skimage import io
from skimage.segmentation import random_walker
from skimage.exposure import rescale_intensity
from skimage.exposure import equalize_hist
from skimage import img_as_float
from skimage.morphology import reconstruction
from skimage.morphology import opening
from skimage.morphology import white_tophat
from skimage.morphology import square
from skimage.morphology import disk

###############################################################################################################################################################################################
################################################################################## Helper Functions ###########################################################################################
###############################################################################################################################################################################################

# Recreated convolution method from mathworks. Taken from https://stackoverflow.com/questions/3731093/is-there-a-python-equivalent-of-matlabs-conv2-function
def conv2(x,y,mode='same'):
	"""
	Emulate the function conv2 from Mathworks.

	Usage:

	z = conv2(x,y,mode='same')

	TODO: 
	 - Support other modes than 'same' (see conv2.m)
	"""

	if not(mode == 'same'):
		raise Exception("Mode not supported")

	# Add singleton dimensions
	if (len(x.shape) < len(y.shape)):
		dim = x.shape
		for i in range(len(x.shape),len(y.shape)):
			dim = (1,) + dim
		x = x.reshape(dim)
	elif (len(y.shape) < len(x.shape)):
		dim = y.shape
		for i in range(len(y.shape),len(x.shape)):
			dim = (1,) + dim
		y = y.reshape(dim)

	origin = ()

	# Apparently, the origin must be set in a special way to reproduce
	# the results of scipy.signal.convolve and Matlab
	for i in range(len(x.shape)):
		if ( (x.shape[i] - y.shape[i]) % 2 == 0 and
			 x.shape[i] > 1 and
			 y.shape[i] > 1):
			origin = origin + (-1,)
		else:
			origin = origin + (0,)

	z = convolve(x,y, mode='constant', origin=origin)

	return z

def lsradialcenterfit(m,b,w):
	"""
	Least squares solution to determine radials symmetry center
	"""

	wm2p1 = w/(m * m+1)
	sw = np.sum(np.sum(wm2p1))
	smmw = np.sum(np.sum(m*m*wm2p1))
	smw = np.sum(np.sum(m*wm2p1))
	smbw = np.sum(np.sum(m*b*wm2p1))
	sbw = np.sum(np.sum(b*wm2p1))
	det = smw * smw - smmw*sw
	xc = (smbw*sw - smw*sbw)/det;
	yc = (smbw*smw - smmw*sbw)/det

	return xc, yc


def rand_jitter(arr):
	# Add a small amount of noise to keep trajectories from overlapping
	# TAken from https://stackoverflow.com/questions/8671808/matplotlib-avoiding-overlapping-datapoints-in-a-scatter-dot-beeswarm-plot
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev
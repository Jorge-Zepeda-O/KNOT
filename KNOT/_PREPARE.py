#%% --- IMPORTS --- %%#
### External ###
import numpy				as np
import numpy.fft			as npf
import scipy.special		as sp
import pyfftw				as fftw
import matplotlib.pyplot	as plt
from mpl_toolkits.mplot3d	import Axes3D

import scipy.io			as spi

import time

### Internal ###
from __ENUM			import FILEFORMAT		as FMT
import __OPERATION	as OP
import __VISUALS	as VIS

from _INITIALIZE	import _MeshLat

import USER

#%% --- EXPOSED METHODS --- %%#
def RUN(scope, *, code='', update=False, visual=False):
	## Update query ##
	if(update or not OP._CheckFile('%s\\%s_prep.tif' % (code, code))):
		# Pre-process the movie and the kernel, obtaining the local threshold #
		img_, ker_, eps_ = _Preprocess2(code, scope.img, scope.ker)

		# Ensure that all values are non-negative #
		if(USER.KER_T > 1):
			img_ = np.maximum(img_, 0)
			ker_ = np.maximum(ker_, 0)
			eps_ = np.maximum(eps_, 0)

		# Save the processed image, kernel, and local threshold #
		OP._SaveMov(img_, '%s\\%s_prep' % (code, code))	# Processed Movie	#
		OP._SaveKer(ker_, '%s\\%s_ker' % (code, code))	# Processed Kernel	#
		OP._SaveMov(eps_, '%s\\%s_eps' % (code, code))	# Local Threshold	#
	else:
		# Check if the movies exist #
		if(OP._CheckFile('%s\\%s_prep.tif' % (code, code))):
			# Load the processed image, kernel, and local threshold #
			img_ = OP._LoadMov('%s\\%s_prep' % (code, code))	# Processed Movie	#
			ker_ = OP._LoadKer('%s\\%s_ker' % (code, code))	# Processed Kernel	#
			eps_ = OP._LoadMov('%s\\%s_eps' % (code, code))	# Local Threshold	#
		else:
			# Just stuff some zeros in there for now #
			img_ = np.zeros_like(scope.img)
			ker_ = np.zeros_like(scope.ker)
			eps_ = np.zeros_like(scope.img)

	## Visualization query ##
	if(visual):
		f = 0
		tru = OP._LoadTruth(code)
		pts = np.zeros([0, 3]);
		for p in range(len(tru)):
			if(f in tru[p].frm):
				idx = np.nonzero(f == tru[p].frm)[0]
				pts = np.concatenate((pts, tru[p].res[idx,:]), axis=0)

		# Movie #
		VIS._VisImg(img_[f,0,:,:], 550, 50, pts)

		# Kernel #		
		if((USER.KER_Z > 1) and (USER.KER_T == 1)):
			VIS._VisImg(ker_[0,0,:,:], 100, 50)
			VIS._VisImg(ker_[0,(USER.KER_Z)//4,:,:], 100, 550)
			VIS._VisImg(ker_[0,(2*USER.KER_Z)//4,:,:], 550, 550)
			VIS._VisImg(ker_[0,(3*USER.KER_Z)//4,:,:], 1000, 550)
			VIS._VisImg(ker_[0,-1,:,:], 1000, 50)
		elif((USER.KER_Z == 1) and (USER.KER_T > 1)):
			VIS._VisImg(ker_[0,0,:,:], 100, 50)
			VIS._VisImg(ker_[(USER.KER_T)//4,0,:,:], 100, 550)
			VIS._VisImg(ker_[(2*USER.KER_T)//4,0,:,:], 550, 550)
			VIS._VisImg(ker_[(3*USER.KER_T)//4,0,:,:], 1000, 550)
			VIS._VisImg(ker_[-1,0,:,:], 1000, 50)

		# Local Threshold #
		VIS._VisImg(eps_[f,0,:,:], 1450, 50)
		VIS._VisImg(((img_ - eps_) * (img_ > eps_))[f,0,:,:], 1450, 550, pts)

		plt.show()

	## Output ##
	return img_, ker_, eps_

#%% --- STATIC METHODS --- %%#
def _ApplyFilter(x, h, *, axes=(-2,-1)):
	"""
	Applies the filter `h` to the object `x` using the FFTW as follows:
	`Y = H . X`
	"""
	## Initialize ##
	# Get the complex shape right #
	cs_x = np.array(np.shape(x))
	cs_h = np.array(np.shape(h))
	cs_x[-1] = cs_x[-1]//2 + 1
	cs_h[-1] = cs_h[-1]//2 + 1

	# Create the FFTW objects #
	x_ = fftw.zeros_aligned(np.shape(x), dtype='float32')
	h_ = fftw.zeros_aligned(np.shape(h), dtype='float32')
	y_ = fftw.zeros_aligned(np.shape(x), dtype='float32')

	X = fftw.zeros_aligned(cs_x, dtype='complex64')
	H = fftw.zeros_aligned(cs_h, dtype='complex64')
	Y = fftw.zeros_aligned(cs_x, dtype='complex64')

	FT_X = fftw.FFTW(x_, X, axes=axes)
	FT_H = fftw.FFTW(h_, H, axes=axes)
	RT_y = fftw.FFTW(Y, y_, axes=axes, direction='FFTW_BACKWARD')

	## Filter ##
	x_[...] = x
	h_[...] = h
	FT_X()
	FT_H()

	Y[...] = X * H
	RT_y()

	## Output ##
	return npf.ifftshift(np.real_if_close(y_), axes=axes)
def _GetFilters(sz, r_bg=USER.PRE_BG, r_ns=USER.PRE_NS, r_lt=USER.PRE_LT):
	"""
	Creates the 2D  filters used for processing background subtraction, denoising, and local threshold.
	"""
	## Initialization ##
	# Meshes #
	xx, yy = np.meshgrid(*_MeshLat(*sz, shift=True, scale=True))
	rr2 = (xx**2 + yy**2)

	## Filter Construction ##
	h_bg = np.exp(-1/2 * rr2/r_bg**2)	# Background Subtraction - Gaussian high pass	#
	h_ns = 1/(1 + rr2/r_ns**2)			# Noise Suppression - Matched filter			#
	h_lt = np.exp(-1/2 * rr2/r_lt**2)	# Local Threshold - Gaussian low pass			#

	## Normalization ##
	h_bg /= np.sum(h_bg)
	h_ns /= np.sum(h_ns)
	h_lt /= np.sum(h_lt)

	## Output ##
	return h_bg, h_ns, h_lt

#%% --- METHODS --- %%#
def _Preprocess2(code, img, ker):
	stpwch = time.time()
	VIS._ProgressBar(0, 20, prefix='(%s):\t%8.3f sec' % (code, time.time() - stpwch), suffix='Initializing...')

	## Initialization ##
	sz = np.shape(img)[2:]	# Get the true size of the image #

	# Inflated size #
	sz_off = int(np.ceil(2*np.max([USER.PRE_BG, USER.PRE_NS, USER.PRE_LT])/USER.RES[0]))	# 2 sigma #
	sz_ = (sz[0] + 2*sz_off, sz[1] + 2*sz_off)

	# Construct filters for both sizes - ker will use the original sizes #
	h_bg, h_ns, h_lt = _GetFilters(sz)
	h_bg_, h_ns_, h_lt_ = _GetFilters(sz_)

	# Place the image in the inflated array #
	im = np.zeros([*np.shape(img)[:2], *sz_])
	im[:,:,sz_off:-sz_off,:][:,:,:,sz_off:-sz_off] = img

	# Smooth out the edges #
	for i in range(sz_off+1):
		fwd = (np.sin( (np.pi/2) * (i/sz_off) ) + 1)/2;
		rwd = 1 - fwd

		im[:,:,i,:] = im[:,:,sz_off,:] * fwd + im[:,:,-sz_off-2,:] * rwd
		im[:,:,:,i] = im[:,:,:,sz_off] * fwd + im[:,:,:,-sz_off-2] * rwd
		im[:,:,-i-1,:] = im[:,:,sz_off,:] * rwd + im[:,:,-sz_off-2,:] * fwd
		im[:,:,:,-i-1] = im[:,:,:,sz_off] * rwd + im[:,:,:,-sz_off-2] * fwd

	## Filtering ##
	# Kernel #
	VIS._ProgressBar(1, 20, prefix='(%s):\t%8.3f sec' % (code, time.time() - stpwch), suffix='Filtering Kernel...')

	ker_bg = ker - _ApplyFilter(ker, h_bg)
	ker_ns = _ApplyFilter(ker_bg, h_ns)
	ker_ = ker_ns * np.ptp(ker)/np.ptp(ker_ns)	# Renormalize #

	# Inflated Image #
	VIS._ProgressBar(2, 20, prefix='(%s):\t%8.3f sec' % (code, time.time() - stpwch), suffix='BG Subtraction...  ')

	img_bg = im - _ApplyFilter(im, h_bg_)
	VIS._ProgressBar(7, 20, prefix='(%s):\t%8.3f sec' % (code, time.time() - stpwch), suffix='Noise Suppression...')

	img_ns = _ApplyFilter(img_bg, h_ns_)
	VIS._ProgressBar(12, 20, prefix='(%s):\t%8.3f sec' % (code, time.time() - stpwch), suffix='Temporal Smoothing...')

	if((np.shape(img)[0] > USER.PRE_TS//10 + 1) and (USER.PRE_TS > 0)):
		for f in range(np.shape(img)[0]):			# Temporal smoothing #
			lft = np.maximum(0, f - USER.PRE_TS//2)
			rgt = np.minimum(f + USER.PRE_TS//2, np.shape(img)[0])
			img_ts = np.mean(img_ns[lft:rgt,...], axis=0)
			img_ns[f] = img_ns[f] - img_ts * np.ptp(img_ts)/np.ptp(img_ns[f])
			VIS._ProgressBar(12 + 4*f/np.shape(img)[0], 20, prefix='(%s):\t%8.3f sec' % (code, time.time() - stpwch), suffix='Temporal Smoothing...')

	img_ = img_ns * np.ptp(img)/np.ptp(img_ns)	# Renormalize #
	img_ = img_[:,:,sz_off:-sz_off,:][:,:,:,sz_off:-sz_off]	# Truncate #

	## Threshold ##
	VIS._ProgressBar(16, 20, prefix='(%s):\t%8.3f sec' % (code, time.time() - stpwch), suffix='Obtaining Threshold...')

	eps_mu = _ApplyFilter(im.astype(np.float32), h_lt_)
	eps_var = _ApplyFilter(im.astype(np.float32)**2, h_lt_) - eps_mu**2
	
	# Compare statistics of the variance and the mean #
	rat_min = np.minimum(np.min(eps_var)/np.min(eps_mu), 1)		
	rat_avg = np.minimum(np.mean(eps_var)/np.mean(eps_mu), 1)
	rat_max = np.minimum(np.max(eps_var)/np.max(eps_mu), 1)

	# Use to weight the mean #
	eps_exp = np.sqrt(rat_min**2 + rat_avg**2 + rat_max**2)
	eps_global = 0.0#np.mean(img)**2/np.var(img)
	eps_ = np.sqrt(np.maximum(eps_mu**eps_exp - 1, 0) + 2*eps_var + eps_global)
	eps_ = eps_[:,:,sz_off:-sz_off,:][:,:,:,sz_off:-sz_off]	# Truncate #

	## Output ##
	prefix = '(%s):\t%8.3f sec' % (code, time.time() - stpwch)
	VIS._ProgressBar(20, 20, prefix=prefix, suffix='Finished!             ')

	return img_, ker_, eps_
def _Preprocess(img, ker, apr):
	## Initialization ##
	sz = np.shape(img)[2:]
	h_bg, h_ns, h_lt = _GetFilters(sz)
	#xx, yy = np.meshgrid(*_MeshLat(*sz, shift=True))

	## Filtering ##
	# Kernel #
	ker_bg = ker - _ApplyFilter(ker, h_bg)
	ker_ns = _ApplyFilter(ker_bg, h_ns)
	ker_ = ker_ns * np.ptp(ker)/np.ptp(ker_ns)	# Renormalize #

	# Image #
	img_bg = img - _ApplyFilter(img, h_bg)
	img_ns = _ApplyFilter(img_bg, h_ns)
	img_ = img_ns * np.max(img)/np.max(img_ns)	# Renormalize #

	## Threshold ##
	eps_mu = _ApplyFilter(img.astype(np.float32), h_lt)
	eps_var = _ApplyFilter(img.astype(np.float32)**2, h_lt) - eps_mu**2
	
	# Compare statistics of the variance and the mean #
	rat_min = np.minimum(np.min(eps_var)/np.min(eps_mu), 1)		
	rat_avg = np.minimum(np.mean(eps_var)/np.mean(eps_mu), 1)
	rat_max = np.minimum(np.max(eps_var)/np.max(eps_mu), 1)

	# Use to weight the mean #
	eps_exp = np.sqrt(rat_min**2 + rat_avg**2 + rat_max**2)
	eps_global = 0#np.mean(img)**2/np.var(img)
	eps_ = np.sqrt(np.maximum(eps_mu**eps_exp - 1, 0) + 2*eps_var + eps_global)

	## Determine where signal ##
	sig = img_ > np.max(eps_, axis=(0,1))
	# Perform a Bayesian test to see if this frame "belongs" with the rest of the others. This will check if a single frame just has no signal, versus an overall Low-SNR image

	## Output ##
	return img_, ker_, eps_

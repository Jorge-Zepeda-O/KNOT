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

from _INITIALIZE	import _MeshLat, _MeshMeta
from _SEGMENT		import PointCloud, _CloudThr, _Separate

import USER

#%% --- EXPOSED METHODS --- %%#
def RUN(img, ker, eps, *, code='', update=False, visual=False):
	## Update query ##
	if(not OP._CheckFile('%s\\%s_pts.json' % (code, code)) or update):
		# Perform ADMM and recover the point clouds #
		pos, wgt = _Recover(img, ker, eps, code=code, vis=visual)

		# Save data #
		pts = {'pos':[pos[f].tolist() for f in range(len(pos))], 
			   'wgt':[wgt[f].tolist() for f in range(len(wgt))]}
		OP._SaveJSON(pts, '%s\\%s_pts' % (code, code))
	else:
		# Load in the data #
		data = OP._LoadJSON('%s\\%s_pts' % (code, code))
		pos = [np.array(data['pos'][f]) for f in range(len(data['pos']))]
		wgt = [np.array(data['wgt'][f]) for f in range(len(data['wgt']))]

	## Visualization query ##
	if(visual):
		f = 0
		plt.figure()
		plt.imshow(img[f,0,:,:], cmap='gray')
		plt.scatter(pos[f][:,0], pos[f][:,1], s=100*wgt[f], c='r')
		plt.show()

	## Output ##
	return pos, wgt
def TEST_LIMITS(img, ker, eps, *, code='', visual=False):
	# Test runs for limitations only - 64x64! #
	## Initialize ##
	F = np.shape(img)[0]
	Z = np.shape(img)[1]
	Y = np.shape(img)[2]
	X = np.shape(img)[3]

	pos = [None] * F
	wgt = [None] * F
	H_A, H_S = _IDFilters([Z*np.shape(ker)[1], Y, X])

	tru = OP._LoadTruth(code)
	error = np.full([int(USER.REC_ITER//3), F, 3], np.nan)

	# Progress #
	stpwch = time.time()
	timers = np.zeros((F))
	t_remain = np.nan

	## Recover Emitter Positions ##
	admm = ADMM(ker)
	for f in np.arange(F):
		psi_f = np.zeros((np.shape(ker)[0], Z*np.shape(ker)[1], Y, X))
		for z in range(Z):
			pb = (code, f,F, z,Z, 0,1, 0,1, timers[f - (1 if(f > 0) else 0)], t_remain)
			zrng = [z*USER.KER_Z, (z+1)*USER.KER_Z]

			# Split the image into each plane #
			img_ = img[f,z,:,:] / eps[f,z,:,:]
			eps_ = eps[f,z,:,:] / np.max(eps)

			# Obtain the point clouds per frame #
			psi_f[:,zrng[0]:zrng[1],...], error = admm.Recover(img_, code=code, pb=pb, vis=visual, error=error)

		# Identify points in the cloud #
		Psi_f = npf.fftn(psi_f)
		psi_a = np.real_if_close(npf.ifftshift(npf.ifftn(Psi_f * H_A), axes=(-2,-1)))
		psi_s = np.real_if_close(npf.ifftshift(npf.ifftn(Psi_f * H_S), axes=(-2,-1)))

		# Determine where the smaller blur is bigger than the larger one #
		lhs = psi_a
		rhs = psi_s * (1 + 1 / eps_) + np.mean(psi_f)
		idx = np.nonzero(lhs > rhs)
		pos[f] = np.array([idx[3], idx[2], idx[1], idx[0]/USER.KER_T + f]).T
		wgt[f] = np.round(np.sqrt(psi_f**2 + ((psi_a + psi_s)/2)**2)[idx], 3)

		# Progress Display #
		timers[f] = time.time() - stpwch
		if(sum(timers > 0) > 1):
			t_remain = (F-(f+1)) * np.mean(np.diff(timers[timers > 0]))
			prefix = '(%s):\t%8.3f sec' % (code, timers[f])
			suffix = '(Remain: %5.0f sec)' % (t_remain)
			VIS._ProgressBar(f+1, F, prefix=prefix, suffix=suffix)

	#if(error is not None):
	spi.savemat(OP.FOLD_MAT + code + ' error.mat', {'error': error})
	print(code + ' done!')

#%% --- STATIC METHODS --- %%#
def _IDFilters(sz, r_apr=1*USER.RES[0], r_seg=3*USER.RES[0]):
	## Initialize ##
	mesh_x, mesh_y = _MeshLat(*(sz[2:]), shift=True, scale=True)
	mesh_z, mesh_t = _MeshMeta(ker_z=sz[0], ker_t=sz[1])

	zz, tt, yy, xx = np.meshgrid(mesh_z, mesh_t, mesh_y, mesh_x)
	rr2 = xx**2 + yy**2 + zz**2 + (4*tt)**2	# How to accomodate t (1 frame max)? #

	h_a = np.exp(-1/2 * (rr2/r_apr**2))		# Particle size			#
	h_s = np.exp(-1/2 * (rr2/r_seg**2))		# Segmentation radius	#
	
	return npf.fftn(h_a / np.sum(h_a)), npf.fftn(h_s / np.sum(h_s))

#%% --- METHODS --- %%#
def _Recover(img, ker, eps, *, code='', step=1, vis=False):
	## Initialize ##
	f0 = 0;
	F = np.shape(img)[0]
	Z = np.shape(img)[1]
	Y = np.shape(img)[2]
	X = np.shape(img)[3]
	T = np.shape(ker)[0]
	C = np.minimum(X, Y) if((not USER.REC_CHUNK) or ((X <= 128) and (Y <= 128))) else 64

	pos = [None] * F
	wgt = [None] * F
	H_A, H_S = _IDFilters([T, Z*np.shape(ker)[1], Y, X])

	tru = OP._LoadTruth(code)

	# Progress #
	stpwch = time.time()
	timers = np.zeros((F))
	t_remain = np.nan

	# Truth #
	#error = np.full([int(USER.REC_ITER//3), F], np.nan)

	## Recover Emitter Positions ##
	ker_ = ker[...,(Y-C)//2:(Y+C)//2,:][...,(X-C)//2:(X+C)//2]
	admm = ADMM(ker_)
	for f in np.arange(F, step=step):
		psi_f = np.zeros((np.shape(ker)[0], Z*np.shape(ker)[1], Y, X))
		for z in range(Z):
			zrng = [z*USER.KER_Z, (z+1)*USER.KER_Z]

			# Split the image into each plane #
			img_ = img[f+f0,z,:,:] / eps[f+f0,z,:,:]	# << ------------------------------------------------------------ #
			eps_ = eps[f+f0,z,:,:] / np.max(eps)

			# Chunk the image and obtain point clouds per frame #
			img_chunks, xrng, yrng, overlay = _Chunk(img_, C=C)
			M = np.shape(xrng)[0]
			N = np.shape(yrng)[0]
			for m in range(M):
				for n in range(N):
					pb = (code, f,F, z,Z, m,M, n,N, timers[f - (1 if(M == 1 and N == 1 and f > 0) else 0)], t_remain)
					if(np.ptp(img_chunks[n,m,...]) > 2*np.std(img_)):
						psi, _ = admm.Recover(img_chunks[n,m,...], code=code, pb=pb, vis=False)
						psi = np.fft.fftshift(psi, axes=(-2,-1))
						psi_f[:,zrng[0]:zrng[1],...][...,yrng[n,0]:yrng[n,1],:][...,xrng[m,0]:xrng[m,1]] += psi
					timers[f] = time.time() - stpwch
			psi_f[:,zrng[0]:zrng[1],...] /= np.maximum(overlay, 1)

		# Identify points in the cloud #
		Psi_f = npf.fftn(psi_f)
		psi_a = np.real_if_close(npf.ifftshift(npf.ifftn(Psi_f * H_A), axes=(-2,-1)))
		psi_s = np.real_if_close(npf.ifftshift(npf.ifftn(Psi_f * H_S), axes=(-2,-1)))

		# Determine where the smaller blur is bigger than the larger one #
		lhs = psi_a
		rhs = psi_s * (1 + 1 / eps_) + np.mean(psi_f) * (1 + (USER.KER_T > 1)) #eps_ * np.std(psi_f)/np.mean(psi_f)
		idx = np.nonzero(lhs > rhs)
		pos[f] = np.array([idx[3], idx[2], idx[1], idx[0]/USER.KER_T + f]).T
		wgt[f] = np.round(np.sqrt(psi_f**2 + ((psi_a + psi_s)/2)**2)[idx], 3)

		# Visualization #
		if(vis):
			plt.figure(figsize=(15,5))
			ax = plt.axes(position=[0,0,1/3,0.9])
			ax.imshow(img_, cmap='gray')
			ax.set_title('Input image #%i/%i' % (f+1, F+1))

			ax = plt.axes(position=[1/3,0,1/3,0.9])
			ax.imshow(np.sum(psi_f, axis=(0,1)), cmap='gray')
			ax.set_title('Deconvolution')

			ax = plt.axes(position=[2/3,0,1/3,0.9])
			ax.imshow(img_, cmap='gray')
			if(len(wgt[f]) > 0):
				ax.scatter(pos[f][:,0], pos[f][:,1], s=100*(wgt[f]/np.max(wgt[f])), c='r')
			ax.set_title('Point Cloud')

			if(USER.KER_Z > 1):
				plt.figure(figsize=(6, 6))
				ax = plt.axes(projection='3d', position=[-0.05,-0.07,1.1,1.1])
				if(len(wgt[f]) > 0):
					ax.scatter(pos[f][:,0], pos[f][:,1], pos[f][:,2], s=100*(wgt[f]/np.max(wgt[f])))
				ax.view_init(azim=30, elev=10)
				ax.set_xlim(0, np.shape(img)[3])
				ax.set_ylim(0, np.shape(img)[2])
				ax.set_zlim(0, USER.KER_Z)
			plt.show()

		# Progress Display #
		timers[f] = time.time() - stpwch
		if(sum(timers > 0) > 1):
			t_remain = (F-(f+1)) * np.mean(np.diff(timers[timers > 0]))
			prefix = '(%s):\t%8.3f sec' % (code, timers[f])
			#suffix = '(Remain: %5.0f sec)' % (t_remain)
			suffix = '(Remain: %3.0f:%2.0f:%2.0f)' % (t_remain // 3600, (t_remain % 3600) // 60, t_remain % 60)
			VIS._ProgressBar(f+1, F, prefix=prefix, suffix=suffix)

	#import scipy.io as spi
	#spi.savemat(OP.FOLD_MAT + code + ' error.mat', {'error': error})

	## Output ##
	return pos, wgt
def _CloudRep(psi, eps, *, code, step=1, vis=False):
	## Initialize ##
	F = int(np.shape(psi)[0] / USER.KER_T)
	pos = [None] * F
	wgt = [None] * F

	tru = OP._LoadTruth(code)

	H_A, H_S = _IDFilters(np.shape(psi)[1:])

	## Identify ##
	for f in np.arange(F, step=step):
		eps_f = eps[f,...] / np.max(eps)
		psi_f = psi[f*USER.KER_T:(f+1)*USER.KER_T, ...]
		Psi_f = npf.fftn(psi_f)
		psi_f = npf.ifftshift(psi_f)
		
		# Obtain the local blurring for each point found #
		psi_a = np.real_if_close(npf.ifftn(Psi_f * H_A))
		psi_s = np.real_if_close(npf.ifftn(Psi_f * H_S))

		# Determine where the smaller blur is bigger than the larger one #
		idx = np.nonzero(psi_a > psi_s + eps_f * (np.mean(psi_f) + np.std(psi_f)))# + eps_f**2 *(np.mean(psi_f) + np.std(psi_f)))
		pos[f] = np.array([idx[3], idx[2], idx[1], idx[0]/USER.KER_T + f]).T
		wgt[f] = np.round(np.sqrt(psi_f**2 + ((psi_a + psi_s)/2)**2)[idx], 3)

		if(vis):
			pts = np.zeros([0, 3]);
			for p in range(len(tru)):
				if(f in tru[p].frm):
					idx = np.nonzero(f == tru[p].frm)[0]
					pts = np.concatenate((pts, tru[p].res[idx,:]), axis=0)

			plt.figure()
			plt.gca(position=[0,0,1,1])
			plt.imshow(psi_f[0,0,:,:], cmap='gray')
			plt.scatter(pos[f][:,0], pos[f][:,1], 100*wgt[f], c='b')
			plt.plot(pts[:,0], pts[:,1], color='r', marker='o', linewidth=0, fillstyle='none')
			plt.show()

	## Output ##
	return pos, wgt
def _Chunk(img, *, C=64):
	## Initialize ##
	sz = np.shape(img)
	X = sz[1]
	Y = sz[0]

	"""
	cX = int(np.ceil((4*X)/(3*C)))
	cY = int(np.ceil((4*Y)/(3*C)))
	
	## Get the window ranges ##
	xrng = np.zeros([cX, 2], dtype=int)
	yrng = np.zeros([cY, 2], dtype=int)
	for m in range(cX):
		dx = (m*C*(2/3) + C) - X
		xmin = int(m*C*(2/3) - (dx if(dx > 0) else 0))
		xmax = int(np.minimum(m*C*(2/3) + C, X))
		xrng[m,:] = [xmin, xmax]
	for n in range(cY):
		dy = (n*C*(2/3) + C) - Y
		ymin = int(n*C*(2/3) - (dy if(dy > 0) else 0))
		ymax = int(np.minimum(n*C*(2/3) + C, Y))
		yrng[n,:] = [ymin, ymax]
	"""
	
	cX = int(np.ceil(2*X/C)-1)
	cY = int(np.ceil(2*Y/C)-1)
	
	## Get the window ranges ##
	xrng = np.zeros([cX, 2], dtype=int)
	yrng = np.zeros([cY, 2], dtype=int)
	for m in range(cX):
		dx = (m*C*(1/2) + C) - X
		xmin = int(m*C*(1/2) - (dx if(dx > 0) else 0))
		xmax = int(np.minimum(m*C*(1/2) + C, X))
		xrng[m,:] = [xmin, xmax]
	for n in range(cY):
		dy = (n*C*(1/2) + C) - Y
		ymin = int(n*C*(1/2) - (dy if(dy > 0) else 0))
		ymax = int(np.minimum(n*C*(1/2) + C, Y))
		yrng[n,:] = [ymin, ymax]
	
	## Crop the image and kernel ##
	img_ = np.zeros([cY,cX,C,C])
	overlay = np.zeros([Y,X])
	for m in range(cX):
		for n in range(cY):
			img_[n,m,...] = img[yrng[n,0]:yrng[n,1],:][:,xrng[m,0]:xrng[m,1]]
			overlay[yrng[n,0]:yrng[n,1], :][:, xrng[m,0]:xrng[m,1]] += 1

	## Output ##
	return img_, xrng, yrng, overlay

#%% --- ADMM CLASS DEFINITION --- %%#
class ADMM:
	### Constants ###
	LOC_Z = USER.KER_Z > 1
	LOC_T = USER.KER_T > 1
	D = (2 + LOC_Z + LOC_T)

	# (Accuracy, Sparsity) #
	LAMBDA 	= np.array([1, 4, 0]) + np.array([0, 0, 1])*LOC_T	# 1 4 0 (+0 +0 +1)
	MU 		= np.array([1, 20, 0]) + np.array([0, 0, 1])*LOC_T	# 1 20 0 (+4 -12 +1)

	### Constructor ###
	def __init__(self, ker):
		## pyFFTW ##
		# Determine the axes to perform the transformation over and the complex-FFT size to use #
		if(self.LOC_Z and self.LOC_T):	axes = (0, 1, 2, 3)	# (T, Z, Y, X) #
		elif(self.LOC_Z):				axes = (   1, 2, 3)	# (-, Z, Y, X) #
		elif(self.LOC_T):				axes = (0,    2, 3)	# (T, -, Y, X) #
		else:							axes = (      2, 3)	# (-, -, Y, X) #

		## Argument Transfer ##
		# Since we will never need the real-space version of the kernel, we can save just its Fourier transform.
		self.K = npf.fftn(ker, axes=axes)

		## Initialization ##
		self.sz = np.shape(ker)		# Kernel shape 				#

		# Sample sizes #
		Ls = np.size(ker[0,0,:,:])	# Lateral samples (X & Y) 	#
		Ms = np.size(ker[:,:,0,0])	# Meta-samples (Z & T)		#
		
		# Constant matrices #
		self.Nu_0 	= fftw.zeros_aligned(self.sz, dtype='complex64')	# Regularization weights #
		self.Nu_1 	= fftw.zeros_aligned(self.sz, dtype='complex64')
		self.Nu_2	= fftw.zeros_aligned(self.sz, dtype='complex64')

		self.w		= fftw.zeros_aligned(self.sz, dtype='float32')		# Weighting matrix 	#
		self.sig	= fftw.zeros_aligned((Ms,Ms), dtype='float32')		# Selection matrix 	#
		self.y		= fftw.zeros_aligned(self.sz, dtype='float32')		# Goal matrix 		#

		self.H		= fftw.zeros_aligned(self.sz, dtype='complex64')	# Time-smoothing filter #

		# Forward transforms #
		self.phi_0 	= fftw.zeros_aligned(self.sz, dtype='complex64')	# phi_0 = k * psi 	#
		self.Phi_0 	= fftw.zeros_aligned(self.sz, dtype='complex64')

		self.phi_1 	= fftw.zeros_aligned(self.sz, dtype='complex64')	# phi_1 = psi		#
		self.Phi_1 	= fftw.zeros_aligned(self.sz, dtype='complex64')

		self.Phi_2 	= fftw.zeros_aligned(self.sz, dtype='complex64')	# phi_2 = h * psi	#

		# Backward transforms #
		self.KPsi_S0= fftw.zeros_aligned(self.sz, dtype='complex64')	# K . Psi + S_0		#
		self.kpsi_s0= fftw.zeros_aligned(self.sz, dtype='complex64')

		self.Psi_S1 = fftw.zeros_aligned(self.sz, dtype='complex64')	# Psi + S_1			#
		self.psi_s1 = fftw.zeros_aligned(self.sz, dtype='complex64')
		self.Theta	= fftw.zeros_aligned(self.sz, dtype='complex64')	# Local Threshold	#
		self.theta 	= fftw.zeros_aligned(self.sz, dtype='complex64')

		self.HPsi_S2 = fftw.zeros_aligned(self.sz, dtype='complex64')	# H . Psi + S_2		#

		self.Psi	= fftw.zeros_aligned(self.sz, dtype='complex64')	# Emitter response 	#
		self.psi 	= fftw.zeros_aligned(self.sz, dtype='complex64')

		# Working matrices #
		self.S_0 	= fftw.zeros_aligned(self.sz, dtype='complex64')	# Slack variables #
		self.S_1 	= fftw.zeros_aligned(self.sz, dtype='complex64')
		self.S_2	= fftw.zeros_aligned(self.sz, dtype='complex64')

		# FFTW Objects #
		self.FT_phi_0	= fftw.FFTW(self.phi_0, self.Phi_0, 	axes=axes)
		self.FT_phi_1 	= fftw.FFTW(self.phi_1, self.Phi_1, 	axes=axes)

		self.BT_KPsi_S0	= fftw.FFTW(self.KPsi_S0, self.kpsi_s0, axes=axes, direction='FFTW_BACKWARD')
		self.BT_Psi_S1	= fftw.FFTW(self.Psi_S1, self.psi_s1, 	axes=axes, direction='FFTW_BACKWARD')
		self.BT_Theta	= fftw.FFTW(self.Theta, self.theta, 	axes=axes, direction='FFTW_BACKWARD')
		self.BT_Psi		= fftw.FFTW(self.Psi, self.psi, 		axes=axes, direction='FFTW_BACKWARD')

		## Precalculations ##
		# Simple vectors #
		e_L = np.ones(Ls)	# 1-Vectors #
		e_M = np.ones(Ms)
		
		d_M = np.zeros(Ms)	# Kronecker Delta #
		d_M[0] = 1

		# Phi_0 matrices #
		w 			= np.outer(self.MU[0]*e_M + d_M, e_L)			# Weigthing matrix #
		self.w[:] 	= self.MU[0]/w.reshape(self.sz, order="F")

		sig 		= self.MU[0]*np.eye(Ms) + np.outer(d_M, d_M)	# Selection matrix #
		self.sig[:] = np.linalg.inv(sig)

		# Phi_2 matrices #
		t2 = int(np.floor((self.sz[0]-1)/2))	# The middle point of the filter #
		z2 = int(np.floor((self.sz[1]-1)/2))
		y2 = int(np.floor((self.sz[2]-1)/2))
		x2 = int(np.floor((self.sz[3]-1)/2))
		h = np.zeros(self.sz)			# Initialize #

		h[t2,z2,y2,x2] = +1				# 0 offsets #
		h[t2,z2,y2+1,x2] = -1			# 1 offset #
		h[t2,z2,y2,x2+1] = -1
		h[t2,z2,y2+1,x2+1] = +1			# 2 offsets #
		if(USER.KER_Z > 1):
			h[t2,z2+1,y2,x2] = -1		# 1 offset #
			h[t2,z2+1,y2+1,x2] = +1		# 2 offsets #
			h[t2,z2+1,y2,x2+1] = +1
			h[t2,z2+1,y2+1,x2+1] = -1	# 3 offsets #
		if(USER.KER_T > 1):
			h[t2+1,z2,y2,x2] = -1		# 1 offset #
			h[t2+1,z2,y2+1,x2] = +1		# 2 offsets #
			h[t2+1,z2,y2,x2+1] = +1
			h[t2+1,z2,y2+1,x2+1] = -1	# 3 offsets #
		if((USER.KER_Z > 1) and (USER.KER_T > 1)):
			h[t2+1,z2+1,y2,x2] = +1		# 2 offsets #
			h[t2+1,z2+1,y2+1,x2] = -1	# 3 offsets #
			h[t2+1,z2+1,y2,x2+1] = -1
			h[t2+1,z2+1,y2+1,x2+1] = +1	# 4 offsets #

		self.H = npf.fftn(h)	# The N-dim fourier transform is saved #

		# Psi matrices #
		G 	= self.MU[0]*np.power(np.abs(self.K), 2) + self.MU[1] + self.MU[2]*np.power(np.abs(self.H), 2)

		self.Nu_0[:]	= self.MU[0]*np.conj(self.K) / G			# Regularization weights #
		self.Nu_1[:]	= self.MU[1]				 / G
		self.Nu_2[:]	= self.MU[2]*np.conj(self.H) / G

	### Methods ###
	def Recover(self, xi, *, code='', pb=('', 0,1, 0,0, 0,0, 0,0, 0,0), vis=False, error=None, err=None):
		## Initialization ##
		# Hard-code the random seed to make it reproducible #
		np.random.seed(0)

		# Pass xi into the goal image `y`, but pass it through the selection matrix `sig` first #
		y = np.zeros((np.shape(self.sig)[0], np.size(xi)))
		y[0, ...] = xi.reshape(np.size(xi), order="F")
		self.y[:] = np.reshape(self.sig @ y, self.sz, order="F")
		self.y_ = self.y.copy()

		# Reinitialize psi and the slack variables #
		self.Psi[:] = fftw.zeros_aligned(self.sz, dtype='complex64')
		self.S_0[:] = fftw.zeros_aligned(self.sz, dtype='complex64')
		self.S_1[:] = fftw.zeros_aligned(self.sz, dtype='complex64')
		self.S_2[:] = fftw.zeros_aligned(self.sz, dtype='complex64')

		# Localization error #
		if(vis or (error is not None)):
			pts = np.zeros([0, 3]);
			if(error is not None):
				tru = OP._LoadMot(code)
				pts = tru[tru[:,:,3] == pb[1],:]
				temp = []
				temp[:] = pts[:,0]
				pts[:,0] = pts[:,1]		# Swap X & Y #
				pts[:,1] = temp[:]
				H_A, H_S = _IDFilters(self.sz[1:])
			else:
				tru = OP._LoadTruth(code)
				for p in range(len(tru)):
					if(pb[1] in tru[p].frm):
						f = np.nonzero(pb[1] == tru[p].frm)[0]
						pts = np.concatenate((pts, tru[p].res[f,:]), axis=0)

		## Iterate Through ADMM ##
		stpwch = time.time()
		timers = np.zeros((USER.REC_ITER))
		for i in range(USER.REC_ITER):
			# Separate the result of this iteration from the previous solution `self.Psi`.  This allows us to build Psi incrementally, modularizing the regularization.
			Psi = fftw.zeros_aligned(self.sz, dtype='complex64')

			# Perform Regularizations #
			Psi = self.Reg_Accuracy(Psi, i)
			Psi = self.Reg_Sparcity(Psi, 1)#np.minimum(np.maximum(2*i/USER.REC_ITER, 1/2), 3/2))
			Psi = self.Reg_Temporal(Psi)

			# Copy in the new result #
			self.Psi[:] = Psi.copy()

			# Alert us if we have an issue! #
			if(np.any(np.isnan(Psi))): raise ValueError("Psi has NaN values!")

			# Visualization #
			if(vis and (np.mod(i, USER.REC_ITER // 20) == 0)):
				self.BT_Psi()
				plt.clf()
				plt.gca(position=[0,0,1,1])
				plt.imshow(np.log10(npf.fftshift(np.sum(np.abs(self.psi), axis=(0,1)))), cmap='gray')
				plt.plot(pts[:,0], pts[:,1], color='r', marker='o', linewidth=0, fillstyle='none')
				plt.clim(-3, 0)
				plt.draw()
				plt.pause(0.1)
			if((error is not None) and (np.mod(i, 3) == 0) and (i > 60)):
				# Get psi #
				self.BT_Psi()
				
				# Find where psi is important #
				psi_f = npf.fftshift(np.abs(self.psi), axes=(-2,-1))

				# Identify points in the cloud #
				Psi_f = npf.fftn(psi_f)
				psi_a = np.real_if_close(npf.ifftshift(npf.ifftn(Psi_f * H_A), axes=(-2,-1)))
				psi_s = np.real_if_close(npf.ifftshift(npf.ifftn(Psi_f * H_S), axes=(-2,-1)))
				lhs = psi_a
				rhs = psi_s * (1 + 1) + np.mean(psi_f)
				idx = np.nonzero(lhs > rhs)
				pos = np.array([idx[3], idx[2], idx[1], idx[0]/USER.KER_T]).T
				wgt = np.round(np.sqrt(psi_f**2 + ((psi_a + psi_s)/2)**2)[idx], 3)

				if(0 < len(wgt) < 10000):
					# Attempt a triangulation #
					# Create a point cloud based on the points #
					pnts = np.concatenate([pos, wgt[:,None]], axis=1)

					# Segment the point cloud to find clusters #
					cloud = PointCloud(pnts, seg=True)

					# Why?? # vvv #
					# Weight threshold #
					clust = _CloudThr(cloud.clust)
					#		# ^^^ #

					clust = _Separate(clust)
					
					if(len(clust) == 0): continue

					# Evaluate the average minimum error per particle #
					dist_x = np.zeros([np.shape(pts)[0], len(clust)])
					dist_y = np.zeros([np.shape(pts)[0], len(clust)])
					dist_z = np.zeros([np.shape(pts)[0], len(clust)])

					# Evaluate the distance between each point and all clusters #
					for c in range(len(clust)):
						diff = (pts[:,:3] - clust[c].res) * [*USER.RES, USER.DOF[0]/USER.KER_Z]
						dist_x[:,c] = np.abs(diff[:,0])
						dist_y[:,c] = np.abs(diff[:,1])
						dist_z[:,c] = np.abs(diff[:,2])

					# Get the minimum error per cluster, average over all particles #
					error[int(i//3), pb[1], 0] = np.mean(np.min(dist_x, 1))	# X
					error[int(i//3), pb[1], 1] = np.mean(np.min(dist_y, 1))	# Y
					error[int(i//3), pb[1], 2] = np.mean(np.min(dist_z, 1))	# Z
			#if((err is not None) and (np.mod(i, 3) == 0)):
			#	err[pb(1),i,:] = ComputeError(xi)

			# Progress Bar #
			timers[i] = time.time() - stpwch
			if(i > 0):
				prefix = '(%s):\t%8.3f sec' % (pb[0], pb[-2] + timers[i])
				#suffix = '(Remain: %5.0f sec)' % (pb[-1])
				suffix = '(Remain: %3.0f:%2.0f:%2.0f)  ' % (pb[-1] // 3600, (pb[-1] % 3600) // 60, pb[-1] % 60)
				if(pb[4] > 1):	# Show Z progress #
					VIS._ProgressBar(pb[1]+1, pb[2], sub_i=pb[3]+1, sub_I=pb[4], prefix=prefix, suffix=suffix)
				elif(pb[6] > 1 or pb[8] > 1):			# Show chunked iteration progress #
					i_ = i+1 + (pb[5]*pb[8] + pb[7])*USER.REC_ITER
					I_ = pb[6]*pb[8]*USER.REC_ITER
					VIS._ProgressBar(pb[1]+1, pb[2], sub_i=i_, sub_I=I_, prefix=prefix, suffix=suffix)
				else:
					VIS._ProgressBar(pb[1]+1, pb[2], sub_i=i, sub_I=USER.REC_ITER, prefix=prefix, suffix=suffix)
		if(vis):
			plt.ioff()
			plt.show()

		## Output ##
		self.BT_Psi()
		return np.abs(self.psi), error

	## Regularization ##
	def Reg_Accuracy(self, Psi, i):
		## Regularization ##
		# Check to make sure that we can do this regularization properly #
		if(self.LAMBDA[0] > 0 and self.MU[0] > 0):
			# Precalculations #
			self.KPsi_S0[:] = self.K * self.Psi + self.S_0
			self.BT_KPsi_S0()

			# Gradient Descent Solution #
			self.phi_0[:]	= self.y_ + self.kpsi_s0 * self.w
			self.FT_phi_0()

			# Update #
			Psi += self.Nu_0 * (self.Phi_0 - self.S_0)		# Response 			#
			self.S_0[:] = self.KPsi_S0 - self.Phi_0			# Slack variable 	#

			# Jiggle y a bit? #
			self.y_[:] = self.y + 1E-5*np.max(self.y)*np.random.randn(*np.shape(self.y)) * USER.REC_ITER/(i+1)
		
		## Output ##
		return Psi
	def Reg_Sparcity(self, Psi, eps):
		## Regularization ##
		# Check to make sure that we can do this regularization properly #
		if(self.LAMBDA[1] > 0 and self.MU[1] > 0):
			# Precalculations #
			self.Psi_S1[:] = self.Psi + self.S_1
			self.BT_Psi_S1()

			# LASSO Solution #
			mask = np.greater(np.real(self.psi_s1), eps)	# Soft threshold mask #
			self.phi_1[:] = mask*(self.psi_s1 - np.sign(np.real(self.psi_s1))*eps)
			self.FT_phi_1()

			# Update response and slack variables #
			Psi += self.Nu_1 * (self.Phi_1 - self.S_1)		# Response 			#
			self.S_1[:] = self.Psi_S1 - self.Phi_1			# Slack variable 	#

		## Output ##
		return Psi
	def Reg_Temporal(self, Psi):
		## Regularization ##
		# Check to make sure that we can do this regularization properly #
		if(self.LAMBDA[2] > 0 and self.MU[2] > 0):
			# Precalculations #
			self.HPsi_S2[:] = self.H * self.Psi + self.S_2
			prefactor = self.MU[2] / (self.MU[2] + self.LAMBDA[2])

			# Gradient Descent Solution #
			self.Phi_2[:] = prefactor * self.HPsi_S2

			# Update response and slack variables #
			Psi += self.Nu_2 * (self.Phi_2 - self.S_2)		# Response 			#
			self.S_2[:] = self.HPsi_S2 - self.Phi_2			# Slack variable 	#

		## Output ##
		return Psi
		
	## Error Computation ##
	def ComputeError(self, xi):
		## Compute Errors ##
		# To properly calculate the l1 norm, we need to inverse Fourier transform Psi #
		self.BT_Psi()
		
		# Compute each error #
		err_accuracy = self.LAMBDA[0] * np.linalg.norm(xi - npf.ifftshift(np.sum(self.kpsi_s0, axis=(0,1))),ord=2)
		err_sparsity = self.LAMBDA[1] * np.linalg.norm(self.psi.reshape(np.size(self.psi)), ord=1)

		## Output ##
		return np.round(err_accuracy,2), np.round(err_sparsity,2)
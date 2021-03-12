#%% --- IMPORTS --- %%#
### External ###
import numpy			as np
import numpy.fft		as npf
import scipy.special	as sp
import pyfftw			as fftw
import time
import imageio

import matplotlib.pyplot as plt

### Internal ###
from __ENUM	import LOCALIZATION	as LOC
from __ENUM import FILEFORMAT	as FMT
from __ENUM import PHASEMASK	as PM
import __OPERATION				as OP
import __FUNCTION				as FXN
import __VISUALS				as VIS
import USER

#%% --- EXPOSED METHOD --- %%#
def RUN(code, *, update=False, visual=False):
	## Update query ##
	if(update):
		# Create a new microscope #
		scope = Microscope(code)

		# No saving necessary here, as the image exists in \Images and the aperture and kernel exist in their respective folders
	else:
		# Load in what we already have #
		scope = Microscope()
		scope.code = code
		if(OP._CheckFile(code + str(FMT.TIF), OP.FOLD_IMG)):	# Experimental	#
			scope.img = OP._LoadMov('%s' % (code), fold=OP.FOLD_IMG)
		elif(OP._CheckFile(code + str(FMT.TIF), OP.FOLD_SIM)):	# Simulation	#
			scope.img = OP._LoadMov('%s' % (code), fold=OP.FOLD_SIM)
		else:
			# Override - make a fake image #
			scope.img = np.zeros([USER.SIM_FRAMES, 1, *USER.CHIP])
			#raise FileNotFoundError('The specified image or simulation code was not found')
		scope.apr, scope.ker = scope.SimLoad(sz=np.shape(scope.img)[2:])

	## Visualization query ##
	if(visual):
		VIS._VisImg(scope.img[0,0,:,:,], 1000)
		VIS._VisImg(scope.ker[0,0,:,:,], 100)
		VIS._VisImg(scope.ker[0,(USER.KER_Z)//4,:,:,], 550)
		VIS._VisImg(scope.ker[0,(2*USER.KER_Z)//4,:,:,], 550, 550)
		VIS._VisImg(scope.ker[0,(3*USER.KER_Z)//4,:,:,], 100, 550)
		plt.pause(0.1)
		plt.show()
		
	pass	# TODO #

	## Output ##
	return scope

#%% --- STATIC METHODS --- %%#
### Meshes ###
def _MeshLat(y=USER.CHIP[1], x=USER.CHIP[0], *, shift=False, scale=False):
	## Lateral Meshes ##
	# Create the x and y meshes that span the entire chip #
	mesh_x = (np.arange(x)-(x/2 if shift else 0))*(USER.RES[0] if scale else 1)
	mesh_y = (np.arange(y)-(y/2 if shift else 0))*(USER.RES[1] if scale else 1)

	## Output ##
	return mesh_x, mesh_y
def _MeshFou(nu=USER.CHIP[1], mu=USER.CHIP[0], *, shift=False, scale=False):
	## Fourier Meshes ##
	# Create the mu and nu meshes that span -pi to pi, with a resolution of the whole chip #
	mesh_mu = np.pi * np.linspace(-1, 1, mu)
	mesh_nu = np.pi * np.linspace(-1, 1, nu)

	## Output ##
	return mesh_mu, mesh_nu
def _MeshMeta(mot=None, ker_z=USER.KER_Z, ker_t=USER.KER_T):
	## Meta-Meshes ##
	# Then, determine if we need to sample depth and sub-frame multiple times #
	if(mot is None):
		# Get both meshes ready with the default (XY) value #
		mesh_z = np.array([0])
		mesh_t = np.array([0])

		# Check what localization mode we're dealing with #
		if(ker_t > 1 and ker_z > 1):	# Stretching lobe --> both meshes #
			mesh_z = (np.arange(ker_z)/ker_z - 1/2) * USER.DOF[1]
			mesh_t = 1-np.linspace(0, 1 - 1/ker_t, ker_t)
		elif(ker_z > 1):				# Stationary lobe --> only z mesh #
			mesh_z = (np.arange(ker_z)/ker_z - 1/2) * USER.DOF[0]
		elif(ker_t > 1):				# Rotating lobe --> only t mesh #
			mesh_t = 1-np.linspace(0, 1 - 1/ker_t, ker_t)
	else:
		# Then we have the depth and sub-frame positions we need already #
		mesh_z = mot[:,:,2]
		mesh_t = mot[:,:,-2] * (USER.FPS if USER.KER_T > 1 else 1)
	
	## Output ##
	return mesh_z, mesh_t

### Kernel Functions ###
def _FxnRot(ker_type=PM.NONE, loc_z=False, loc_t=False):
	## Output ##
	if(ker_type == PM.NONE):
		return lambda v: 0*v	# No phase modulation #

	elif(ker_type == PM.HELIX):
		if((not loc_z) and (not loc_t)):	# No double helix #
			return lambda v: 0*v
		elif(loc_t):			# The double helix rotates with time #
			return lambda f: (np.pi/USER.KER_LOOP) * f						# In frames #
		else:					# The double helix rotates with depth #
			return lambda z: -(USER.KER_ROT*np.pi) * (z/USER.DOF[0] + 1/2)	# In pixels #
def _FxnSep(ker_type=PM.NONE, loc_z=False, loc_t=False):
	## Output ##
	if(ker_type == PM.NONE):
		return lambda v: 0*v	# No phase modulation #

	elif(ker_type == PM.HELIX):
		if((not loc_z) and (not loc_t)):	# No double helix at all #
			return lambda v: 0*v
		elif(loc_z and (not loc_t)):	# The double helix stays at a constant separation #
			return lambda z: USER.KER_SEP + 0*z	# In microns #
		elif((not loc_z) and loc_t):	# The double helix stays at a constant separation #
			return lambda t: USER.KER_SEP + 0*t	# In frames #
		else:						# The double helix will stretch according to the depth of the emitter #
			return lambda z: USER.KER_SEP + USER.KER_RNG[1] * (z/USER.DOF[1] + 1/2)	# In microns #
def _FxnStr(ker_type=PM.NONE, loc_z=False, loc_t=False):
	## Output ##
	if(ker_type == PM.NONE):
		return lambda v: 0*v	# No phase modulation #

	elif(ker_type == PM.HELIX):
		if((not loc_z) and (not loc_t)):	# No double helix at all #
			return lambda v: 0*v
		elif(loc_z and (not loc_t)):	# The double helix symmetrically varies separation #
			return lambda z: USER.KER_RNG[0] * np.abs(z/USER.DOF[0])	# In microns #
		elif((not loc_z) and loc_t):	# The double helix stays at a constant separation #
			return lambda t: USER.KER_RNG[0] + 0*t	# In frames #
		else:						# The double helix will stretch according to the depth of the emitter #
			return lambda z: USER.KER_RNG[1] * (z/USER.DOF[1] + 1/2)	# In microns #

#%% --- INSTRUMENT CLASS DEFINITION --- %%#
class Microscope:
	### Constructor ###
	def __init__(self, code=None, *fxns, vis=False, blur=None):
		## Initialization ##
		self.fxn_rot = _FxnRot(USER.KER_TYPE, USER.KER_Z > 1, USER.KER_T > 1)
		self.fxn_sep = _FxnSep(USER.KER_TYPE, USER.KER_Z > 1, USER.KER_T > 1)
		self.fxn_str = _FxnStr(USER.KER_TYPE, USER.KER_Z > 1, USER.KER_T > 1)
		
		# Check if we're even going to be doing anything #
		self.code = code
		if(code is None):	return	# Do it manually #		

		## Image ##
		if(USER.SIM_IMG or len(fxns) > 0):
			# Simulate the aperture and kernel if needed #
			apr, ker = self.SimLoad()

			# Simulate the image (this simulates the appropriate aperture and kernel) #
			img, mot = self.SimImage((USER.CHIP[1], USER.CHIP[0]), apr, *fxns, blur=blur)
		else:
			# Check if the code has a `.tif` at the end, and if so, remove it #
			if(code[-4:] == '.tif'): code = code[:-4]

			# Check if an image or simulation exists and load it (Assumes [F, Z, Y, X]) #
			if(OP._CheckFile(code + str(FMT.TIF), OP.FOLD_IMG)):	# Experimental	#
				img = OP._LoadMov(code, OP.FOLD_IMG)
			elif(OP._CheckFile(code + str(FMT.TIF), OP.FOLD_SIM)):	# Simulation	#
				img = OP._LoadMov(code, OP.FOLD_SIM)
			else:													# Oops!			#
				raise FileNotFoundError('The specified image or simulation code was not found')

			sz = np.shape(img)[2:]	# Get the Y-X dimensions #

			# Simulate the aperture and kernel, if need be #
			apr, ker = self.SimLoad(sz)

			# Since we have no idea what the actual motion is, leave it blank #
			mot = None
		
		# Dump into the object #
		self.apr = apr
		self.ker = ker
		self.img = img
		self.mot = mot

	### Methods ###
	def SimLoad(self, sz=(USER.CHIP[1], USER.CHIP[0])):
		# Determine the appropriate filenames #
		kerstr = "NONE" if((USER.KER_Z == 1) and (USER.KER_T == 1)) else str(USER.KER_TYPE)
		fname_apr = 'APR_%ix%i' % sz
		fname_ker = '%s_%ix%ix%ix%i' % (kerstr, USER.KER_T, USER.KER_Z, *sz)

		if(USER.SIM_KER):
			# Simulate and save #
			apr = self.SimAperture(sz, USER.APR_RAD, shape=USER.APR_SHP)
			OP._SaveImg(apr, fname_apr, OP.FOLD_APR)

			ker = self.SimKernel(sz, apr)
			OP._SaveKer(ker, fname_ker, OP.FOLD_KER)
		else:
			# Check if the aperture exists #
			if(OP._CheckFile(fname_apr + '.tif', OP.FOLD_APR)):	# Load them in then #
				apr = OP._LoadImg(fname_apr, OP.FOLD_APR)
			else:
				apr = self.SimAperture(sz, USER.APR_RAD, shape=USER.APR_SHP)
				OP._SaveImg(apr, fname_apr, OP.FOLD_APR)

			# Check if the kernel exists #
			if(OP._CheckFile(fname_ker + '.tif', OP.FOLD_KER)):	# Load them in then #
				ker = OP._LoadKer(fname_ker, OP.FOLD_KER)
			else:
				ker = self.SimKernel(sz, apr)
				OP._SaveKer(ker, fname_ker, OP.FOLD_KER)

		## Output ##
		return apr, ker

	def SimAperture(self, sz, rad_x, rad_y=None, *, shape='cir'):
		## Initialize ##
		# Create the Lateral meshes #
		xx, yy = np.meshgrid(*_MeshLat(*sz, shift=True))

		# Determine boundaries #
		if((shape.lower in ('cir', 'sqr')) or (rad_y is None)):	# Isotropic, only first input #
			rad_y = rad_x
		
		# Transform the aperture into pixels, and un-normalize the sinc functions #
		rad_x /= USER.RES[0]/(np.pi/2)
		rad_y /= USER.RES[1]/(np.pi/2)

		# Check if lobes need to be puffed up a bit (I don't know why?) #
		#if((USER.KER_TYPE == PM.HELIX) and (USER.KER_Z > 1)):
		#	rad_x *= 4/3
		#	rad_y *= 4/3

		## Calculate Aperture ##
		# We work with the Fourier transform directly to prevent edge effects #
		if(shape.lower() in ('cir', 'ell')):
			apr = sp.sinc(np.sqrt((xx/rad_x)**2 + (yy/rad_y)**2))
		elif(shape.lower() in ('sqr', 'rec')):
			apr = sp.sinc(xx/rad_x) + sp.sinc(yy/rad_y)

		## Output ##
		return apr
	def SimKernel(self, sz, apr, *, mot=None, ker_z=USER.KER_Z, ker_t=USER.KER_T, blur=None):
		## Initialization ##
		# Create the meta and Fourier meshes #
		mesh_z, mesh_t = _MeshMeta(mot=mot, ker_z=ker_z, ker_t=ker_t)
		nu, mu = np.meshgrid(*_MeshFou(*sz))

		# Remove scaling #
		mu /= USER.RES[0]
		nu /= USER.RES[1]

		# Initialize the size of the kernel and a few other intermediate steps #
		if(mot is None):	# Proper 4D #
			pts_ker = (mesh_t.size, mesh_z.size, *sz)
		else:				# 3D kernel, 4th dimension is particles present #
			pts_ker = (np.shape(mesh_t)[1], np.shape(mesh_t)[0], *sz)

		## Construct Double Helix ##
		if(USER.KER_T > 1):		# Rotating #
			if(mot is None):	# Outer product #
				x = np.outer(np.cos(self.fxn_rot(mesh_t)), self.fxn_sep(mesh_z))
				y = np.outer(np.sin(self.fxn_rot(mesh_t)), self.fxn_sep(mesh_z))
			else:				# Hadamard product #
				x = np.cos(self.fxn_rot(mesh_t)) * self.fxn_sep(mesh_z)
				y = np.sin(self.fxn_rot(mesh_t)) * self.fxn_sep(mesh_z)
		elif(USER.KER_Z > 1):	# Stationary #
			if(mot is None):	# Outer product #
				x = np.outer(np.ones_like(mesh_t), (self.fxn_sep(mesh_t) + self.fxn_str(mesh_z)) * np.cos(self.fxn_rot(mesh_z)))
				y = np.outer(np.ones_like(mesh_t), (self.fxn_sep(mesh_t) + self.fxn_str(mesh_z)) * np.sin(self.fxn_rot(mesh_z)))
			else:				# Hadamard product #
				x = (self.fxn_sep(mesh_t) + self.fxn_str(mesh_z)) * np.cos(self.fxn_rot(mesh_z))
				y = (self.fxn_sep(mesh_t) + self.fxn_str(mesh_z)) * np.sin(self.fxn_rot(mesh_z))
		else:					# No double helix #
			x = np.zeros_like(mesh_t.T)
			y = np.zeros_like(mesh_t.T)

		# Create Dirac delta points pairwise and reshape into the kernel #
		order = 'C' if(mot is None) else 'F'
		
		if((USER.KER_TYPE == PM.HELIX) and (USER.KER_Z == 1) and (USER.KER_T == 1)):								# Single point #
			Dirac = np.cos( np.outer(x,mu)/np.pi + np.outer(y,nu)/np.pi )

		elif((USER.KER_TYPE == PM.HELIX) and (USER.KER_Z > 1 and USER.KER_T == 1)):	# Double helix - 3D #
			Dirac = np.sin( np.outer(x,mu)/np.pi + np.outer(y,nu)/np.pi )

			# Puff up the lobes a bit #
			"""
			x = (self.fxn_sep(mesh_t) + self.fxn_str(mesh_z)) * np.cos(self.fxn_rot(mesh_z) + np.pi/4)
			y = (self.fxn_sep(mesh_t) + self.fxn_str(mesh_z)) * np.sin(self.fxn_rot(mesh_z) + np.pi/4)
			Dirac += np.sin( np.outer(x,mu)/np.pi + np.outer(y,nu)/np.pi )/3

			x = (self.fxn_sep(mesh_t) + self.fxn_str(mesh_z)) * np.cos(self.fxn_rot(mesh_z) + np.pi/8)
			y = (self.fxn_sep(mesh_t) + self.fxn_str(mesh_z)) * np.sin(self.fxn_rot(mesh_z) + np.pi/8)
			Dirac += np.sin( np.outer(x,mu)/np.pi + np.outer(y,nu)/np.pi )/2

			x = (self.fxn_sep(mesh_t) + self.fxn_str(mesh_z)) * np.cos(self.fxn_rot(mesh_z) - np.pi/8)
			y = (self.fxn_sep(mesh_t) + self.fxn_str(mesh_z)) * np.sin(self.fxn_rot(mesh_z) - np.pi/8)
			Dirac += np.sin( np.outer(x,mu)/np.pi + np.outer(y,nu)/np.pi )/2

			x = (self.fxn_sep(mesh_t) + self.fxn_str(mesh_z)) * np.cos(self.fxn_rot(mesh_z) - np.pi/4)
			y = (self.fxn_sep(mesh_t) + self.fxn_str(mesh_z)) * np.sin(self.fxn_rot(mesh_z) - np.pi/4)
			Dirac += np.sin( np.outer(x,mu)/np.pi + np.outer(y,nu)/np.pi )/3
			"""

		elif((USER.KER_TYPE == PM.HELIX) and (USER.KER_Z == 1 and USER.KER_T > 1)):	# Double helix - 2D #
			Dirac = np.sin( np.outer(x,mu)/np.pi + np.outer(y,nu)/np.pi )

		elif((USER.KER_TYPE == PM.HELIX) and (USER.KER_Z > 1 and USER.KER_T > 1)):
			Dirac = np.sin( np.outer(x,mu)/np.pi + np.outer(y,nu)/np.pi )

		# Reshape #
		Dirac = Dirac.reshape(pts_ker, order=order)
	
		## Construct aperture ##
		Apr_ = np.tile(npf.fftshift(npf.fft2(apr)), [*pts_ker[:2],1,1])

		## Create Fourier Displacement ##
		if(mot is None):
			Psi = np.ones(pts_ker)
		else:
			Psi = np.zeros(pts_ker, dtype='complex64')

			for t in range(np.shape(mesh_t)[0]):
				for p in range(np.shape(mesh_t)[1]):
					# Define the positional phase shifts #
					pos_x = nu * mot[t,p,0]
					pos_y = mu * mot[t,p,1]
					pos_i = mot[t,p,4]

					# Evaluate the emitter response for this time point #
					Psi[p,t,:,:] = pos_i * np.exp(1j * (pos_x + pos_y) )

					# Motion blur if needed #
					if(blur is not None):
						ang = np.pi + np.arctan2(mot[t,p,1], mot[t,p,0])
						dx = 5*blur/USER.FPS * np.cos(ang)	# um/s #
						dy = 5*blur/USER.FPS * np.sin(ang)	# um/s #

						for b in range(20):
							pos_x_ = mu * (mot[t,p,0] + (b+1)*dx/20)
							pos_y_ = nu * (mot[t,p,1] + (b+1)*dy/20)
							pos_i = mot[t,p,4]/(1 + b/10)

							Psi[p,t,:,:] += pos_i * np.exp(1j * (pos_x_ + pos_y_))

		## Circular Convolution ##
		# Establish the fftw objects to perform these operations quickly #
		Ker_norm 	= fftw.zeros_aligned(pts_ker, dtype='complex64')
		ker_norm 	= fftw.zeros_aligned(pts_ker, dtype='complex64')
		BT_ker_norm = fftw.FFTW(Ker_norm, ker_norm, axes=(-2,-1), direction='FFTW_BACKWARD')

		Ker 	= fftw.zeros_aligned(pts_ker, dtype='complex64')
		ker 	= fftw.zeros_aligned(pts_ker, dtype='complex64')
		BT_ker 	= fftw.FFTW(Ker, ker, axes=(-2,-1), direction='FFTW_BACKWARD')

		# Normalizing Kernel #
		Ker_norm[:] = Dirac * Apr_
		BT_ker_norm()

		# Proper kernel #
		Ker[:] = Dirac * Apr_ * Psi
		BT_ker()

		# Normalize and bring to intensity space #
		ker_int = np.abs(np.square(ker / np.max(np.abs(ker_norm))))

		## Output ##
		return ker_int

	def SimImage(self, sz, apr, *fxns, blur=None):
		## Attribute Arguments ##
		dims = len(fxns)
		keys = list()
		for d in range(dims):
			keys.extend(list(fxns[d].keys()))
		keys = np.unique(keys)

		# Fill in missing dimensions #
		fxn_x = {k:FXN._Point(off=0) for k in keys} if dims < 1 else fxns[0]
		fxn_y = {k:FXN._Point(off=0) for k in keys} if dims < 2 else fxns[1]
		fxn_z = {k:FXN._Point(off=0) for k in keys} if dims < 3 else fxns[2]
		fxn_w = {k:FXN._Point(off=1) for k in keys} if dims < 4 else fxns[3]

		# Fill in missing particles #
		for k in keys:
			if(k not in fxn_x):	fxn_x[k] = FXN._Point(off=0)
			if(k not in fxn_y):	fxn_y[k] = FXN._Point(off=0)
			if(k not in fxn_z):	fxn_z[k] = FXN._Point(off=0)
			if(k not in fxn_w):	fxn_w[k] = FXN._Point(off=1)

		## Initialize ##
		if(USER.KER_T == 1):
			dom = np.arange(USER.SIM_FRAMES) / USER.FPS
			mot = np.zeros((USER.SIM_FRAMES, len(keys), 5))
		else:
			dom = np.arange(USER.SIM_FRAMES * USER.SIM_SUB) / (USER.FPS * USER.SIM_SUB)
			mot = np.zeros((USER.SIM_FRAMES * USER.SIM_SUB, len(keys), 5))
		for k in range(len(keys)):
			# mot is [x, y, z, t, w] #
			mot[:,k,:] = np.array([-fxn_x[keys[k]](dom), fxn_y[keys[k]](dom), fxn_z[keys[k]](dom), dom, fxn_w[keys[k]](dom)]).T

		## Simulate the image ##
		# Make the appropriate kernel for each particle #
		kerm = self.SimKernel(sz, apr, mot=mot, blur=blur)

		# Combine together to form the image #
		img = np.sum(kerm, axis=0)
		if(USER.KER_T > 1):
			# This image has sub-frame components that need to be summed #
			img_sub = np.zeros([USER.SIM_FRAMES, *np.shape(img)[1:]])
			for f in range(USER.SIM_FRAMES):
				img_sub[f,:,:] = np.mean(img[f*USER.SIM_SUB:(f+1)*USER.SIM_SUB,:,:], axis=0)

			# Vis #
			fig = plt.figure(figsize=(3,3))
			ax = fig.gca(position=[0,0,1,1])
			frames = []
			img_strem = np.zeros([USER.SIM_FRAMES, *np.shape(img)[1:]])

			for f in range(USER.SIM_FRAMES):
				for sf in range(1, USER.SIM_SUB):
					ax.cla()
					ax.set_xticklabels([])
					ax.set_yticklabels([])

					img_strem[f,:,:] = np.mean(img[f*USER.SIM_SUB:(f*USER.SIM_SUB+sf),:,:], axis=0)
					ax.imshow(img_strem[f,:,:], cmap='gray')

					fig.canvas.draw()
					image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
					image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))				

					frames.append(image)
			imageio.mimsave('./STReM build.gif', frames, fps=12)

			img = img_sub[...]
		img = np.moveaxis(img, (1,2), (2,1))
		img = img[:,None,:,:]		# Insert the z-slice dimension for consistency #

		## Output ##
		return img, mot
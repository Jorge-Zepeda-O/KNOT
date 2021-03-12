#%% --- IMPORTS --- %%#
### External ###
import numpy as np
import numpy.fft as npf
import matplotlib.pyplot as plt

### Internal ###
import __OPERATION	as OP
import __VISUALS	as VIS
import _INITIALIZE	as INIT
import _PREPARE		as PREP
import _RECOVER		as REC
import _SEGMENT		as SEG
import _TRACK		as TRK
import USER

#%% --- USER PARAMETERS --- %%#
CodeCalib = 'roi_Test Data_(230,320)_(64x64)_(f100-101)'
CodeCalib = 'roi_Test Data_(230,245)_(64x64)_(f100-101)'
#CodeCalib = 'roi_344SQ Cell 6_(100,250)_(96x96)_(f0-1)'

#%% FUNCTION DEFINITIONS %%#
def Reconstruct(ker, eps, pos, wgt):
	# Recreate psi #
	psi = np.zeros_like(ker)
	for i in range(len(wgt)):
		psi[int(pos[i,3]), int(pos[i,2]), int(pos[i,1]), int(pos[i,0])] = wgt[i]

	# Construct the image piecewise because I don't trust the FFTN here #
	img = np.zeros(np.shape(ker))
	for f in range(np.shape(ker)[0]):
		for z in range(np.shape(ker)[1]):
			Psi_fz = npf.fft2(psi[f,z,:,:])
			Ker_fz = npf.fft2(((ker[f,z,:,:] * (ker[f,z,:,:]>0))))
			img[f,z,:,:] = npf.fftshift(np.real_if_close(npf.ifft2(Psi_fz * Ker_fz)))
	"""
	fig, ax = plt.subplots(4,4)
	for i in range(4):
		for j in range(4):
			ax[i,j].imshow(ker[0,4*i+j,:,:])

	fig, ax = plt.subplots(4,4)
	for i in range(4):
		for j in range(4):
			ax[i,j].imshow(img[0,4*i+j,:,:])
	plt.show()
	"""
	## Output ##
	return np.sum(img, axis=(0,1)) * eps**2
def Error(img, eps, rec, wgt):
	img_ = (img - eps) * (img > eps)
	rec_ = rec
	return np.sqrt(np.sum((img_ - rec_)**2)) + 4*np.sum(wgt)

def Calibrate(fxn, x0, dx0, strings, iter=5, vis=True):
	# Initialize #
	x = x0
	dx = dx0
	winner = np.nan

	# Gradient descent #
	
	if(vis):
		plt.ion()
		plt.figure(figsize=(8,8))
	for i in range(iter):
		for f in range(len(fxn)):
			if(dx[f] < 1E-3): continue	# Good enough #
			if(vis): plt.clf()

			# Get the three different values #
			fxn[f](x[f] - dx[f])
			scope_m = INIT.Microscope(CodeCalib)

			fxn[f](x[f])
			scope_0 = INIT.Microscope(CodeCalib)

			fxn[f](x[f] + dx[f])
			scope_p = INIT.Microscope(CodeCalib)

			# Preprocess #
			img_m, ker_m, eps_m = PREP._Preprocess2(CodeCalib, scope_m.img, scope_m.ker)
			img_0, ker_0, eps_0 = PREP._Preprocess2(CodeCalib, scope_0.img, scope_0.ker)
			img_p, ker_p, eps_p = PREP._Preprocess2(CodeCalib, scope_p.img, scope_p.ker)

			# Recover positions #
			[pos_m, wgt_m] = REC._Recover(img_m, ker_m, eps_m, code=CodeCalib)
			[pos_0, wgt_0] = REC._Recover(img_0, ker_0, eps_0, code=CodeCalib)
			[pos_p, wgt_p] = REC._Recover(img_p, ker_p, eps_p, code=CodeCalib)

			# Reconstruct images and get errors #
			rec_m = Reconstruct(ker_m, eps_m[0,0,:,:], pos_m[0], wgt_m[0])
			rec_0 = Reconstruct(ker_0, eps_0[0,0,:,:], pos_0[0], wgt_0[0])
			rec_p = Reconstruct(ker_p, eps_p[0,0,:,:], pos_p[0], wgt_p[0])

			err_m = Error(img_m[0,0,:,:], eps_m[0,0,:,:], rec_m, wgt_m[0])
			err_0 = Error(img_0[0,0,:,:], eps_0[0,0,:,:], rec_0, wgt_0[0])
			err_p = Error(img_p[0,0,:,:], eps_p[0,0,:,:], rec_p, wgt_p[0])

			# Check which is best #
			if(err_m == np.min([err_m, err_0, err_p])):
				print(("%s -*Error @ %0.3fum: %0.3f*| Error @ %0.3fum: %0.3f | Error @ %0.3fum: %0.3f " + " "*100) \
					% (strings[f], x[f]-dx[f], err_m, x[f], err_0, x[f]+dx[f], err_p), end='')
				x[f] -= dx[f]
				winner = -1
			elif(err_p == np.min([err_m, err_0, err_p])):
				print(("%s - Error @ %0.3fum: %0.3f | Error @ %0.3fum: %0.3f |*Error @ %0.3fum: %0.3f*" + " "*100) \
					% (strings[f], x[f]-dx[f], err_m, x[f], err_0, x[f]+dx[f], err_p), end='')
				x[f] += dx[f]
				winner = +1
			else:
				print(("%s - Error @ %0.3fum: %0.3f |*Error @ %0.3fum: %0.3f*| Error @ %0.3fum: %0.3f " + " "*100) \
					% (strings[f], x[f]-dx[f], err_m, x[f], err_0, x[f]+dx[f], err_p), end='')
				dx[f] = dx[f] / 2
				winner = 0

			# Visualization #
			if(vis): 
				plt.clf()

				# Axes definitions #
				ax = [None] * 4
				ax[0] = plt.axes(position=[0,1/2,1/2,0.45])		# Original image#
				ax[1] = plt.axes(position=[0,0,1/2,0.45])		# Lower value	#
				ax[2] = plt.axes(position=[1/2,1/2,1/2,0.45])	# Middle value	#
				ax[3] = plt.axes(position=[1/2,0,1/2,0.45])		# Upper value	#

				# Image showing #
				ax[0].imshow(img_0[0,0,:,:], cmap='gray')
				ax[1].imshow(rec_m, cmap='gray')
				ax[2].imshow(rec_0, cmap='gray')
				ax[3].imshow(rec_p, cmap='gray')

				# Scatter plots #
				ax[1].scatter(pos_m[0][:,0], pos_m[0][:,1], s=100*wgt_m[0], color='g' if winner == -1 else 'r')
				ax[2].scatter(pos_0[0][:,0], pos_0[0][:,1], s=100*wgt_0[0], color='g' if winner == 0 else 'r')
				ax[3].scatter(pos_p[0][:,0], pos_p[0][:,1], s=100*wgt_p[0], color='g' if winner == +1 else 'r')

				# Setting titles #
				ax[0].set_title("Original Image")
				ax[1].set_title("%s Error @ %0.3fum: %0.3f" % (strings[f], x[f] - dx[f], err_m))
				ax[2].set_title("%s Error @ %0.3fum: %0.3f" % (strings[f], x[f], err_0))
				ax[3].set_title("%s Error @ %0.3fum: %0.3f" % (strings[f], x[f] + dx[f], err_p))

				# Accessories #
				for p in range(4):
					ax[p].set_xticklabels([])
					ax[p].set_yticklabels([])

				plt.draw()
				plt.pause(0.01)

			# Re-set the value for the next check #
			fxn[f](x[f])
	if(vis):
		plt.ioff()
		plt.show()

	return x

#%% RUNTIME %%#
## Methods to mutate user parameters ##
def SetSep(s): USER.KER_SEP = s
def SetRad(r): USER.APR_RAD = r
def SetStr(s): USER.KER_RNG[0] = s

## Initialize ##
print("Obtaining original guess...")
scope_init = INIT.Microscope(CodeCalib)
img_init, ker_init, eps_init = PREP._Preprocess2(CodeCalib, scope_init.img, scope_init.ker)
[pos_init, wgt_init] = REC._Recover(img_init, ker_init, eps_init, code=CodeCalib)
rec_init = Reconstruct(ker_init, eps_init[0,0,:,:], pos_init[0], wgt_init[0])
err_init = Error(img_init[0,0,:,:], eps_init[0,0,:,:], rec_init, wgt_init[0])
print('\nInitial error: %0.3f\n\n' % (err_init))

## Calibration ##
[sep, rad, str] = Calibrate([SetSep, SetRad, SetStr], \
	[USER.KER_SEP, USER.APR_RAD, USER.KER_RNG[0]], [1/8, 1/16, 1/16], ['Sep', 'Rad', 'Str'])

# Get the optimized result #
print("\nObtaining calibrated guess...")
SetSep(sep)
SetRad(rad)
scope_opt = INIT.Microscope(CodeCalib)
img_opt, ker_opt, eps_opt = PREP._Preprocess2(CodeCalib, scope_opt.img, scope_opt.ker)
[pos_opt, wgt_opt] = REC._Recover(img_opt, ker_opt, eps_opt, code=CodeCalib)
rec_opt = Reconstruct(ker_opt, eps_opt[0,0,:,:], pos_opt[0], wgt_opt[0])
err_opt = Error(img_opt[0,0,:,:], eps_opt[0,0,:,:], rec_opt, wgt_opt[0])
print('\nCalibrated error: %0.3f\n\n' % (err_opt))

print('Calibration results: Sep %0.3f um, Rad %0.3f um, Str %0.3f um' % (sep, rad, str))

# Display #
plt.figure(figsize=(15, 5))
ax_init = plt.axes(position=[0,0,1/3,0.95])
ax_init.imshow(rec_init)
ax_init.scatter(pos_init[0][:,0], pos_init[0][:,1], s=100*wgt_init[0], c='r')
ax_init.set_title("Pre-Calibration reconstruction")

ax = plt.axes(position=[1/3,0,1/3,0.95])
ax.imshow(img_opt[0,0,:,:])
ax.set_title("Original Image")

ax_opt = plt.axes(position=[2/3,0,1/3,0.95])
ax_opt.imshow(rec_opt)
ax_opt.scatter(pos_opt[0][:,0], pos_opt[0][:,1], s=100*wgt_opt[0], c='r')
ax_opt.set_title("Post-Calibration reconstruction")

plt.show()
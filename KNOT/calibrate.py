#%% --- IMPORTS --- %%#
### External ###
import numpy as np
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

#%% FUNCTION DEFINITIONS %%#
def Calibrate(fxn, x0, dx0, iter=3, vis=True):
	# Initialize #
	x = x0
	dx = dx0
	winner = np.nan

	# Gradient descent #
	if(vis):
		plt.ion()
		plt.figure(figsize=(15,5))
	for i in range(iter):
		if(vis): plt.clf()

		# Decrease value #
		fxn(x - dx)
		scope_m = INIT.Microscope(CodeCalib)
		img_m, ker_m, eps_m = PREP._Preprocess2(CodeCalib, scope_m.img, scope_m.ker)
		[pos_m, wgt_m] = REC._Recover(img_m, ker_m, eps_m, code=CodeCalib)
		clouds_m = SEG._Identify(pos_m, wgt_m, img_m, code=CodeCalib)
		prec_m = np.mean([clouds_m[c].m2a for c in range(len(clouds_m))], axis=0)*1000
		if(vis):
			ax = plt.axes(position=[0,0,1/3,0.9])
			ax.imshow(img_m[0,0,:,:], cmap='gray')
			ax.scatter(pos_m[0][:,0], pos_m[0][:,1], s=100*wgt_m[0], color='r')
			ax.set_title("Error @ %0.3fum: %0.3fnm" % (x - dx, np.mean(prec_m)))
			plt.draw()
			plt.pause(0.01)

		# Keep value #
		fxn(x)
		scope_0 = INIT.Microscope(CodeCalib)
		img_0, ker_0, eps_0 = PREP._Preprocess2(CodeCalib, scope_0.img, scope_0.ker)
		[pos_0, wgt_0] = REC._Recover(img_0, ker_0, eps_0, code=CodeCalib)
		clouds_0 = SEG._Identify(pos_0, wgt_0, img_0, code=CodeCalib)
		prec_0 = np.mean([clouds_0[c].m2a for c in range(len(clouds_0))], axis=0)*1000
		if(vis):
			ax = plt.axes(position=[1/3,0,1/3,0.9])
			ax.imshow(img_0[0,0,:,:], cmap='gray')
			ax.scatter(pos_0[0][:,0], pos_0[0][:,1], s=100*wgt_0[0], color='r')
			ax.set_title("Error @ %0.3fum: %0.3fnm" % (x, np.mean(prec_0)))
			plt.draw()
			plt.pause(0.01)

		# Increase value #
		fxn(x + dx)
		scope_p = INIT.Microscope(CodeCalib)
		img_p, ker_p, eps_p = PREP._Preprocess2(CodeCalib, scope_p.img, scope_p.ker)
		[pos_p, wgt_p] = REC._Recover(img_p, ker_p, eps_p, code=CodeCalib)
		clouds_p = SEG._Identify(pos_p, wgt_p, img_p, code=CodeCalib)	
		prec_p = np.mean([clouds_p[c].m2a for c in range(len(clouds_p))], axis=0)*1000
		if(vis):
			ax = plt.axes(position=[2/3,0,1/3,0.9])
			ax.imshow(img_p[0,0,:,:], cmap='gray')
			ax.scatter(pos_p[0][:,0], pos_p[0][:,1], s=100*wgt_p[0], color='r')
			ax.set_title("Error @ %0.3fum: %0.3fnm" % (x + dx, np.mean(prec_p)))
			plt.draw()
			plt.pause(0.01)

		# Check which is best #
		if(np.mean(prec_m) < np.mean(prec_0)):
			print(("*Error @ %0.3fum: %0.3fnm*| Error @ %0.3fum: %0.3fnm | Error @ %0.3fum: %0.3fnm " + " "*100) \
				% (x-dx, np.mean(prec_m), x, np.mean(prec_0), x+dx, np.mean(prec_p)), end='')
			x -= dx
			winner = -1
		elif(np.mean(prec_p) < np.mean(prec_0)):
			print((" Error @ %0.3fum: %0.3fnm | Error @ %0.3fum: %0.3fnm |*Error @ %0.3fum: %0.3fnm*" + " "*100) \
				% (x-dx, np.mean(prec_m), x, np.mean(prec_0), x+dx, np.mean(prec_p)), end='')
			x += dx
			winner = +1
		else:
			print((" Error @ %0.3fum: %0.3fnm |*Error @ %0.3fum: %0.3fnm*| Error @ %0.3fum: %0.3fnm " + " "*100) \
				% (x-dx, np.mean(prec_m), x, np.mean(prec_0), x+dx, np.mean(prec_p)), end='')
			dx = dx / 2
			winner = 0

		# Break-out #
		if dx < 1E-3: break
	if(vis):
		plt.ioff()
		plt.show()
	return x

#%% RUNTIME %%#
def SetSep(s): USER.KER_SEP = s
def SetRad(r): USER.APR_RAD = r

if(USER.KER_Z > 1):
	# Else there's no need to #
	print("\nLobe Separation")
	sep = Calibrate(SetSep, USER.KER_SEP, 0.2)
	print("Best value: %5.3f um" % (sep))

print("\nAperture Radius")
rad = Calibrate(SetRad, USER.APR_RAD, 0.1)
print("Best value: %5.3f um" % (rad))
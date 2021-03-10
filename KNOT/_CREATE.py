#%% --- IMPORTS --- %%#
### External ###
import numpy 			as np
import scipy.special 	as sp
import scipy.io			as spi

from mpl_toolkits.mplot3d	import Axes3D
import matplotlib.pyplot	as plt

### Internal ###
from __ENUM		import FILEFORMAT	as FMT
import __OPERATION	as OP
import __VISUALS	as VIS

from _INITIALIZE	import Microscope

import __FUNCTION		as FXN
import USER

#%% --- USER PARAMETERS --- %%#
code = 'TOC3D'
snr = 4 if(code[:3] != 'SNR') else int(code[4:])
blur = None if(code[:4] != 'BLUR') else int(code[4:])

#%% SIMULATION DEFINITIONS %%#
# A word of note: simulations use units! Please write your parameters down! #
def FIG_1a():
	## Parameters ##
	# CHIP: 128x128	(px)
	# RES:	0.0685	(um/px)
	# FPS:	10		(frames/s)
	# LEN:	20		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	# Diffusive #
	fxn_x[0] = FXN._Wiener(0, amp=-0.9, off=-1,	seed=5)
	fxn_y[0] = FXN._Wiener(0, amp=-0.9, off=1, 		seed=4)
	fxn_z[0] = FXN._Wiener(0, amp=0.6, off=0, 		seed=1)

	# Axially Directed #
	fxn_x[1] = FXN._Wiener(0, amp=0.3, off=2, 			seed=11)
	fxn_y[1] = FXN._Wiener(0, amp=0.3, off=1.5, 		seed=12)
	fxn_z[1] = FXN._Wiener(7, 0.2, amp=0.2, off=-1.5, 	seed=13)

	# Mixed motion overlapping #
	imp_x = FXN._Gauss(1.6, 0.4, amp=6,		seed=34)
	imp_y = FXN._Gauss(0.4, 0.4, amp=-6,	seed=35)
	var_x = FXN._Gauss(1.7, 0.2, amp=-2, off=1, seed=36)
	var_y = FXN._Gauss(0.4, 0.2, amp=2, off=-1, seed=37)
	fxn_x[2] = FXN._Wiener(imp_x, var_x, off=-2, 	seed=31)
	fxn_y[2] = FXN._Wiener(imp_y, var_y, off=2.0, 	seed=30)
	fxn_z[2] = FXN._Wiener(off=+0.7, 				seed=34)

	imp_x = FXN._Gauss(1.5, 0.4, amp=6,		seed=34)
	imp_y = FXN._Gauss(0.4, 0.4, amp=-6,	seed=35)
	var_x = FXN._Gauss(1.5, 0.2, amp=-2, off=1, seed=36)
	var_y = FXN._Gauss(0.4, 0.2, amp=2, off=-1, seed=37)
	imp_z = FXN._Gauss(0.4, 0.2, amp=3, seed=4)
	fxn_x[3] = FXN._Wiener(imp_x, var_x, off=-2, 	seed=30)
	fxn_y[3] = FXN._Wiener(imp_y, var_y, off=2.0, 	seed=31)
	fxn_z[3] = FXN._Wiener(imp_z, off=0.3, 		seed=32)

	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def FIG_1b():
	## Parameters ##
	# CHIP: 128x128	(px)
	# RES:	0.0685	(um/px)
	# DOF:	4.000	(um)
	# FPS:	10		(frames/s)
	# LEN:	20		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	# Diffusive #
	fxn_x[0] = FXN._Wiener(0, amp=-0.9, off=-1,	seed=5)
	fxn_y[0] = FXN._Wiener(0, amp=-0.9, off=1, 		seed=4)
	fxn_z[0] = FXN._Wiener(0, amp=0.6, off=0, 		seed=1)

	# Axially Directed #
	fxn_x[2] = FXN._Wiener(0, amp=0.3, off=2, 			seed=11)
	fxn_y[2] = FXN._Wiener(0, amp=0.3, off=1.5, 		seed=12)
	fxn_z[2] = FXN._Wiener(7, 0.2, amp=0.2, off=-1.5, 	seed=13)

	# Mixed motion overlapping #
	imp_x = FXN._Gauss(1.6, 0.4, amp=6,		seed=34)
	imp_y = FXN._Gauss(0.4, 0.4, amp=-6,	seed=35)
	var_x = FXN._Gauss(1.7, 0.2, amp=-2, off=1, seed=36)
	var_y = FXN._Gauss(0.4, 0.2, amp=2, off=-1, seed=37)
	fxn_x[3] = FXN._Wiener(imp_x, var_x, off=-2, 	seed=31)
	fxn_y[3] = FXN._Wiener(imp_y, var_y, off=2.0, 	seed=30)
	fxn_z[3] = FXN._Wiener(off=+0.7, 				seed=34)

	imp_x = FXN._Gauss(1.5, 0.4, amp=6,		seed=34)
	imp_y = FXN._Gauss(0.4, 0.4, amp=-6,	seed=35)
	var_x = FXN._Gauss(1.5, 0.2, amp=-2, off=1, seed=36)
	var_y = FXN._Gauss(0.4, 0.2, amp=2, off=-1, seed=37)
	imp_z = FXN._Gauss(0.4, 0.2, amp=3, seed=4)
	fxn_x[1] = FXN._Wiener(imp_x, var_x, off=-2, 	seed=30)
	fxn_y[1] = FXN._Wiener(imp_y, var_y, off=2.0, 	seed=31)
	fxn_z[1] = FXN._Wiener(imp_z, off=-0.7, 		seed=32)

	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def FIG_1di():
	## Parameters ##
	# CHIP: 32x32	(px)
	# RES:	0.0685	(um/px)
	# DOF:	4.000	(um)
	# FPS:	30		(frames/s)
	# LEN:	1		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	fxn_x[0] = FXN._Wiener(0, amp=1, off=0,	seed=5)
	fxn_y[0] = FXN._Wiener(0, amp=1, off=0,	seed=4)
	fxn_z[0] = FXN._Wiener(0, amp=1, off=np.random.randn(1),	seed=1)

	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def FIG_1dii():
	## Parameters ##
	# CHIP: 32x32	(px)
	# RES:	0.0685	(um/px)
	# DOF:	4.000	(um)
	# FPS:	30		(frames/s)
	# LEN:	1		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	np.random.seed(0)
	fxn_x[0] = FXN._Wiener(0, amp=1, off=-0.25)
	fxn_y[0] = FXN._Wiener(0, amp=1, off=0.1)
	fxn_z[0] = FXN._Wiener(0, amp=1, off=4/3)

	fxn_x[1] = FXN._Wiener(0, amp=1, off=0.25)
	fxn_y[1] = FXN._Wiener(0, amp=1, off=0.1)
	fxn_z[1] = FXN._Wiener(0, amp=1, off=-4/3)

	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def FIG_1diii():
	## Parameters ##
	# CHIP: 32x32	(px)
	# RES:	0.0685	(um/px)
	# DOF:	4.000	(um)
	# FPS:	30		(frames/s)
	# LEN:	1		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	np.random.seed(0)
	for p in range(10):
		fxn_x[p] = FXN._Wiener(0, amp=1, off=np.random.randn(1)/3)
		fxn_y[p] = FXN._Wiener(0, amp=1, off=np.random.randn(1)/3)
		fxn_z[p] = FXN._Wiener(0, amp=1, off=np.random.randn(1))

	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def FIG_2x():
	## Parameters ##
	# CHIP: 32x32	(px)
	# RES:	0.0685	(um/px)
	# DOF:	4.000	(um)
	# KER_Z:16		(planes)
	# FPS:	30		(frames/s)
	# LEN:	1		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	np.random.seed(0)
	fxn_x[0] = FXN._Wiener(0, amp=1, off=-1/4)
	fxn_y[0] = FXN._Wiener(0, amp=1, off=0.1)
	fxn_z[0] = FXN._Wiener(0, amp=1, off=5/4)

	fxn_x[1] = FXN._Wiener(0, amp=1, off=1/4)
	fxn_y[1] = FXN._Wiener(0, amp=1, off=0.1)
	fxn_z[1] = FXN._Wiener(0, amp=1, off=-5/4)

	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def FIG_3b():
	## Parameters ##
	# CHIP: 64x64	(px)
	# RES:	0.0685	(um/px)
	# DOF:	0.000	(um)
	# FPS:	30		(frames/s)
	# LEN:	13		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	# Accelerating curved #
	fxn_x[0] = FXN._Cosine(1, -np.pi/6, amp=FXN._Poly(15, 5, off=0), off=1.5)
	fxn_y[0] = FXN._Sine(1, -np.pi/6, amp=FXN._Poly(15, 5, off=0), off=-1)

	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def FIG_6a():
	## Parameters ##
	# CHIP: 48x48	(px)
	# RES:	0.0685	(um/px)
	# DOF:	0.000	(um)
	# FPS:	200		(frames/s)
	# LEN:	37		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	# Diffusive Motion #
	fxn_x[0] = FXN._Wiener(0, 2, amp=1.5, off=0.5, seed=101)
	fxn_y[0] = FXN._Wiener(0, 2, amp=1.5, off=0, seed=103)

	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def FIG_6b():
	## Parameters ##
	# CHIP: 48x48	(px)
	# RES:	0.0685	(um/px)
	# DOF:	0.000	(um)
	# FPS:	200		(frames/s)
	# LEN:	37		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	# Directed Motion #
	fxn_x[0] = FXN._Wiener(5, 0.25, amp=2, off=-1, seed=0)
	fxn_y[0] = FXN._Wiener(5, 0.25, amp=2, off=-1, seed=1)

	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def FIG_6c():
	## Parameters ##
	# CHIP: 48x48	(px)
	# RES:	0.0685	(um/px)
	# DOF:	0.000	(um)
	# FPS:	200		(frames/s)
	# LEN:	37		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	# Mixed Motion #
	imp_x = FXN._Gauss(0, 0.05, amp=-20,	seed=0)
	imp_y = FXN._Gauss(0, 0.05, amp=-20,	seed=1)
	var_x = FXN._Gauss(0, 0.05, amp=-0.5, off=1, seed=30)
	var_y = FXN._Gauss(0, 0.05, amp=-0.5, off=1, seed=11)
	fxn_x[0] = FXN._Wiener(imp_x, var_x, amp=1.5, off=1.00, seed=80)
	fxn_y[0] = FXN._Wiener(imp_y, var_y, amp=1.5, off=1.00, seed=6)

	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def FIG_6x():
	## Parameters ##
	# CHIP: 48x48	(px)
	# RES:	0.0685	(um/px)
	# DOF:	0.000	(um)
	# FPS:	200		(frames/s)
	# LEN:	37		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	# Mixed Motion #
	imp_x = FXN._Gauss(0.15, 0.05, amp=-20,	seed=0)
	imp_y = FXN._Gauss(0, 0.05, amp=-20,	seed=1)
	var_x = FXN._Gauss(0.15, 0.05, amp=-0.5, off=1, seed=30)
	var_y = FXN._Gauss(0, 0.05, amp=-0.5, off=1, seed=11)
	fxn_x[0] = FXN._Wiener(0, 1, amp=1.5, off=1.00, seed=8)
	fxn_y[0] = FXN._Wiener(0, 1, amp=1.5, off=0.00, seed=6)

	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def TOC():
	## Parameters ##
	# CHIP: 64x128	(px)
	# RES:	0.0685	(um/px)
	# DOF:	0.000	(um)
	# FPS:	200		(frames/s)
	# LEN:	37		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	imp_x = FXN._Gauss(0.5, 0.1, amp=-12, seed=1)
	imp_y = FXN._Gauss(0.5, 0.1, amp=12, seed=1)
	off_z = FXN._Gauss(0.5, 0.09, amp=2, off=-1, seed=1)
	fxn_x[0] = FXN._Wiener(imp_x, amp=1, off=1.3, seed=6)
	fxn_y[0] = FXN._Wiener(imp_y, amp=1, off=-0.8, seed=9)
	fxn_z[0] = FXN._Wiener(amp=0, off=off_z, seed=1)

	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w

def ADMM(P):
	## Parameters ##
	# CHIP: 128x128	(px)
	# RES:	0.0685	(um/px)
	# FPS:	1		(frames/s)
	# LEN:	50		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	np.random.seed(0)
	pos_x = 0.7*(np.random.rand(P, USER.SIM_FRAMES) - 0.5) * (USER.CHIP[0] * USER.RES[0])
	pos_y = 0.7*(np.random.rand(P, USER.SIM_FRAMES) - 0.5) * (USER.CHIP[1] * USER.RES[1])
	pos_z = 0.7*(np.random.rand(P, USER.SIM_FRAMES) - 0.5) * (USER.DOF[0])
	for p in range(P):
		fxn_x[p] = FXN._Point(amp=1, off=pos_x[p,:], seed=0)
		fxn_y[p] = FXN._Point(amp=1, off=pos_y[p,:], seed=0)
		fxn_z[p] = FXN._Point(amp=1, off=pos_z[p,:], seed=0)
	
	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def TEST_LOC():
	## Parameters ##
	# CHIP: 64x64	(px)
	# RES:	0.0685	(um/px)
	# FPS:	1		(frames/s)
	# LEN:	50		(frames)

	# SNR:	      4
	# DEN:	      4
	# BLUR: 0
	P = 4

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	np.random.seed(0)
	pos_x = 0.7*(np.random.rand(P, USER.SIM_FRAMES) - 0.5) * (USER.CHIP[0] * USER.RES[0])
	pos_y = 0.7*(np.random.rand(P, USER.SIM_FRAMES) - 0.5) * (USER.CHIP[1] * USER.RES[1])
	pos_z = 0.7*(np.random.rand(P, USER.SIM_FRAMES) - 0.5) * (USER.DOF[0])
	for p in range(P):
		# Make sure the particles aren't too close if you can avoid it #
		for f in range(USER.SIM_FRAMES):
			dist = [np.sqrt((pos_x[p,f] - pos_x[q,f])**2 + (pos_y[p,f] - pos_y[q,f])**2 + (pos_z[p,f] - pos_z[q,f])**2) for q in range(p)]
			for _ in range(10):
				if((p > 0) and (min(dist) < USER.CHIP[0]*USER.RES[0]/P)):
					# Generate a new random position #
					pos_x[p,f] = 0.7*(np.random.rand() - 0.5) * (USER.CHIP[0] * USER.RES[0])
					pos_y[p,f] = 0.7*(np.random.rand() - 0.5) * (USER.CHIP[1] * USER.RES[1])
					pos_z[p,f] = 0.7*(np.random.rand() - 0.5) * (USER.DOF[0])

					# Recompute the distance #
					dist = [np.sqrt((pos_x[p,f] - pos_x[q,f])**2 + (pos_y[p,f] - pos_y[q,f])**2 + (pos_z[p,f] - pos_z[q,f])**2) for q in range(p)]

		# Place the particles down #
		fxn_x[p] = FXN._Point(amp=1, off=pos_x[p,0], seed=0)
		fxn_y[p] = FXN._Point(amp=1, off=pos_y[p,0], seed=0)
		fxn_z[p] = FXN._Point(amp=1, off=pos_z[p,0], seed=0)
	
	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def TEST_SNR(snr):
	## Parameters ##
	# CHIP: 64x64	(px)
	# RES:	0.0685	(um/px)
	# FPS:	1		(frames/s)
	# LEN:	50		(frames)

	# SNR:	1, 2, 4, 7, 10
	# DEN:	      4
	# BLUR: 0

	P = 4

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	np.random.seed(0)
	pos_x = 0.7*(np.random.rand(P, USER.SIM_FRAMES) - 0.5) * (USER.CHIP[0] * USER.RES[0])
	pos_y = 0.7*(np.random.rand(P, USER.SIM_FRAMES) - 0.5) * (USER.CHIP[1] * USER.RES[1])
	pos_z = 0.7*(np.random.rand(P, USER.SIM_FRAMES) - 0.5) * (USER.DOF[0])
	for p in range(P):
		# Make sure the particles aren't too close if you can avoid it #
		for f in range(USER.SIM_FRAMES):
			dist = [np.sqrt((pos_x[p,f] - pos_x[q,f])**2 + (pos_y[p,f] - pos_y[q,f])**2 + (pos_z[p,f] - pos_z[q,f])**2) for q in range(p)]
			for _ in range(10):
				if((p > 0) and (min(dist) < USER.CHIP[0]*USER.RES[0]/P)):
					# Generate a new random position #
					pos_x[p,f] = 0.7*(np.random.rand() - 0.5) * (USER.CHIP[0] * USER.RES[0])
					pos_y[p,f] = 0.7*(np.random.rand() - 0.5) * (USER.CHIP[1] * USER.RES[1])
					pos_z[p,f] = 0.7*(np.random.rand() - 0.5) * (USER.DOF[0])

					# Recompute the distance #
					dist = [np.sqrt((pos_x[p,f] - pos_x[q,f])**2 + (pos_y[p,f] - pos_y[q,f])**2 + (pos_z[p,f] - pos_z[q,f])**2) for q in range(p)]

		# Place the particles down #
		fxn_x[p] = FXN._Point(amp=1, off=pos_x[p,:], seed=0)
		fxn_y[p] = FXN._Point(amp=1, off=pos_y[p,:], seed=0)
		fxn_z[p] = FXN._Point(amp=1, off=pos_z[p,:], seed=0)
	
	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def TEST_DEN(P):
	## Parameters ##
	# CHIP: 64x64	(px)
	# RES:	0.0685	(um/px)
	# FPS:	1		(frames/s)
	# LEN:	50		(frames)

	# SNR:	      4
	# DEN:	1, 2, 4, 7, 10
	# BLUR: 0

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	np.random.seed(0)
	pos_x = 0.7*(np.random.rand(P, USER.SIM_FRAMES) - 0.5) * (USER.CHIP[0] * USER.RES[0])
	pos_y = 0.7*(np.random.rand(P, USER.SIM_FRAMES) - 0.5) * (USER.CHIP[1] * USER.RES[1])
	pos_z = 0.7*(np.random.rand(P, USER.SIM_FRAMES) - 0.5) * (USER.DOF[0])
	for p in range(P):
		# Make sure the particles aren't too close if you can avoid it #
		for f in range(USER.SIM_FRAMES):
			dist = [np.sqrt((pos_x[p,f] - pos_x[q,f])**2 + (pos_y[p,f] - pos_y[q,f])**2 + (pos_z[p,f] - pos_z[q,f])**2) for q in range(p)]
			for _ in range(10):
				if((p > 0) and (min(dist) < USER.CHIP[0]*USER.RES[0]/P)):
					# Generate a new random position #
					pos_x[p,f] = 0.7*(np.random.rand() - 0.5) * (USER.CHIP[0] * USER.RES[0])
					pos_y[p,f] = 0.7*(np.random.rand() - 0.5) * (USER.CHIP[1] * USER.RES[1])
					pos_z[p,f] = 0.7*(np.random.rand() - 0.5) * (USER.DOF[0])

					# Recompute the distance #
					dist = [np.sqrt((pos_x[p,f] - pos_x[q,f])**2 + (pos_y[p,f] - pos_y[q,f])**2 + (pos_z[p,f] - pos_z[q,f])**2) for q in range(p)]

		# Place the particles down #
		fxn_x[p] = FXN._Point(amp=1, off=pos_x[p,:], seed=0)
		fxn_y[p] = FXN._Point(amp=1, off=pos_y[p,:], seed=0)
		fxn_z[p] = FXN._Point(amp=1, off=pos_z[p,:], seed=0)
	
	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def TEST_BLUR():
	## Parameters ##
	# CHIP: 64x64	(px)
	# RES:	0.0685	(um/px)
	# FPS:	1		(frames/s)
	# LEN:	50		(frames)

	# SNR:	      4
	# DEN:	      4
	# BLUR: 1, 2, 4, 7, 10
	P = 1

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	np.random.seed(0)
	pos_x = 0.7*(np.random.rand(P, USER.SIM_FRAMES) - 0.5) * (USER.CHIP[0] * USER.RES[0])
	pos_y = 0.7*(np.random.rand(P, USER.SIM_FRAMES) - 0.5) * (USER.CHIP[1] * USER.RES[1])
	pos_z = 0.7*(np.random.rand(P, USER.SIM_FRAMES) - 0.5) * (USER.DOF[0])
	for p in range(P):
		# Make sure the particles aren't too close if you can avoid it #
		for f in range(USER.SIM_FRAMES):
			dist = [np.sqrt((pos_x[p,f] - pos_x[q,f])**2 + (pos_y[p,f] - pos_y[q,f])**2 + (pos_z[p,f] - pos_z[q,f])**2) for q in range(p)]
			for _ in range(10):
				if((p > 0) and (min(dist) < USER.CHIP[0]*USER.RES[0]/P)):
					# Generate a new random position #
					pos_x[p,f] = 0.7*(np.random.rand() - 0.5) * (USER.CHIP[0] * USER.RES[0])
					pos_y[p,f] = 0.7*(np.random.rand() - 0.5) * (USER.CHIP[1] * USER.RES[1])
					pos_z[p,f] = 0.7*(np.random.rand() - 0.5) * (USER.DOF[0])

					# Recompute the distance #
					dist = [np.sqrt((pos_x[p,f] - pos_x[q,f])**2 + (pos_y[p,f] - pos_y[q,f])**2 + (pos_z[p,f] - pos_z[q,f])**2) for q in range(p)]

		# Place the particles down #
		fxn_x[p] = FXN._Point(amp=1, off=pos_x[p,:], seed=0)
		fxn_y[p] = FXN._Point(amp=1, off=pos_y[p,:], seed=0)
		fxn_z[p] = FXN._Point(amp=1, off=pos_z[p,:], seed=0)
	
	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w

def STReM():
	## Parameters ##
	# CHIP: 64x64	(px)
	# RES:	0.0685	(um/px)
	# FPS:	1		(frames/s)
	# LEN:	3		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	np.random.seed(0)
	fxn_x[0] = FXN._Cosine(1/3, amp=1, off=2, seed=0)
	fxn_y[0] = FXN._Sine(1/3, amp=1, off=2, seed=0)
	fxn_z[0] = FXN._Point(amp=1, off=0, seed=0)

	fxn_x[1] = FXN._Cosine(1/12, np.pi, amp=4, off=2, seed=0)
	fxn_y[1] = FXN._Sine(1/12, np.pi, amp=4, off=2, seed=0)
	fxn_z[1] = FXN._Point(amp=1, off=0, seed=0)
	
	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def STReM_easy():
	## Parameters ##
	# CHIP: 96x96	(px)
	# RES:	0.0685	(um/px)
	# FPS:	1		(frames/s)
	# LEN:	4		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	np.random.seed(0)
	fxn_x[0] = FXN._Line(amp=-2, off=1, seed=0)
	fxn_y[0] = FXN._Line(amp=2, off=-1, seed=0)
	fxn_z[0] = FXN._Point(amp=1, off=0, seed=0)	
	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def STReM_med():
	## Parameters ##
	# CHIP: 96x96	(px)
	# RES:	0.0685	(um/px)
	# FPS:	1		(frames/s)
	# LEN:	4		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	np.random.seed(0)
	fxn_x[0] = FXN._Cosine(1/12, amp=3, off=-1.0, seed=0)
	fxn_y[0] = FXN._Sine(1/12, amp=3, off=-1.5, seed=0)
	fxn_z[0] = FXN._Point(amp=1, off=0, seed=0)	
	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def STReM_hard():
	## Parameters ##
	# CHIP: 96x96	(px)
	# RES:	0.0685	(um/px)
	# FPS:	4		(frames/s)
	# LEN:	4		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	np.random.seed(0)
	fxn_x[0] = FXN._Wiener(amp=1, off=0, seed=4)
	fxn_y[0] = FXN._Wiener(amp=3/4, off=2.5, seed=5)
	fxn_z[0] = FXN._Point(amp=1, off=0, seed=0)	
	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def STReM_3D():
	## Parameters ##
	# CHIP: 64x64	(px)
	# RES:	0.0685	(um/px)
	# FPS:	1		(frames/s)
	# LEN:	1		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	np.random.seed(0)
	fxn_y[0] = FXN._Point(amp=1, off=0, seed=0)
	fxn_x[0] = FXN._Point(amp=1, off=0, seed=0)
	fxn_z[0] = FXN._Line(amp=1.6, off=-0.8, seed=0)

	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w

def Easy_DHPSF():
	## Parameters ##
	# CHIP: 64x64	(px)
	# RES:	0.0685	(um/px)
	# FPS:	1		(frames/s)
	# LEN:	3		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	fxn_x[0] = FXN._Point(amp=1, off=0, seed=0)
	fxn_y[0] = FXN._Point(amp=1, off=0, seed=0)
	fxn_z[0] = FXN._Line(amp=1, off=-2, seed=0)

	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def Easy_DHPSF_Dark():
	## Parameters ##
	# CHIP: 64x64	(px)
	# RES:	0.0685	(um/px)
	# FPS:	1		(frames/s)
	# LEN:	3		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	fxn_x[0] = FXN._Point(amp=1, off=0, seed=0)
	fxn_y[0] = FXN._Point(amp=1, off=0, seed=0)
	fxn_z[0] = FXN._Line(amp=1, off=-2, seed=0)
	fxn_w[0] = FXN._Point(amp=1, off=0, seed=0)

	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w

#%% IMAGE GENERATION %%#
## Switch statement ##

# UtGK manuscript figures #
if(code == 'FIG 1a'):		fxns = FIG_1a()
elif(code == 'FIG 1b'):		fxns = FIG_1b()
elif(code == 'FIG 1di'):	fxns = FIG_1di()
elif(code == 'FIG 1dii'):	fxns = FIG_1dii()
elif(code == 'FIG 1diii'):	fxns = FIG_1diii()
elif(code == 'FIG 2x'):		fxns = FIG_2x()
elif(code == 'FIG 3b'):		fxns = FIG_3b()
elif(code == 'FIG 6a'):		fxns = FIG_6a()
elif(code == 'FIG 6b'):		fxns = FIG_6b()
elif(code == 'FIG 6c'):		fxns = FIG_6c()
elif(code == 'FIG 6x'):		fxns = FIG_6x()
elif(code == 'TOC'):		fxns = TOC()
elif(code == 'TOC3D'):		fxns = TOC()

# Boundary testing #
# SNR:	1, 2, 4, 7, 10
# DEN:	1, 2, 4, 7, 10	(Particles)
# BLUR: 1, 2, 4, 7, 10	(um/s)
elif(code[:4] == 'ADMM'):	fxns = ADMM(int(code[5:]))
elif(code == 'LOC'):		fxns = TEST_LOC()
elif(code[:3] == 'SNR' and "Easy" not in code):	fxns = TEST_SNR(int(code[4:]))
elif(code[:3] == 'DEN' and "Easy" not in code):	fxns = TEST_DEN(int(code[4:]))
elif(code[:4] == 'BLUR'):	fxns = TEST_BLUR()

# Sub-frame testing #
elif(code == 'STReM'):		fxns = STReM()
elif(code == 'STReM_easy'):	fxns = STReM_easy()
elif(code == 'STReM_med'):	fxns = STReM_med()
elif(code == 'STReM_hard'):	fxns = STReM_hard()
elif(code == 'STReM_3D'):	fxns = STReM_3D()

# Easy-DHPSF calibration #
elif(code == 'Easy_DHPSF'):	fxns = Easy_DHPSF()
elif(code == 'Easy_DHPSF_Dark'):	fxns = Easy_DHPSF_Dark()

elif(code[:8] == 'SNR_Easy'):	fxns = TEST_SNR(int(code[9:]))
elif(code[:8] == 'DEN_Easy'):	fxns = TEST_DEN(int(code[9:]))

## Image generation ##
scope = Microscope(code, *fxns, blur=blur)
if(np.max(scope.img) > 0):
	scope.img /= np.max(scope.img)

# Add noise #
if(snr == 0): snr = 0.5
scope.img = (20*(snr**1.3)*scope.img + np.random.poisson(100, np.shape(scope.img)))/((1 + snr))

# Change the motion to voxels #
mot = np.zeros_like(scope.mot)
mot[:,:,0] = (-scope.mot[:,:,0]/USER.RES[0]) + USER.CHIP[1]/2
mot[:,:,1] = (-scope.mot[:,:,1]/USER.RES[1]) + USER.CHIP[0]/2
mot[:,:,2] = np.round((scope.mot[:,:,2]/USER.DOF[0] + 1/2) * USER.KER_Z)
mot[:,:,3] = scope.mot[:,:,3]
mot[:,:,4] = scope.mot[:,:,4]

# Save #
OP._SaveMov(scope.img, code, OP.FOLD_SIM, fmt=FMT.GIF)
OP._SaveMov(scope.img, code, OP.FOLD_SIM, fmt=FMT.TIF)
OP._SaveSim(scope.mot, code)
OP._SaveMot(mot, code)
spi.savemat(code + '.mat', {'mot':mot})

#if(USER.KER_Z == 1):
#	VIS.FIG1a(mot, scope.img)
#else:
#	VIS.FIG1b(mot, scope.img)

## Visualization ##
plt.figure(figsize=(4,4))
if(True):#USER.KER_Z == 1): 
	ax = plt.axes(position=[0,0,1,1])
	ax.imshow(scope.img[-1,0,:,:], cmap='gray')
	for p in range(np.shape(mot)[1]):
		ax.plot(mot[:,p,0], mot[:,p,1], marker='o')
	ax.invert_yaxis()
else:
	clr = plt.get_cmap('tab20').colors
	ax = plt.axes(position=[-0.15,-0.15,1.3,1.3], projection='3d')
	VIS._Disp3Dimg(ax, scope.img, stride=2)
	for p in range(np.shape(mot)[1]):
		ax.plot(mot[:,p,0], mot[:,p,1], mot[:,p,2], color=clr[2*p], marker='o', zorder=1000 - np.mean(mot[:,p,1]))
		ax.plot(mot[:,p,0], mot[:,p,1], 0*mot[:,p,2], color=clr[2*p+1], linestyle='--', zorder=500 - np.mean(mot[:,p,1]))
	ax.view_init(15,-120)
	ax.set_zlim(0, USER.KER_Z)
	ax.set_zticklabels([])
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.show()
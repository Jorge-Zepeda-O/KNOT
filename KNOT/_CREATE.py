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

#%% SIMULATION DEFINITIONS %%#
# A word of note: simulations use units! Please write your parameters down! #
def Test_Function():
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
	fxn_x[0] = FXN._Line(amp=2, off=-2)
	fxn_y[0] = FXN._Poly(1,0, amp=1, off=-1)

	fxn_x[1] = FXN._Line(amp=2, off=-2)
	fxn_y[1] = FXN._Exp(0,-1) * FXN._Sine() + FXN._Line(amp=-2, off=1)

	fxn_x[2] = FXN._Cosine(amp=FXN._Exp(0,-1,amp=4))
	fxn_y[2] = FXN._Sine(amp=FXN._Exp(0,-1,amp=4))

	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w
def Test_Mixed():
	## Parameters ##
	# CHIP: 128x128	(px)
	# RES:	0.0685	(um/px)
	# DOF:	4.000	(um)
	# FPS:	40		(frames/s)
	# LEN:	100		(frames)

	## Initialization ##
	fxn_x = dict()
	fxn_y = dict()
	fxn_z = dict()
	fxn_w = dict()

	## Plotting ##
	# Initialization #
	imp_x = FXN._Point()
	imp_y = FXN._Point()
	imp_z = FXN._Point()

	# Impulse Generation #
	np.random.seed(0)	# For repeatability #
	for i in range(2):
		# Random generation of time and strength #
		mu = USER.SIM_FRAMES/USER.FPS * np.random.rand()
		amp = 30 * (np.random.rand(3) - 1/2)

		# Concatenate 0.5 second impulses #
		imp_x = imp_x + FXN._Gauss(mu, 0.1, amp=2*amp[0])
		imp_y = imp_y + FXN._Gauss(mu, 0.1, amp=2*amp[1])
		imp_z = imp_z + FXN._Gauss(mu, 0.1, amp=amp[2])

	# Function definition #
	fxn_x[0] = FXN._Wiener(imp_x, 0.5, off=0, seed=1)
	fxn_y[0] = FXN._Wiener(imp_y, 0.5, off=2.5, seed=2)
	fxn_z[0] = FXN._Wiener(imp_z, 0.5, off=1, seed=3)

	## Output ##
	return fxn_x, fxn_y, fxn_z, fxn_w

#%% FUNCTION GENERATION %%#
def _GetFunctions(code):
	# This is just a glorified switch statement #
	if(code == 'Test_function'):		return Test_Function()
	elif(code == 'Test_mixed'):			return Test_Mixed()
	else: raise Exception('Code not defined: %s' % (code))
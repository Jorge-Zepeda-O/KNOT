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
code = 'Test_mixed'
snr = 4

#%% INITIALIZE %%#
# Make sure all needed folders exist #
OP._MakeDir(OP.FOLD_APR)
OP._MakeDir(OP.FOLD_KER)
OP._MakeDir(OP.FOLD_IMG)

OP._MakeDir(OP.FOLD_SIM)
OP._MakeDir(OP.FOLD_TMP)
OP._MakeDir(OP.FOLD_EVL)
OP._MakeDir(OP.FOLD_MAT)

OP._MakeDir(OP.FOLD_TRUE)

#%% SIMULATION %%#
# Import _CREATE.py to get the simulation definitions and properly obtain our functions #
import _CREATE
fxns = _CREATE._GetFunctions(code)

### Image Generation ###
scope = Microscope(code, *fxns)
if(np.max(scope.img) > 0):
	scope.img /= np.max(scope.img)

# Add noise #
if(snr == 0): snr = 0.5
scope.img = (20*(snr**1.3)*scope.img + np.random.poisson(100, np.shape(scope.img)))/((1 + snr))

# Change the motion to voxels #
mot = np.zeros_like(scope.mot)
mot[:,:,0] = (-scope.mot[:,:,0]/USER.RES[0]) + USER.CHIP[1]/2
mot[:,:,1] = (-scope.mot[:,:,1]/USER.RES[1]) + USER.CHIP[0]/2
mot[:,:,2] = (scope.mot[:,:,2]/USER.DOF[0] + 1/2) * USER.KER_Z
mot[:,:,3] = scope.mot[:,:,3]
mot[:,:,4] = scope.mot[:,:,4]

### Save ###
OP._SaveMov(scope.img, code, OP.FOLD_SIM, fmt=FMT.GIF)
OP._SaveMov(scope.img, code, OP.FOLD_SIM, fmt=FMT.TIF)
OP._SaveSim(scope.mot, code)
OP._SaveMot(mot, code)
spi.savemat(code + '.mat', {'mot':mot})

### Visualization ###
plt.figure(figsize=(4,4))
ax = plt.axes(position=[0,0,1,1])
ax.imshow(scope.img[-1,0,:,:], cmap='gray')
for p in range(np.shape(mot)[1]):
	ax.plot(mot[:,p,0], mot[:,p,1], marker='o')
ax.set_xticklabels([])
ax.set_yticklabels([])

if(USER.KER_Z > 1): 
	plt.figure(figsize=(4,4))
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
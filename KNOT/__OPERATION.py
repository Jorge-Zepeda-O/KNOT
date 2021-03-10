#%% IMPORTS %%#
### External ###
from datetime	import datetime
from os			import path, mkdir
import glob
import json
import imageio

import multiprocessing	as mp
import numpy 			as np
import scipy.io			as spi

### Internal ###
from __ENUM import LOCALIZATION as LOCAL
from __ENUM import FILEFORMAT 	as FMT

import USER

#%% --- STATIC PARAMETERS --- %%#
### Input/Output ###
VER = 'v1_0'								# The current version #
DIR = path.dirname(path.abspath(__file__))	# Current directory #

## Folder Paths ##
FOLD_APR = DIR + '\\Apertures\\'	# Folder to load apertures		#
FOLD_KER = DIR + '\\Phase Masks\\'	# Folder to load phase masks	#
FOLD_IMG = DIR + '\\Images\\'		# Folder to load taken images 	#

FOLD_SIM = DIR + '\\Simulation\\'	# Folder to store simulations 					#
FOLD_TMP = DIR + '\\Temp\\'			# Folder to store temporary/in progress data 	#
FOLD_EVL = DIR + '\\Evaluation\\'	# Folder to store data ready for evaluation 	#
FOLD_MAT = DIR + '\\Matlab\\'		# Folder to store data as if Troika gave it 	#

FOLD_TRUE = DIR + '\\Truth\\'		# Folder to use to compare to ground truths		#

#%% --- I/O METHODS --- %%#
### Existence ###
def _CheckDir(fold):
	"""
	Checks if the given directory `fold` exists
	"""
	return path.isdir(fold)
def _CheckTemp(code, fold=FOLD_TMP):
	"""
	Checks if the given code-directory exists in the TMP foler
	"""
	return path.isdir(fold + '%s\\' % code)
def _CheckCode(code, file, fold=FOLD_TMP, fmt=FMT.TIF):
	"""
	Checks if the given code-directory has a specific file. Format is `'%s\\%s - ' + file`
	"""
	return path.exists(fold + ('%s\\%s' % (code, code)) + file + str(fmt))
def _CheckFile(file, fold=FOLD_TMP):
	"""
	Checks if the given file exists
	"""
	return path.exists(fold + file)

### Creation ###
def _MakeDir(fold):
	""" 
	Makes a new directory `fold` in the local directory
	"""
	if(not _CheckDir(fold)): mkdir(fold)
def _MakeCode(code, fold=FOLD_TMP):
	"""
	Makes a new code-directory in the TMP folder
	"""
	if(not _CheckTemp(code, fold)): mkdir(fold + '%s\\' % code)

### Saving ###
## Images ## 
def _SaveImg(img, file, fold=FOLD_TMP, *, fmt=FMT.TIF):
	"""
	Writes the singular 2D array `img` [Y,X] to .tif format
	"""
	## Write to File ##
	imageio.imwrite(fold + file + str(fmt), img)
def _SaveMov(mov, file, fold=FOLD_TMP, *, fmt=FMT.TIF):
	"""
	Writes multiple 2D arrays `mov` [F,Z,Y,X] to a multilayer .tif format or .gif movie
	"""
	# Check if desired format is .gif #
	if(fmt is FMT.GIF):
		# Normalize appropriately for best viewing experience #
		mov_ = np.mean(mov, 1)
		mov_ = np.minimum(np.maximum((mov_ - np.min(mov_)) / np.ptp(mov_), 0), 1)
		mov = (mov_ * 255).astype(np.uint8)

		# Write to file #
		imageio.mimwrite(fold + file + str(fmt), mov)		# Normal image #
	else:
		# Write to file #
		if(np.shape(mov)[1] == 1):
			imageio.mimwrite(fold + file + str(fmt), mov[:,0,:,:].astype(np.int16))	# Normal image #
		else:
			imageio.mvolwrite(fold + file + str(fmt), np.moveaxis(mov, 1, 3).astype(np.int16))	# Z-stack enabled #
def _SaveKer(ker, file, fold=FOLD_TMP, *, fmt=FMT.TIF):
	"""
	Writes the 4D array `ker` [T,Z,Y,X] to a multilayer .tif image
	"""
	# Reorganize to maximize I/O speed #
	ker_ = np.moveaxis(ker, (0,1), (3,2))

	## Write to File ##
	imageio.volwrite(fold + file + str(fmt), ker_)

## Data ##
def _SaveJSON(data, file, fold=FOLD_TMP):
	"""
	Writes serializable data `data` to a .json file
	"""
	## Write to File ##
	with open(fold + file + str(FMT.JSON), 'w') as jfile:
		json.dump(data, jfile)

def _SaveTracks(traj, code, *, snr=7, den='low', dset='VESICLE'):
	## Save to XML ##
	# Header #
	root = '<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n<root>\n'
	content = '<TrackContestISBI2012 SNR=\"%s\" density=\"%s\" ' % (snr, den) \
	   + 'generationDateTime=\"%s\" ' % (datetime.now().strftime("%d/%m/%Y %H:%M:%S")) \
	   + 'info=\"\" scenario=\"%s\">\n' % (dset)

	# Convert Particles to XML #
	for p in traj:
		# Create a new particle event #
		str_part = '<particle>\n'

		# Evaluate all detection events #
		for f in range(int(min(p.frm)), int(max(p.frm))+1):
			# Check if the particle is exhibiting a jump #
			if(f in p.frm):
				# No jump, write down the particle's position #
				c = np.where(f == p.frm)[0][0]
				str_det = '<detection t=\"%i\" x=\"%0.3f\" y=\"%0.3f\" z=\"%0.3f\" />\n' \
					% (f, p.res[c,0], p.res[c,1], p.res[c,2])
			else:
				# Interpolate through the blink #
				c_hi = np.where(f < p.frm)[0][0]	# Last occurence #
				c_lo = np.where(f > p.frm)[0][-1]	# First occurence #
				
				# Get linear interpolation coefficients #
				hi_val = (p.frm[c_hi] - f)/(p.frm[c_hi] - p.frm[c_lo])
				lo_val = (f - p.frm[c_lo])/(p.frm[c_hi] - p.frm[c_lo])
				
				# Interpolate #
				res = lo_val * p.res[c_lo,:] + hi_val * p.res[c_hi,:]
				str_det = '<detection t=\"%i\" x=\"%0.3f\" y=\"%0.3f\" z=\"%0.3f\" />\n' \
					% (f, res[0], res[1], res[2])

			# Append the detection event to the particle #
			str_part += str_det
		# Append the particle to `content` (and close it) #
		content += str_part + '</particle>\n'
	# Append the content to the root (and close it) #
	root += content + '</TrackContestISBI2012>\n</root>'

	## Save to MAT ##
	# Initialize #
	F = int(max([max(p.frm) for p in traj])) + 1
	P = len(traj)
	D = 3 if USER.KER_Z > 1 else 2
	
	trjR = np.zeros((F, D, P))

	# Report Values #
	for p in range(P):
		for f in range(len(traj[p].frm)):
			trjR[int(traj[p].frm[f]), :, p] = traj[p].res[f,:D]

	## Write to File ##
	with open(FOLD_EVL + code + str(FMT.XML), 'w') as xfile:
		xfile.write(root)
	spi.savemat(FOLD_MAT + code + str(FMT.MAT), {'trjR':trjR})
def _SaveSim(data, code, fold=FOLD_SIM):
	"""
	Saves the simulation information into XML to be read by the tracking evaluator.
	"""
	## Initialize ##
	F, P, D = np.shape(data)

	## Header Information ##
	root = '<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n<root>\n'
	content = '<TrackContestISBI2012 SNR=\"\" density=\"\" ' \
	   + 'generationDateTime=\"%s\" ' % (datetime.now().strftime("%d/%m/%Y %H:%M:%S")) \
	   + 'info=\"SIMULATION\" scenario=\"%s\">\n' % (code)

	## Convert Motion to XML ##
	for p in range(P):
		# Create a new particle event #
		str_part = '<particle>\n'

		# Evaluate all detection events #
		for f in range(F):
			# Check if the particle exists #
			if(data[f,p,4] < 1E-3): continue

			# Write down the detection event #
			str_det = '<detection t=\"%i\" x=\"%0.3f\" y=\"%0.3f\" z=\"%0.3f\" />\n' \
				% (int(np.round(data[f,p,3] * USER.FPS)), \
				-data[f,p,0] / USER.RES[0] + USER.CHIP[0]/2, \
				-data[f,p,1] / USER.RES[1] + USER.CHIP[1]/2, \
				+(data[f,p,2]/USER.DOF[0 if USER.KER_Z > 1 and USER.KER_T == 1 else 1]+ 1/2)*USER.KER_Z)
			
			# Append the detection event to the particle #
			str_part += str_det
		# Append the particle to `content` (and close it) #
		content += str_part + '</particle>\n'
	# Append the content to the root (and close it) #
	root += content + '</TrackConstestISBI2012>\n</root>'

	## Write to File ##
	with open(fold + code + str(FMT.XML), 'w') as xfile:
		xfile.write(root)

def _SaveMot(mot, code, fold=FOLD_TRUE):
	"""
	Saves the motion contained by `mot` into a .json file.
	"""
	mot_json = mot.tolist()
	_SaveJSON(mot_json, code + ' mot', fold)

### Loading ###
## Images ##
def _LoadImg(file, fold=FOLD_TMP, *, fmt=FMT.TIF):
	"""
	Reads the singular 2D array `img` [Y,X] from .tif format
	"""
	## Read From File ##
	return np.array(imageio.imread(fold + file + str(fmt)))
def _LoadMov(file, fold=FOLD_TMP, *, fmt=FMT.TIF):
	"""
	Reads multiple 2D arrays `mov` [F,Z,Y,X] from a multilayer .tif (organized as [F,Y,X,Z]!)
	"""
	## Read from File ##
	# Turn off `memtest` because we're liable to read some very large files #
	mov = np.array(imageio.mvolread(fold + file + str(fmt), memtest=False)[0])
	if(len(np.shape(mov)) == 2):
		mov = mov[None,None,:,:]
	elif(len(np.shape(mov)) == 3):
		mov = mov[:,None,:,:]
	else:	# This allows z-stack imaging! #
		mov = np.moveaxis(mov, 3, 1)
	return mov
def _LoadKer(file, fold=FOLD_TMP, *, fmt=FMT.TIF):
	"""
	Reads the 4D array `ker` [T,Z,Y,X] from a multilayer .tif image
	"""
	## Read from File ##
	return np.moveaxis(np.array(imageio.volread(fold + file + str(fmt))), (3,2), (0,1))

## Data ##
def _LoadJSON(file, fold=FOLD_TMP):
	"""
	Reads serializable data `data` from a .json file
	"""
	## Read from File ##
	with open(fold + file + str(FMT.JSON)) as jfile:
		data = json.load(jfile)

	## Output ##
	return data
def _LoadTruth(code, fold=FOLD_TRUE):
	from _SEGMENT	import PointCloud
	from _TRACK		import Particle

	## Initialization ##
	traj = list()

	## Read XML File ##
	if(_CheckFile(code + str(FMT.XML), fold)):
		with open(fold + code + str(FMT.XML), 'r') as xfile:
			# Read all lines #
			data = xfile.readlines()

			# Parse particles and detections #
			for l in range(len(data)):	# Skip first three lines and last two #
				if('<p' in data[l]):
					# Start a new particle #
					clouds = list()
				elif('<d' in data[l]):
					# Read a detection event #
					det = data[l].split('\"')	# Format: <detection t="#" x="#" y="#" z="#" /> #

					# Create a `PointCloud` from it #
					clouds.append(PointCloud(np.array([[\
						float(det[3]), float(det[5]), float(det[7]), float(det[1]), 1]])))
				elif('</p' in data[l]):
					# End of the particle #
					traj.append(Particle(clouds))

	## Output ##
	return traj
def _LoadMot(code, fold=FOLD_TRUE):
	mot = _LoadJSON(code + ' mot', fold);
	return np.array(mot)
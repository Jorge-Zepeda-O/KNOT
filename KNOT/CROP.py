#%% --- IMPORTS --- %%#
### External ###
from os import path
import numpy as np
import imageio
import glob
import matplotlib.pyplot as plt

### Internal ###
import __OPERATION as OP

#%% --- USER PARAMETERS --- %%#
## Processing parameters ##
UPDATE = True	# Update rois?			#
VISUAL = True	# Visualize results?	#

Frames = [100, 110]		# <(f, f)> Frame range, does not include end frame. Zero indexed	#
Roi_cen = [230, 320]	# <(px, px)> Center point of the ROI								#
Roi_size = [64, 64]		# <(px, px)> Size of the ROI. Keep equal! Else the phase mask simulates incorrectly (BUG) #

## Files for processing ##
DIR = path.dirname(path.abspath(__file__))
FOLD_EXP = DIR + '\\Experiment\\'
OP._MakeDir(FOLD_EXP)	# Make the folder if it doesn't exist #

# If you want to specify files in the folder, uncomment the second line, else comment it out to grab everything in the folder #
files = glob.glob(FOLD_EXP + '*.tif')
files = ['Test Data.tif']

## Additional parameters ##
# Construct the x and y ranges for cropping #
rng_x = range(Roi_cen[0] - Roi_size[0]//2, Roi_cen[0] + Roi_size[0]//2)
rng_y = range(Roi_cen[1] - Roi_size[1]//2, Roi_cen[1] + Roi_size[1]//2)

# The footer for information about how these images were cropped #
footer = '_(%i,%i)_(%ix%i)_(f%i-%i)' % (*Roi_cen, *Roi_size, *Frames)

#%% LOAD AND CROP %%#
print("Starting...")
for f in files:
	filename = f.split('\\')[-1]
	f_name = FOLD_EXP + filename if (FOLD_EXP not in f) else f

	# Skip over already cropped files #
	if(filename[:3] == 'roi'): continue
	print(filename, end='\t')

	# Loading - KEEP IN MIND: THIS IS FOR SINGLE PLANE, MONOCHROMATIC IMAGES #
	data = np.array(imageio.mimread(f_name, memtest=False))
	while(len(np.shape(data)) > 3): data = data[0,...]
	print('| Loaded |', end='\t')

	# Visualize before cropping #
	if(VISUAL):
		plt.figure(figsize=(12,6))
		ax = plt.axes(position=[0,0,0.5,1])
		ax.imshow(data[Frames[0],:,:], cmap='gray')
		ax.plot([rng_x[0], rng_x[0], rng_x[-1], rng_x[-1], rng_x[0]], [rng_y[0], rng_y[-1], rng_y[-1], rng_y[0], rng_y[0]], c='r')
		ax.text(10, 20, "Frame %i" % (Frames[0]), color='w')
		ax.set_xticklabels([])
		ax.set_yticklabels([])

		ax = plt.axes(position=[0.5,0,0.5,1])
		ax.imshow(data[Frames[1],:,:], cmap='gray')
		ax.plot([rng_x[0], rng_x[0], rng_x[-1], rng_x[-1], rng_x[0]], [rng_y[0], rng_y[-1], rng_y[-1], rng_y[0], rng_y[0]], c='r')
		ax.text(10, 20, "Frame %i" % (Frames[1]), color='w')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		plt.show()

	# Cropping #
	if(not UPDATE and path.exists(FOLD_EXP + 'roi_' + filename[:-4] + footer + '.tif')): continue
	data = data[Frames[0]:Frames[1],:,:][:,rng_y,:][:,:,rng_x]
	imageio.mimwrite(FOLD_EXP + 'roi_' + filename[:-4] + footer + '.tif', data.astype(np.uint16))
	print('| Cropped |', end='\n')
print("Finished!")
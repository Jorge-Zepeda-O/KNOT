from os import path
import numpy as np
import imageio
import glob
import matplotlib.pyplot as plt

#%% --- USER PARAMETERS --- %%#
UPDATE = True
VISUAL = True

Frames = [0, 100]
Roi_cen = [367, 197]
Roi_size = [128, 128]
#Frames = [10, 30]		# (Start, End)		# 1/100
#Roi_cen = [200, 240]	# (X, Y)			# 900/760
#Roi_size = [128, 128]	# (Width, Height)	# 360/360

rng_x = range(Roi_cen[0] - Roi_size[0]//2, Roi_cen[0] + Roi_size[0]//2)
rng_y = range(Roi_cen[1] - Roi_size[1]//2, Roi_cen[1] + Roi_size[1]//2)
footer = '_(%i,%i)_(%ix%i)_(f%i-%i)' % (*Roi_cen, *Roi_size, *Frames)

#%% LOAD AND CROP %%#
DIR = path.dirname(path.abspath(__file__))
FOLD_EXP = DIR + '\\Experiment\\'

files = glob.glob(FOLD_EXP + '*.tif')
print("Starting...")
for f in files:
	filename = f.split('\\')[-1]

	# Skip over already cropped files #
	if(filename[:3] == 'roi'): continue
	if(filename[:2] == 'B3'): continue
	print(filename, end='\t')

	# Loading - KEEP IN MIND: THIS IS FOR SINGLE PLANE, MONOCHROMATIC IMAGES #
	data = np.array(imageio.mimread(f, memtest=False))
	while(len(np.shape(data)) > 3): data = data[0,...]
	print('| Loaded |', end='\t')

	# Visualize before cropping #
	if(VISUAL):
		plt.figure()
		plt.gca(position=[0,0,1,1])
		plt.imshow(data[0,:,:])
		plt.plot([rng_x[0], rng_x[0], rng_x[-1], rng_x[-1], rng_x[0]], [rng_y[0], rng_y[-1], rng_y[-1], rng_y[0], rng_y[0]], c='w')
		plt.show()

	# Cropping #
	if(not UPDATE and path.exists(FOLD_EXP + 'roi_' + filename[:-4] + footer + '.tif')): continue
	data = data[Frames[0]:Frames[1],:,:][:,rng_y,:][:,:,rng_x]
	imageio.mimwrite(FOLD_EXP + 'roi_' + filename[:-4] + footer + '.tif', data.astype(np.uint16))
	print('| Cropped |', end='\n')
print("Finished!")
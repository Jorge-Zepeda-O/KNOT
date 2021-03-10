#%% --- IMPORTS --- %%#
### External ###
import numpy 				as np
import scipy				as sp
import matplotlib.pyplot 	as plt
from mpl_toolkits.mplot3d	import Axes3D
from matplotlib.collections import LineCollection as LC
from mpl_toolkits.mplot3d.art3d import Line3DCollection as L3C
import colorsys
import sys

import os
import imageio
import time

### Internal ###
from __ENUM	import FILEFORMAT	as FMT
from _SEGMENT		import PointCloud
from _TRACK			import _SFD
import __OPERATION	as OP
import USER

#%% --- CONSOLE --- %%#
def _ProgressBar(i, I, sub_i=0, sub_I=0, prefix='', suffix='', decimals=3, length=25, fill = '\'', sub_fill=',', both_fill='|', end='\r'):
	"""
    Call in a loop to create terminal progress bar
    @params:
    iteration   - Required  : current iteration (Int)
    total       - Required  : total iterations (Int)
    prefix      - Optional  : prefix string (Str)
    suffix      - Optional  : suffix string (Str)
    decimals    - Optional  : positive number of decimals in percent complete (Int)
    length      - Optional  : character length of bar (Int)
    fill        - Optional  : bar fill character (Str)
    printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
	
	filledLength = int(length * i // I)

	if(sub_I > 0):
		percent = ("{0:6." + str(decimals) + "f}").format(100 * ((i - 1 + sub_i/sub_I)/float(I)))

		subLength = int(length * sub_i // sub_I)
		bothLength = min(subLength, filledLength)
		noneLength = length - max(subLength, filledLength)
		bar = both_fill*bothLength

		if(subLength > filledLength):
			bar = bar + sub_fill*(subLength - filledLength)
		elif(filledLength > subLength):
			bar = bar + fill*(filledLength - subLength)
	else:
		percent = ("{0:6." + str(decimals) + "f}").format(100 * (i/float(I)))
		noneLength = length - filledLength
		bar = both_fill * filledLength

	bar = bar + ' ' * noneLength
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=end)
    # Print New Line on Complete
	if i == I: 
		sys.stdout.flush()

#%% --- STATIC METHODS --- %%#
def _DispSFDDist(data, wgt, kde, mesh, clr, bnds):
	# Initialize #
	plt.figure(figsize=(4,2))
	plt.gca().set_position([0,0,1,1])

	# Plot the histogram and KDE #
	h,_,_ = plt.hist(data.flatten(), weights=wgt.flatten(), bins=mesh[::4], color=clr, alpha=0.6, density=True)
	plt.plot(mesh, kde * np.max(h), color='k', linewidth=5)

	# Drop the mean value #
	plt.plot(np.mean(data)*np.ones(2), [0,np.max(h)], 'k', linestyle='--', linewidth=5)

	# Set boundaries #
	plt.xlim(*bnds)
def _Disp3Dimg(ax, img, zz=None, stride=4):
	# Initialize #
	xx, yy = np.meshgrid(range(np.shape(img)[3]), range(np.shape(img)[2]))
	if(zz is None):					zz = 0 * xx
	elif(len(np.shape(zz)) < 2):	zz = zz + 0*xx
	imclr = np.repeat((img/np.max(img))[-1,0,:,:,None], 3, axis=2)

	# Plot the surface #
	ax.plot_surface(xx, yy, zz, facecolors=imclr, rstride=stride, cstride=stride, zorder=-10)

	# Adjust axes #
	ax.set_xlim(0, np.shape(img)[3])
	ax.set_ylim(0, np.shape(img)[2])
def _DispLineGrad(ax, seg, frm, color, bot=False, fmin=0, fmax=0, sat_rng=None, bot_rng=None, val_rng=None, **kwargs):
	# Establish the endpoints of the color gradient #
	if(fmin == 0): fmin = np.min(frm)
	if(fmax == 0): fmax = np.max(frm)
	frng = fmax - fmin
	F = len(frm)
	if(sat_rng is None):
		sat_rng =	[1/2 + 1/2 * (np.min(frm)-fmin)/frng, 1/2 + 1/2 * (np.max(frm)-fmin)/frng]
	if(bot_rng is None):
		bot_rng =	[1/4 + 1/2 * (np.min(frm)-fmin)/frng, 1/4 + 1/2 * (np.max(frm)-fmin)/frng]

	if(val_rng is None):
		val_rng =	[1 - 1/3 * (np.min(frm)-fmin)/frng, 1 - 1/3 * (np.max(frm)-fmin)/frng]
		if(bot):	val_vals = np.full(F, 1)
		else:		val_vals = np.concatenate((np.full(F-2*F//3, val_rng[0]), np.linspace(*val_rng, 2*F//3)))
	else:
		val_vals = np.linspace(*val_rng, F)


	# Convert the RGB color to HSV so we can modulate the saturation and value #
	chsv = colorsys.rgb_to_hsv(*color)
	if(bot):
		sat_vals = np.concatenate((np.linspace(*bot_rng, 2*F//3), np.full(F-2*F//3, bot_rng[1])))
	else:
		sat_vals = np.concatenate((np.linspace(*sat_rng, 2*F//3), np.full(F-2*F//3, sat_rng[1])))

	crgb_rng = list(map(colorsys.hsv_to_rgb, np.full(F, chsv[0]), sat_vals, val_vals))
	crgb_rng = np.maximum(np.minimum(crgb_rng, 1), 0)	# Make sure nothing explodes #

	# Check if we're pplotting in 2D or 3D, and add the appropriate linecollection #
	if(isinstance(ax, Axes3D)):
		coll = L3C(seg, colors=crgb_rng, **kwargs)	# z order, not like it matters #
	else:
		coll = LC(seg, colors=crgb_rng, **kwargs)

	ax.add_collection(coll)
	return crgb_rng
def _VisImg(img, posx, posy=50, tru=None):
	plt.figure()
	plt.get_current_fig_manager().window.setGeometry(posx,posy,450,500)
	ax = plt.axes(position=[0,0,1,1])
	ax.imshow(img, cmap='gray')
	if(tru is not None):
		plt.plot(tru[:,0], tru[:,1], color='r', marker='o', markersize=6, linewidth=0, fillstyle='none')

#%% --- MANUSCRIPT FIGURES --- %%#
def FIG1a(mot, img):
	colors = plt.get_cmap('tab10').colors
	colors = [colors[0], colors[2], colors[3], colors[1]]

	plt.figure(figsize=(4,4))
	ax = plt.axes(position=[0,0,1,1])
	ax.imshow(img[-1,0,:,:], cmap='gray')
	for p in range(np.shape(mot)[1]):
		F = np.shape(mot)[0]
		seg = [(mot[f,p,:2].tolist(), mot[f+1,p,:2].tolist()) for f in range(F-1)]
		_DispLineGrad(ax, seg, np.arange(F-1), colors[p], linewidth=6)
		
	ax.invert_yaxis()
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	plt.show()
def FIG1b(mot, img):
	colors = plt.get_cmap('tab10').colors

	plt.figure(figsize=(4,4))
	ax = plt.axes(projection='3d', position=[-0.15,-0.15,1.3,1.3])
	_Disp3Dimg(ax, img, stride=2)

	pos = mot[:,:,:3]
	pos_bot = pos * [1,1,0]
	for p in range(np.shape(mot)[1]):
		F = np.shape(mot)[0]
		seg = [(pos[f,p,:].tolist(), pos[f+1,p,:].tolist()) for f in range(F-1)]
		seg_bot = [(pos_bot[f,p,:].tolist(), pos_bot[f+1,p,:].tolist()) for f in range(F-1)]
		_DispLineGrad(ax, seg, np.arange(F-1), colors[p], linewidth=6)
		_DispLineGrad(ax, seg_bot, np.arange(F-1), colors[p], bot=True, linewidth=3, linestyle='--')

	ax.view_init(15,-120)
	ax.set_zlim(0, USER.KER_Z)
	ax.set_zticklabels([])
	ax.set_xticklabels([])
	ax.set_yticklabels([])

	plt.show()

def FIG2x(pos, wgt, img):
	# Dataset: FIG 1dii
	# Parameters:
	#	CHIP = 32x32
	#	KER_Z = 18
	#	REC_ITER = 0, 15, 60, 240

	## Initialize ##
	F, Z, Y, X = np.shape(img)
	clr = [(0.1,0.5,0.7), (1.0,0.2,0.5), (1.0,0.6,0.3)]
	#clr = [(0.1,0.5,0.7), (0.1,0.5,0.7), (0.1,0.5,0.7)]

	## Form the point cloud ##
	np.random.seed(0)
	if(USER.REC_ITER > 0):
		pts = np.concatenate([pos, wgt[:,None]], axis=1)
		#pts[:,:3] += ((np.random.rand(np.shape(pts)[0], 3) - 0.5).T*(1 - wgt / (2*np.max(wgt)))).T
		cloud = PointCloud(pts, seg=True)
	else:
		# ADMM explores everything in the initial step #
		pos = np.array([*USER.CHIP, USER.KER_Z]) * (np.random.rand(2**13, 3))
		frm = np.zeros([2**13, 1])
		wgt = np.full([2**13, 1], 0.001)
		pts = np.concatenate([pos, frm, wgt], axis=1)
		cloud = PointCloud(pts)

	tru = [-np.array([-0.25,0.25])/USER.RES[0] + USER.CHIP[0]/2, \
		   -np.array([0.1,0.1])/USER.RES[1] + USER.CHIP[1]/2, \
		   np.array([4/3,-4/3])*USER.KER_Z/USER.DOF[0] + USER.KER_Z/2]
	atru = [-np.array([-0.25,0.25]) + USER.RES[0]*USER.CHIP[0]/2, \
		   -np.array([0.1, 0.1]) + USER.RES[1]*USER.CHIP[1]/2, \
		   np.array([4/3,-4/3]) + USER.DOF[0]/2]
	
	## Visualization ##
	plt.figure(figsize=(4,4))
	ax = plt.axes(projection='3d', position=[-0.15,-0.12,1.25,1.25])
	_Disp3Dimg(ax, img, stride=1)
	k = 0

	if(USER.REC_ITER == 240):	k_pick = [0, 2]
	elif(USER.REC_ITER == 100):	k_pick = [1, 2]#[11, 31]
	#elif(USER.REC_ITER == 15):	k_pick = [10, 24]
	else:						k_pick = [-1, -1]

	if(USER.REC_ITER > 0):
		for c in cloud.clust:
			# So the colors don't mess up #
			if(len(c.wgt) < 4):
				k += 1
				continue

			# Highlight the point clouds in question #
			if(k == k_pick[0]):
				ax.scatter(c.pos[:,0], c.pos[:,1], c.pos[:,2], s=240*c.wgt, c=clr[1], alpha=0.6, depthshade=False)
			elif(k == k_pick[1]):
				ax.scatter(c.pos[:,0], c.pos[:,1], c.pos[:,2], s=240*c.wgt, c=clr[2], alpha=0.6, depthshade=False)
			else:
				if(USER.REC_ITER == 100):
					ax.scatter(c.pos[:,0], 35-c.pos[:,1], c.pos[:,2], s=240*c.wgt, c=clr[0], alpha=0.6, depthshade=False)
				else:
					ax.scatter(c.pos[:,0], c.pos[:,1], c.pos[:,2], s=240*c.wgt, c=clr[0], alpha=0.6, depthshade=False)
			k += 1
	else:
		ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=240*wgt, c=clr[0], alpha=0.6, depthshade=False)
	ax.scatter(tru[0], tru[1], tru[2], s=360, c='k', marker='+', linewidth=3, depthshade=False)
	ax.set_xlim(0, X)
	ax.set_ylim(0, Y)
	ax.set_zlim(0, USER.KER_Z)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_zticklabels([])
	ax.view_init(15, -120)
	#plt.show()

	# Show Zoom-ins #
	if(-1 not in k_pick):
		for k in range(len(k_pick)):
			c = cloud.clust[k_pick[k]]

			plt.figure(figsize=(2,2))
			ax = plt.axes(projection='3d', position=[-0.15,-0.12,1.25,1.25])

			ax.scatter(c.pos[:,0], c.pos[:,1], c.pos[:,2], s=240*c.wgt, c=clr[k+1], alpha=0.6, depthshade=False)
			ax.scatter(c.res[0], c.res[1], c.res[2], s=360, c='k', marker='x', linewidth=3, depthshade=False)
			ax.scatter(tru[0], tru[1], tru[2], s=480, c='k', marker='+', linewidth=3, depthshade=False)

			ax.set_xlim(c.res[0] - 0.25/USER.RES[0], c.res[0] + 0.25/USER.RES[0])
			ax.set_ylim(c.res[1] - 0.25/USER.RES[1], c.res[1] + 0.25/USER.RES[1])
			ax.set_zlim(c.res[2] - 0.25*USER.KER_Z/USER.DOF[0], c.res[2] + 0.25*USER.KER_Z/USER.DOF[0])
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_zticklabels([])
			ax.view_init(15, -120)

			err = np.sqrt( np.min(np.sum([(c.ares[d] - atru[d])**2 for d in range(3)], axis=0)) )
			print('l2 error: %5.3f um' % err)
	plt.show()
def FIG2del(pos, wgt, img):
	# Dataset: FIG 1dii
	# Parameters:
	#	CHIP = 32x32
	#	KER_Z = 18
	#	REC_ITER = 0, 15, 60, 240

	## Initialize ##
	F, Z, Y, X = np.shape(img)
	clr = [(0.1,0.5,0.7), (1.0,0.2,0.5), (1.0,0.6,0.3)]

	## Form the point cloud ##
	np.random.seed(0)
	pts = np.concatenate([pos, wgt[:,None]], axis=1)
	pts[:,:3] += ((np.random.rand(np.shape(pts)[0], 3) - 0.5).T*(1 - wgt / (2*np.max(wgt)))).T
	#pts[:,0] = 32 - pts[:,0]
	cloud = PointCloud(pts, seg=True)

	tru = [-np.array([-0.20,0.20])/USER.RES[0] + USER.CHIP[0]/2, \
		   -np.array([0.1,0.1])/USER.RES[1] + USER.CHIP[1]/2, \
		   np.array([4/3,-4/3])*USER.KER_Z/USER.DOF[0] + USER.KER_Z/2]
	atru = [-np.array([-0.20,0.20]) + USER.RES[0]*USER.CHIP[0]/2, \
		   -np.array([0.1, 0.1]) + USER.RES[1]*USER.CHIP[1]/2, \
		   np.array([4/3,-4/3]) + USER.DOF[0]/2]

	## Triangulate ##
	import scipy.spatial as spt
	tc = PointCloud(pts)#[pts[:,2] < 16, :])
	tri = spt.Delaunay(tc.pos)
	g_seg = list()
	r_seg = list()
	S, V = np.shape(tri.simplices)
	for s in tri.simplices:
		for u in range(V-1):
			for v in range(u+1, V):
				vdist = np.sum((tc.pos[s[u],:] - tc.pos[s[v],:])**2)
				if(vdist < 3):	g_seg.append([tc.pos[s[u],:], tc.pos[s[v],:]])
				else:			r_seg.append([tc.pos[s[u],:], tc.pos[s[v],:]])

	## Visualization ##
	plt.figure(figsize=(4,4))
	ax = plt.axes(projection='3d', position=[-0.15,-0.12,1.25,1.25])
	_Disp3Dimg(ax, img, stride=1)
	k = 0

	if(USER.REC_ITER == 240):	k_pick = [0, 1]
	elif(USER.REC_ITER == 100):	k_pick = [1, 2]#60 - [11, 31]
	#elif(USER.REC_ITER == 15):	k_pick = [10, 24]
	else:						k_pick = [-1, -1]

	ax.add_collection(L3C(r_seg, color='r', alpha=0.2, linewidth=0.5))
	ax.add_collection(L3C(g_seg, color='g', alpha=0.2, linewidth=0.5))

	for c in cloud.clust:
		# So the colors don't mess up #
		if(len(c.wgt) < 4):
			k += 1
			continue

		# Highlight the point clouds in question #
		if(k == k_pick[0]):
			ax.scatter(c.pos[:,0], c.pos[:,1], c.pos[:,2], s=240*c.wgt, c=clr[1], alpha=0.8, depthshade=False)
		elif(k == k_pick[1]):
			ax.scatter(c.pos[:,0], c.pos[:,1], c.pos[:,2], s=240*c.wgt, c=clr[2], alpha=0.8, depthshade=False)
		else:
			ax.scatter(c.pos[:,0], c.pos[:,1], c.pos[:,2], s=60*c.wgt, alpha=0.8, depthshade=False)
		k += 1
	ax.set_xlim(0, X)
	ax.set_ylim(0, Y)
	ax.set_zlim(0, USER.KER_Z)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_zticklabels([])
	ax.view_init(15, -120)
	#plt.show()

	if(-1 not in k_pick):
		for k in range(len(k_pick)):
			c = cloud.clust[k_pick[k]]

			plt.figure(figsize=(4,4))
			ax = plt.axes(projection='3d', position=[-0.15,-0.12,1.25,1.25])

			ax.add_collection(L3C(r_seg, color='r', alpha=0.2))
			ax.add_collection(L3C(g_seg, color='g', alpha=0.2))

			ax.scatter(c.pos[:,0], c.pos[:,1], c.pos[:,2], s=960*c.wgt, c=clr[k+1], alpha=0.6, depthshade=False)
			ax.scatter(c.res[0], c.res[1], c.res[2], s=360, c='k', marker='x', linewidth=3, depthshade=False)
			#ax.scatter(tru[0], tru[1], tru[2], s=480, c='k', marker='+', linewidth=3, depthshade=False)

			ax.set_xlim(c.res[0] - 2, c.res[0] + 2)
			ax.set_ylim(c.res[1] - 2, c.res[1] + 2)
			ax.set_zlim(c.res[2] - 2, c.res[2] + 2)

			#ax.set_xlim(c.res[0] - 0.20/USER.RES[0], c.res[0] + 0.20/USER.RES[0])
			#ax.set_ylim(c.res[1] - 0.20/USER.RES[1], c.res[1] + 0.20/USER.RES[1])
			#ax.set_zlim(c.res[2] - 0.20*USER.KER_Z/USER.DOF[0], c.res[2] + 0.20*USER.KER_Z/USER.DOF[0])
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_zticklabels([])
			ax.view_init(15, -120)

			err = np.sqrt( np.min(np.sum([(c.ares[d] - atru[d])**2 for d in range(3)], axis=0)) )
			print('l2 error: %5.3f um' % err)
	#plt.show()
def FIG2cin(pos, wgt, img):
	# Dataset: FIG 1dii
	# Parameters:
	#	CHIP = 32x32
	#	KER_Z = 18
	#	REC_ITER = 0, 15, 60, 240

	## Initialize ##
	F, Z, Y, X = np.shape(img)
	clr = [(0.1,0.5,0.7), (1.0,0.2,0.5), (1.0,0.6,0.3)]
	clr = plt.get_cmap('tab10').colors
	clr = [*clr[:8], clr[9]]

	## Form the point cloud ##
	np.random.seed(0)
	pts = np.concatenate([pos, wgt[:,None]], axis=1)
	pts[:,:3] += ((np.random.rand(np.shape(pts)[0], 3) - 0.5).T*(1 - wgt / (2*np.max(wgt)))).T
	#pts[:,0] = 32 - pts[:,0]
	cloud = PointCloud(pts, seg=True)

	tru = [-np.array([-0.20,0.20])/USER.RES[0] + USER.CHIP[0]/2, \
		   -np.array([0.1,0.1])/USER.RES[1] + USER.CHIP[1]/2, \
		   np.array([4/3,-4/3])*USER.KER_Z/USER.DOF[0] + USER.KER_Z/2]
	atru = [-np.array([-0.20,0.20]) + USER.RES[0]*USER.CHIP[0]/2, \
		   -np.array([0.1, 0.1]) + USER.RES[1]*USER.CHIP[1]/2, \
		   np.array([4/3,-4/3]) + USER.DOF[0]/2]

	## Triangulate ##
	import scipy.spatial as spt
	tc = PointCloud(pts)#[pts[:,2] < 16, :])
	tri = spt.Delaunay(tc.pos)
	g_seg = list()
	r_seg = list()
	S, V = np.shape(tri.simplices)
	for s in tri.simplices:
		for u in range(V-1):
			for v in range(u+1, V):
				vdist = np.sum((tc.pos[s[u],:] - tc.pos[s[v],:])**2)
				if(vdist < 3):	g_seg.append([tc.pos[s[u],:2], tc.pos[s[v],:2]])
				else:			r_seg.append([tc.pos[s[u],:2], tc.pos[s[v],:2]])

	## Visualization ##
	plt.figure(figsize=(4,4))
	ax = plt.axes(position=[0,0,1,1])
	#_Disp3Dimg(ax, img, stride=1)
	k = 0

	if(USER.REC_ITER == 240):	k_pick = [0, 1]
	elif(USER.REC_ITER == 100):	k_pick = [1, 2]#60 - [11, 31]
	#elif(USER.REC_ITER == 15):	k_pick = [10, 24]
	else:						k_pick = [-1, -1]

	ax.add_collection(LC(r_seg, color='r', alpha=0.2, linewidth=0.5))
	ax.add_collection(LC(g_seg, color='g', alpha=0.5, linewidth=1))

	for c in cloud.clust:
		# So the colors don't mess up #
		if(len(c.wgt) < 4):
			k += 1
			continue

		# Highlight the point clouds in question #
		if(k == k_pick[0]):
			ax.scatter(c.pos[:,0], c.pos[:,1], s=240*c.wgt, c=clr[1], alpha=0.8)
		elif(k == k_pick[1]):
			ax.scatter(c.pos[:,0], c.pos[:,1], s=240*c.wgt, c=clr[2], alpha=0.8)
		else:
			ax.scatter(c.pos[:,0], c.pos[:,1], s=240*c.wgt, c=clr[np.mod(k, len(clr))], alpha=1)
		k += 1
	ax.set_xlim(X/2, X)
	ax.set_ylim(0, Y/2)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	#plt.show()

	if(-1 not in k_pick):
		for k in range(len(k_pick)):
			c = cloud.clust[k_pick[k]]

			plt.figure(figsize=(4,4))
			ax = plt.axes(projection='3d', position=[-0.15,-0.12,1.25,1.25])

			ax.add_collection(L3C(r_seg, color='r', alpha=0.2))
			ax.add_collection(L3C(g_seg, color='g', alpha=0.2))

			ax.scatter(c.pos[:,0], c.pos[:,1], c.pos[:,2], s=960*c.wgt, c=clr[k+1], alpha=0.6, depthshade=False)
			ax.scatter(c.res[0], c.res[1], c.res[2], s=360, c='k', marker='x', linewidth=3, depthshade=False)
			#ax.scatter(tru[0], tru[1], tru[2], s=480, c='k', marker='+', linewidth=3, depthshade=False)

			ax.set_xlim(c.res[0] - 2, c.res[0] + 2)
			ax.set_ylim(c.res[1] - 2, c.res[1] + 2)
			ax.set_zlim(c.res[2] - 2, c.res[2] + 2)

			#ax.set_xlim(c.res[0] - 0.20/USER.RES[0], c.res[0] + 0.20/USER.RES[0])
			#ax.set_ylim(c.res[1] - 0.20/USER.RES[1], c.res[1] + 0.20/USER.RES[1])
			#ax.set_zlim(c.res[2] - 0.20*USER.KER_Z/USER.DOF[0], c.res[2] + 0.20*USER.KER_Z/USER.DOF[0])
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_zticklabels([])
			ax.view_init(15, -120)

			err = np.sqrt( np.min(np.sum([(c.ares[d] - atru[d])**2 for d in range(3)], axis=0)) )
			print('l2 error: %5.3f um' % err)
	plt.show()

def FIG3a(p:PointCloud, q:PointCloud, img):
	## Connection is from p to q ##
	# Dataset: ADMM 2 hi
	#	p = clouds[5]
	#	q = clouds[14]
	# Parameters:
	#	KER_Z = 16
	#	TRK_KDE = 180
	#	TRK_RAD = 1.500
	
	## Imports ##
	from _TRACK		import _SFD, _SFDwgt, _LKDE
	
	# "Blur" for visualization #
	np.random.seed(0)
	p.pos = p.pos + ((np.random.rand(*np.shape(p.pos)) - 0.5).T*(1 - p.wgt / np.max(p.wgt))).T
	q.pos = q.pos + ((np.random.rand(*np.shape(q.pos)) - 0.5).T*(1 - q.wgt / np.max(q.wgt))).T

	## Cloud connection figure ##
	plt.figure(figsize=(8,8))
	ax = plt.gca(projection='3d', position=[-0.15,-0.15,1.3,1.3])

	# Cloud colors #
	clr_p = np.array([0, 73, 73])/255
	clr_q = np.array([0,109,219])/255

	# Connection lines #
	lines = 7
	clr_lines = np.outer(np.linspace(0,1,lines+1), clr_p) + \
				np.outer(np.linspace(1,0,lines+1), clr_q)

	# Link colors #
	clr_rho = (0.9,0.7,0.0)
	clr_theta = (0.7,0.4,1.0)
	clr_phi = (0.6,0.6,0.6)

	# Draw clouds #
	ax.scatter(p.pos[:,0], p.pos[:,1], p.pos[:,2], s=900*p.wgt, c=clr_p, alpha=0.5, depthshade=False)
	ax.scatter(q.pos[:,0], q.pos[:,1], q.pos[:,2], s=900*q.wgt, c=clr_q, alpha=0.5, marker='s', depthshade=False)

	# Draw lines where needed #
	thr_wgt = 0.1
	for i in range(np.shape(p.pos)[0]):
		if(p.wgt[i] < thr_wgt): continue
		for j in range(np.shape(q.pos)[0]):
			if(q.wgt[j] < thr_wgt): continue
			for l in range(lines):
				# For ease of reading #
				pnt1 = (l/lines)*p.pos[i,:] + (1-l/lines)*q.pos[j,:]
				pnt2 = ((l+1)/lines)*p.pos[i,:] + (1-(l+1)/lines)*q.pos[j,:] 
				width = 5 * np.minimum(2 * min(p.wgt[i], q.wgt[j])**1.5, 1)
				alpha = np.minimum(np.maximum(1 * min(p.wgt[i], q.wgt[j])**1.5, 0.2), 0.7)

				# Plot the lines #
				ax.plot([pnt1[0], pnt2[0]], [pnt1[1], pnt2[1]], [pnt1[2], pnt2[2]], \
					c=clr_lines[l,:], linewidth=width, alpha=alpha, zorder=1)

	# Mark the centroids #
	pcen = np.sum(p.wgt * p.pos.T, axis=1)/np.sum(p.wgt)
	qcen = np.sum(q.wgt * q.pos.T, axis=1)/np.sum(q.wgt)
	ax.plot([pcen[0], qcen[0]], [pcen[1], qcen[1]], [pcen[2], qcen[2]], c='k', \
		linewidth=0, marker='x', markersize=40, markeredgewidth=5, zorder=100)

	# Mark the dropdowns for spherical coordinate marking #
	pass
	
	# Determine the axis limits and viewing angle #
	xrng = [min(min(p.pos[:,0]), min(q.pos[:,0]))-0, max(max(p.pos[:,0]), max(q.pos[:,0]))+0]
	yrng = [min(min(p.pos[:,1]), min(q.pos[:,1]))-0, max(max(p.pos[:,1]), max(q.pos[:,1]))+0]
	zrng = [min(min(p.pos[:,2]), min(q.pos[:,2]))-2, max(max(p.pos[:,2]), max(q.pos[:,2]))+0]

	xcen = np.mean(xrng)
	ycen = np.mean(yrng)
	rad = np.maximum(np.ptp(xrng), np.ptp(yrng)) - 1
	ax.set_xlim(xcen - rad/2, xcen + rad/2)
	ax.set_ylim(ycen - rad/2, ycen + rad/2)
	ax.set_zlim(*zrng)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_zticklabels([])
	#ax.set_xlabel("X")
	ax.view_init(15, -120)

	## SFD Distribution Figures ##
	# Find the SFD #
	rho, phi = _SFD(p, q, units=True)
	omega = _SFDwgt(p, q)

	# Get the KDEs #
	rhoKDE = _LKDE(rho.flatten(), omega.flatten(), USER.MESH_RHO)
	phiKDE = _LKDE(phi[:,:,0].flatten(), omega.flatten(), USER.MESH_PHI, period=2*np.pi)
	thetaKDE = _LKDE(phi[:,:,1].flatten(), omega.flatten(), USER.MESH_PHI, period=0)

	# Draw SFD distributions #
	_DispSFDDist(rho, omega, rhoKDE, USER.MESH_RHO, clr_rho, [0, USER.TRK_RAD])
	_DispSFDDist(phi[:,:,1], omega, thetaKDE, USER.MESH_PHI, clr_theta, [0, np.pi])
	_DispSFDDist(phi[:,:,0], omega, phiKDE, USER.MESH_PHI, clr_phi, [-np.pi, 0])
	
	## Show ##
	plt.show()
def FIG3asel(p:PointCloud, q:PointCloud, img, c:PointCloud, o:PointCloud):
	## Connection is from p to q ##
	# Dataset: ADMM 2 hi
	#	p = clouds[5]
	#	q = clouds[14]
	#	c = clouds[3] [11]
	# Parameters:
	#	KER_Z = 16
	#	TRK_KDE = 180
	#	TRK_RAD = 1.500
	
	## Imports ##
	from _TRACK		import _SFD, _SFDwgt, _LKDE
	
	# "Blur" for visualization #
	np.random.seed(0)
	p.pos = p.pos + ((np.random.rand(*np.shape(p.pos)) - 0.5).T*(1 - p.wgt / np.max(p.wgt))).T
	q.pos = q.pos + ((np.random.rand(*np.shape(q.pos)) - 0.5).T*(1 - q.wgt / np.max(q.wgt))).T
	c.pos = c.pos + ((np.random.rand(*np.shape(c.pos)) - 0.5).T*(1 - c.wgt / np.max(c.wgt))).T
	o.pos = o.pos + ((np.random.rand(*np.shape(o.pos)) - 0.5).T*(1 - o.wgt / np.max(o.wgt))).T

	## Cloud connection figure ##
	plt.figure(figsize=(8,8))
	ax = plt.gca(position=[0,0,1,1])

	# Cloud colors #
	clr_p = np.array([0, 73, 73])/255
	clr_q = np.array([0,109,219])/255
	clr_c = np.array([109,0,219])/255
	clr_o = np.array([0,0,0])/255

	# Draw clouds #
	ax.scatter(p.pos[:,0], p.pos[:,1], s=900*p.wgt, c=clr_p, alpha=0.4)
	ax.scatter(q.pos[:,0], q.pos[:,1], s=900*q.wgt, c=clr_q, alpha=0.4, marker='s')
	ax.scatter(c.pos[:,0], c.pos[:,1], s=900*c.wgt, c=clr_c, alpha=0.4, marker='s')
	ax.scatter(o.pos[:,0], o.pos[:,1], s=900*o.wgt, c=clr_o, alpha=0.4, marker='s')

	# Draw lines where needed #
	"""
	thr_wgt = 0.1
	for i in range(np.shape(p.pos)[0]):
		if(p.wgt[i] < thr_wgt): continue
		for j in range(np.shape(q.pos)[0]):
			if(q.wgt[j] < thr_wgt): continue
			for l in range(lines):
				# For ease of reading #
				pnt1 = (l/lines)*p.pos[i,:] + (1-l/lines)*q.pos[j,:]
				pnt2 = ((l+1)/lines)*p.pos[i,:] + (1-(l+1)/lines)*q.pos[j,:] 
				width = 5 * np.minimum(2 * min(p.wgt[i], q.wgt[j])**1.5, 1)
				alpha = np.minimum(np.maximum(1 * min(p.wgt[i], q.wgt[j])**1.5, 0.2), 0.7)

				# Plot the lines #
				ax.plot([pnt1[0], pnt2[0]], [pnt1[1], pnt2[1]], [pnt1[2], pnt2[2]], \
					c=clr_lines[l,:], linewidth=width, alpha=alpha, zorder=1)
	"""

	# Mark the centroids #
	pcen = np.sum(p.wgt * p.pos.T, axis=1)/np.sum(p.wgt)
	qcen = np.sum(q.wgt * q.pos.T, axis=1)/np.sum(q.wgt)
	ccen = np.sum(c.wgt * c.pos.T, axis=1)/np.sum(c.wgt)
	c2cen = np.sum(o.wgt * o.pos.T, axis=1)/np.sum(o.wgt)
	ax.plot([pcen[0], qcen[0], ccen[0], c2cen[0]], [pcen[1], qcen[1], ccen[1], c2cen[1]], c='k', \
		linewidth=0, marker='x', markersize=40, markeredgewidth=5, zorder=100)

	# Determine the axis limits and viewing angle #
	xrng = [min(min(p.pos[:,0]), min(q.pos[:,0]), min(c.pos[:,0]), min(o.pos[:,0]))-0, max(max(p.pos[:,0]), max(q.pos[:,0]), max(c.pos[:,0]), max(o.pos[:,0]))+0]
	yrng = [min(min(p.pos[:,1]), min(q.pos[:,1]), min(c.pos[:,1]), min(o.pos[:,1]))-0, max(max(p.pos[:,1]), max(q.pos[:,1]), max(c.pos[:,1]), max(o.pos[:,1]))+0]

	xcen = np.mean(xrng)
	ycen = np.mean(yrng)
	rad = np.maximum(np.ptp(xrng), np.ptp(yrng))+1

	ax.plot([xcen-0.5/USER.RES[0], xcen+0.5/USER.RES[0]], [ycen, ycen], c='k')

	print(rad)
	ax.set_xlim(xcen - rad/2, xcen + rad/2)
	ax.set_ylim(ycen - rad/2, ycen + rad/2)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
		
	## Show ##
	plt.show()
def FIG3b(traj, img):
	# Dataset: FIG 3b
	# Parameters:
	#	TRK_LEN = 6
	#	TRK_KDE = 180
	#	TRK_RAD = 1.500
	#clr = [(1,0,0), (1,0.5,0), (0.7,0.7,0), (0,1,0), (0,0.5,0.5), (0,0,1)][::-1];

	# The image + trajectory #
	plt.figure(figsize=(4,4))
	ax = plt.axes(position=[0,0,1,1])
	ax.imshow(img[-1,0,:,:], cmap='gray')
	for p in traj:
		ax.plot(p.res[:,0], p.res[:,1], 'w', linewidth=3, marker='o')
		seg = [(p.res[-f-1,:2].tolist(), p.res[-f-2,:2].tolist()) for f in range(USER.TRK_LEN)]
		clr = _DispLineGrad(ax, seg[::-1], np.arange(USER.TRK_LEN), [1,0,1], linewidth=6)
		#plt.plot(p.res[-2:,0], p.res[-2:,1], color=clr[0], linewidth=6, marker='o', markersize=12)
		#for k in range(1, USER.TRK_LEN):
		#	plt.plot(p.res[-k-2:-k,0], p.res[-k-2:-k,1], color=clr[k], linewidth=6, marker='o', markersize=12)

	clr = clr[::-1]
	## Rho KDEs ##
	# Individual #
	plt.figure(figsize=(4,2))
	plt.gca(position=[0,0,1,1])
	for p in traj:
		for k in range(USER.TRK_LEN)[::-1]:
			plt.plot(USER.MESH_RHO, p.rho[:,k], color=clr[k], linewidth=4)
	plt.xlim(0, USER.TRK_RAD)
	plt.ylim(0, 1.1)
	plt.xticks([])
	plt.yticks([])

	# Characteristic #
	plt.figure(figsize=(4,2))
	plt.gca(position=[0,0,1,1])
	for p in traj:
		plt.plot(USER.MESH_RHO, np.mean(p.rho, 1), 'k', linewidth=6)
	plt.xlim(0, USER.TRK_RAD)
	plt.ylim(0, 1.1)
	plt.xticks([])
	plt.yticks([])

	## Phi KDEs ##
	# Individual #
	plt.figure(figsize=(4,2))
	plt.gca(position=[0,0,1,1])
	for p in traj:
		for k in range(USER.TRK_LEN)[::-1]:
			plt.plot(USER.MESH_PHI, p.phi[:,k,0], color=clr[k], linewidth=4)
	plt.xlim(-np.pi, np.pi)
	plt.ylim(0, 1.1)
	plt.xticks([])
	plt.yticks([])

	# Characteristic #
	plt.figure(figsize=(4,2))
	plt.gca(position=[0,0,1,1])
	for p in traj:
		plt.plot(USER.MESH_PHI, np.mean(p.phi[:,:,0], 1), 'k', linewidth=6)
	plt.xlim(-np.pi, np.pi)
	plt.ylim(0, 1.1)
	plt.xticks([])
	plt.yticks([])

	plt.show()
def FIG3c(traj, img):
	# Dataset: FIG 1b
	#	traj = [traj[2], traj[1], traj[0], traj[3]]
	# Parameters:
	#	KER_Z = 16
	#	TRK_LEN = 10
	#	TRK_KDE = 120
	#	TRK_RAD = 1.500

	clr = plt.get_cmap('tab10').colors

	## The Trajectories ##
	plt.figure(figsize=(4,4))
	ax = plt.axes(projection='3d', position=[-0.15,-0.15,1.3,1.3])
	_Disp3Dimg(ax, img, stride=2)
	k = 0
	for p in traj:
		F = len(p.frm)
		seg = [(p.res[f,:].tolist(), p.res[f+1,:].tolist()) for f in range(F-1)]
		seg_bot = [((p.res[f,:]*[1,1,0]).tolist(), (p.res[f+1,:]*[1,1,0]).tolist()) for f in range(F-1)]
		_DispLineGrad(ax, seg, p.frm, clr[k % len(clr)], linewidth=6)
		_DispLineGrad(ax, seg_bot, p.frm, clr[k % len(clr)], bot=True, linewidth=3, linestyle='--')
		ax.text(np.mean(p.res[:,0]), np.mean(p.res[:,1]), np.mean(p.res[:,2])+1, k)
		k += 1
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_zticklabels([])
	ax.set_zlim(0, USER.KER_Z)
	ax.view_init(15, -120)

	## Characteristic SFDs ##
	plt.figure(figsize=(4,2))
	plt.gca(position=[0,0,1,1])
	for p in traj:
		plt.plot(USER.MESH_RHO, np.mean(p.rho, 1), linewidth=6)
	plt.xlim(0, USER.TRK_RAD)
	plt.ylim(0, 1.1)
	plt.xticks([])
	plt.yticks([])

	plt.figure(figsize=(4,2))
	plt.gca(position=[0,0,1,1])
	for p in traj:
		plt.plot(np.sin(USER.MESH_PHI[USER.TRK_KDE//2:] - np.pi/2), np.mean(p.phi[USER.TRK_KDE//2:,:,1], 1), linewidth=6)
	plt.xlim(-1, 1)
	plt.ylim(0, 1.1)
	plt.xticks([])
	plt.yticks([])

	plt.figure(figsize=(4,2))
	plt.gca(position=[0,0,1,1])
	for p in traj:
		plt.plot(USER.MESH_PHI, np.mean(p.phi[:,:,0], 1), linewidth=6)
	plt.xlim(-np.pi, np.pi)
	plt.ylim(0, 1.1)
	plt.xticks([])
	plt.yticks([])
	
	plt.show()

def FIG4(code):
	print('loading')
	#img = OP._LoadMov(code, fold=OP.FOLD_IMG)
	truth = OP._LoadTruth(code)
	knot = OP._LoadTruth(code + '_KNOT')
	#troika = OP._LoadXML(code + '_Troika')
	trackmate = OP._LoadTruth(code + '_TrackMate')
	utrack = OP._LoadTruth(code + '_ConstVelocity')

	c_truth = np.array([0,0,0])/255
	c_knot = np.array([0,109,219])/255
	c_trackmate = np.array([255,109,182])/255
	c_utrack = np.array([36,209,36])/255

	ct_truth = [0.0,0.0,0.0]
	ct_knot = [0.5,0.0,0.3]
	ct_trackmate = [0.0,0.3,0.8]
	ct_utrack = [0.0,0.6,0.0]

	if(False):
		print('plotting main')
		plt.figure(figsize=(8, 8))
		plt.gca(position=[0,0,1,1])
		#plt.imshow(img[-1,:,:], cmap='gray')

		for p in truth:
			if(len(p.res[:,0]) < 20): continue
			plt.plot(p.res[:,0], p.res[:,1], c=c_truth, linewidth=4)
		#for p in troika:	plt.plot(p.res[:,0], p.res[:,1], c=[1.0,0.0,0.0])
		for p in trackmate:	
			if(len(p.res[:,0]) < 15): continue
			plt.plot(p.res[:,0], p.res[:,1], c=c_trackmate, linewidth=2)
		for p in utrack:	
			if(len(p.res[:,0]) < 15): continue
			plt.plot(p.res[:,0], p.res[:,1], c=c_utrack, linewidth=2)
		for p in knot:		
			if(len(p.res[:,0]) < 15): continue
			plt.plot(p.res[:,0], p.res[:,1], c=c_knot, linewidth=2)

		plt.show()
	## Zoom-ins ##
	xlocs = [226,33, 144, 363,461,461, 306, 250, 279,279,168]
	ylocs = [260,314, 28, 318,236,236, 68, 265, 426,426,500]
	num_truth = [[1],[6],[0],[2],[5],[0],[1],[0],[7],[2],[0]]	# [2,5]
	num_knot = [[1,5],[5],[0],[2,4],[5],[1],[1],[0],[5],[2],[0]]	# [2,4,5]
	num_trackmate = [[6,10,17,14,3,7],[0,15,14,20,2],[20,17,19,14,9,21,10,8,3],[2,8,10,12,18],[15,14,12],[13,11,4,7,1],[11,10],[0],[20,18,15],[6],[0,2,4]]  # [2,24,8,10,20,23,12,18] [2,24,8,10,12,18]
	num_utrack = [range(30),range(30),range(30),[2,12],range(30),range(30),[3,5],[1,2],range(30),[3],range(30)]
	#num_utrack = [[10,5],[12,13,16,15,0],[12,13,9,7,5,1],[3],[6,9],[8,7,3],[6],[1],[16,14,11],[3],[0,2]] # [3, 11]#

	#zoom = [16, 20, 52, 32, 24, 24, 28,32, 28, 20, 24]
	zoom = [32, 32, 32, 30, 32, 32, 30, 30, 32, 30, 32]
	#		1   2   3   4   5   6   7  8   9   10   11
	xcen = [228,34,144,363,475,461,305,249,277,282,170]
	ycen = [260,316,29,318,230,234,65,265,424,448,499]
	
	#xlocs = [13,193,465,86,62]
	#ylocs = [167,257,222,344,55]
	for n in [3,6,7,9]:#range(len(xlocs)):
		## ANALysis ##
		truth_ = [p for p in truth if \
			np.all(p.res[:,0] > xlocs[n]-32) and np.all(p.res[:,0] < xlocs[n]+32) and \
			np.all(p.res[:,1] > ylocs[n]-32) and np.all(p.res[:,1] < ylocs[n]+32)]
		knot_ = [p for p in knot if \
			np.all(p.res[:,0] > xlocs[n]-32) and np.all(p.res[:,0] < xlocs[n]+32) and \
			np.all(p.res[:,1] > ylocs[n]-32) and np.all(p.res[:,1] < ylocs[n]+32)]
		trackmate_ = [p for p in trackmate if \
			np.all(p.res[:,0] > xlocs[n]-32) and np.all(p.res[:,0] < xlocs[n]+32) and \
			np.all(p.res[:,1] > ylocs[n]-32) and np.all(p.res[:,1] < ylocs[n]+32)]
		utrack_ = [p for p in utrack if \
			np.all(p.res[:,0] > xlocs[n]-32) and np.all(p.res[:,0] < xlocs[n]+32) and \
			np.all(p.res[:,1] > ylocs[n]-32) and np.all(p.res[:,1] < ylocs[n]+32)]

		#truth__ = [truth_[p] for p in range(len(truth_)) if p in num_truth[n]]
		#knot__ = [knot_[p] for p in range(len(knot_)) if p in num_knot[n]]
		#trackmate__ = [trackmate_[p] for p in range(len(trackmate_)) if p in num_trackmate[n]]
		#utrack__ = [utrack_[p] for p in range(len(utrack_)) if p in num_utrack[n]]

		#OP._SaveTracks(truth__, 'VESICLE 2 low crop %i' % n, snr=2)
		#OP._SaveTracks(knot__, 'VESICLE 2 low crop %i KNOT' % n, snr=2)
		#OP._SaveTracks(trackmate__, 'VESICLE 2 low crop %i TrackMate' % n, snr=2)
		#OP._SaveTracks(utrack__, 'VESICLE 2 low crop %i const' % n, snr=2)
		
		fmin = np.min(truth_[num_truth[n][0]].frm)
		fmax = np.max(truth_[num_truth[n][0]].frm)

		plt.figure(num=n+1, figsize=(4,4))
		ax = plt.gca(position=[0,0,1,1])
		for p in range(len(knot_)):		
			if(p not in num_knot[n]): continue
			F = len(knot_[p].frm)
			seg = [(knot_[p].res[f,:2].tolist(), knot_[p].res[f+1,:2].tolist()) for f in range(F-1)]
			_DispLineGrad(ax, seg, knot_[p].frm, c_knot, fmin=fmin, fmax=fmax, linewidth=6)

		for p in range(len(utrack_)):
			if(p not in num_utrack[n]): continue
			F = len(utrack_[p].frm)
			seg = [(utrack_[p].res[f,:2].tolist(), utrack_[p].res[f+1,:2].tolist()) for f in range(F-1)]
			_DispLineGrad(ax, seg, utrack_[p].frm, c_utrack, fmin=fmin, fmax=fmax, linewidth=3)

		for p in range(len(trackmate_)):
			if(p not in num_trackmate[n]): continue
			F = len(trackmate_[p].frm)
			seg = [(trackmate_[p].res[f,:2].tolist(), trackmate_[p].res[f+1,:2].tolist()) for f in range(F-1)]
			_DispLineGrad(ax, seg, trackmate_[p].frm, c_trackmate, fmin=fmin, fmax=fmax, linewidth=3)

		for p in range(len(truth_)):	
			if(p not in num_truth[n]): continue
			F = len(truth_[p].frm)
			seg = [(truth_[p].res[f,:2].tolist(), truth_[p].res[f+1,:2].tolist()) for f in range(F-1)]
			_DispLineGrad(ax, seg, truth_[p].frm, c_truth, fmin=fmin, fmax=fmax, sat_rng=[0,0], val_rng=[1/2,0], linewidth=3)

		plt.plot(xcen[n]+zoom[n]/2-np.array([0,10])-zoom[n]*0.05, ycen[n]+zoom[n]/2-np.array([1,1])*zoom[n]*0.05, linewidth=6, c=[0,0,0])
		plt.xlim(xcen[n]-zoom[n]/2, xcen[n]+zoom[n]/2)
		plt.ylim(ycen[n]-zoom[n]/2, ycen[n]+zoom[n]/2)
		#print(len(knot__), len(trackmate__), len(utrack__))

	plt.show()

def FIG5(code):
	# Datasets are included below: (*_t indicates a troika run) #
	lims = {
			'test': [[0, 128], [0, 128], [0, 16], 1],
			'test_t': [[0, 128], [0, 128], [-1.25, 1.25], 10],
			'bead': [[0, 128], [32, 128], [1, 10], 10],
			'bead_t': [[0, 128], [32, 128], [-1.5, 0.5], 5],
			'protein': [[260, 360], [80, 180], [0, 16], 1],
			'protein_t': [[260, 360], [80, 180], [-2, 2], 1],
			'cell': [[0, 128], [0, 128], [1, 16], 15],
			'cell_t': [[0, 128], [0, 128], [-2, 2], 15],
			}

	clr = plt.get_cmap('tab10').colors
	img = OP._LoadMov(code, OP.FOLD_IMG)
	trk = OP._LoadTruth(code)
	plt.figure(figsize=(8,8))
	ax = plt.gca(position=[-0.15,-0.15,1.25,1.25], projection='3d')
	F = np.shape(img)[0]
	k = 0
	df = 1

	if(code == 'protein' or code == 'protein_t'):
		img_crop = np.max(img, axis=(0,1))
		img_crop = img_crop[lims[code][1][0]:lims[code][1][1], :][:, lims[code][0][0]:lims[code][0][1]]
		VIS._Disp3Dimg(ax, img_crop[None,...][None,...], stride=1)
	else:
		img_crop = img[..., lims[code][1][0]:lims[code][1][1], :][..., lims[code][0][0]:lims[code][0][1]]
		VIS._Disp3Dimg(ax, img_crop, stride=1)

	for p in range(len(trk)):
		# Only get ones that are in the field of view #
		if(np.mean(trk[p].res[:,0]) < lims[code][0][0] or np.mean(trk[p].res[:,0]) > lims[code][0][1]): continue
		if(np.mean(trk[p].res[:,1]) < lims[code][1][0] or np.mean(trk[p].res[:,1]) > lims[code][1][1]): continue
		if(np.mean(trk[p].res[:,2]) < lims[code][2][0] or np.mean(trk[p].res[:,2]) > lims[code][2][1]): continue
		if(len(trk[p].frm) < lims[code][3]): continue

		# Move the trajectory #
		trk_ = trk[p].res[:,:3] * ([1,1,-1] if code=='test_t' else [1,1,1]) - [lims[code][0][0], lims[code][1][0], lims[code][2][0]]
		if(len(trk_) > 1 and np.any(np.ptp(trk_, axis=0) > 0.25)):	# Smooth #
			trk__ = np.zeros_like(trk_)
			for f in range(np.shape(trk_)[0]):
				lft = np.maximum(0, f - df)
				rgt = np.minimum(f + df + 1, np.shape(trk_)[0])
				trk__[f,:] = np.mean(trk_[lft:rgt, :], axis=0)
			trk_ = trk__[...]

			if(code=='test_t'):
				temp = np.array([trk_[:,1], trk_[:,0], trk_[:,2]]).T
				trk_ = temp[...]

			# Make gradient colors if applicable #
			seg = [(trk_[f,:].tolist(), trk_[f+1,:].tolist()) for f in range(len(trk[p].frm)-1)]
			seg_bot = [((trk_[f,:]*[1,1,0.01]).tolist(), (trk_[f+1,:]*[1,1,0.01]).tolist()) for f in range(len(trk[p].frm)-1)]

			VIS._DispLineGrad(ax, seg, trk[p].frm, clr[np.mod(k, len(clr))], fmin=0, fmax=F, linewidth=6)
			VIS._DispLineGrad(ax, seg_bot, trk[p].frm, clr[np.mod(k, len(clr))], bot=True, fmin=0, fmax=F, linewidth=3, linestyle='--')

		else:
			ax.scatter(trk_[:,0], trk_[:,1], trk_[:,2], s=100, c=clr[np.mod(k, len(clr))])
			ax.scatter(trk_[:,0], trk_[:,1], 0*trk_[:,2], s=100/4, c=clr[np.mod(k, len(clr))])		

		#ax.text(np.mean(trk_[:,0]), np.mean(trk_[:,1]), np.mean(trk_[:,2])+1, str(k))

		k += 1;

	ax.set_xlim(0, lims[code][0][1] - lims[code][0][0])
	ax.set_ylim(0, lims[code][1][1] - lims[code][1][0])
	ax.set_zlim(0, lims[code][2][1] - lims[code][2][0])
	ax.set_zticks(np.linspace(0, lims[code][2][1] - lims[code][2][0], 5))
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_zticklabels([])
	ax.view_init(15, -120)	# Test -120, Bead Protein Cell 150, 
	plt.show()

def FIG6x(clouds, traj, img, step):
	# Dataset: FIG 6a
	#	traj = [traj[2], traj[1], traj[0], traj[3]]
	# Parameters:
	#	KER_Z = 16
	#	TRK_LEN = 10
	#	TRK_KDE = 120
	#	TRK_RAD = 1.500
	clr = [(1.0, 0.6, 0.0), (0.0, 0.5, 0.8), (0.8, 0.3, 0.8)]
	clr = clr[2]
	
	## Trajectories ##
	# Get the points #
	pts = np.zeros([USER.SIM_FRAMES, 3])
	cloud = []
	for f in range(USER.SIM_FRAMES):
		cloud.append(PointCloud(np.concatenate([c.pts for c in clouds if f in c.frames], axis=0)))
		pts[f,:] = np.mean(cloud[f].pos, axis=0)

	# Plot all points #
	plt.figure(figsize=(4,4))
	ax = plt.axes(position=[0,0,1,1])
	ax.imshow(img[-1,0,:,:], cmap='gray')

	pts_ = pts[::step,:2]
	seg = seg = [(pts_[f,:].tolist(), pts_[f+1,:].tolist()) for f in range(np.shape(pts_)[0]-1)]
	_DispLineGrad(ax, seg, range(np.shape(pts_)[0]-1), clr, linewidth=10)

	seg_ = [(pts[f,:2].tolist(), pts[f+1,:2].tolist()) for f in range(np.shape(pts)[0]-1)]
	_DispLineGrad(ax, seg_, range(np.shape(pts)[0]-1), [1,1,1], sat_rng=[0,0], val_rng=[1,2/3], linewidth=6)
	#ax.plot(pts[::step,0], pts[::step,1], color=clr, linewidth=6, marker='o', markersize=12)
	#ax.plot(pts[:,0], pts[:,1], color='w', linewidth=6)

	# Plot only SFD points #
	ax.invert_yaxis()

	## MSD ##
	# Evaluate the MSD #
	T = USER.SIM_FRAMES
	tau = int(T/3)
	msd = np.zeros(T)
	for t in range(T):
		slice_left = np.arange(USER.SIM_FRAMES-t-1)
		slice_right = np.arange(t+1, USER.SIM_FRAMES)
		msd[t] = np.mean(np.sum((pts[slice_right,:] - pts[slice_left,:])**2, axis=1))

	# Fit to a power law #
	pwrlaw = lambda t, p: p[0] * (np.arange(t[0], t[1]) ** p[1]) + p[2]
	p0 = [1, 1, 0]
	p = sp.optimize.minimize(lambda p: np.sum((msd[:tau] - pwrlaw([0,tau], p))**2), p0).x
	print(p)
	
	# Visualize #
	plt.figure(figsize=(4,2))
	ax = plt.axes(position=[0,0,1,1])
	ax.plot(np.arange(T)+1, msd, color='k', linewidth=6)
	ax.plot(np.arange(T)+1, pwrlaw([0,T], p), color=clr, linewidth=3, linestyle='--')
	ax.set_xlim(1, T)
	ax.set_ylim(0, 1000)

	## SFD ##
	# Use the trajectory we've obtained #
	csfd = np.mean(traj[0].phi[:,:,0], 1)
	phi_std = []
	for f in range(0, len(cloud)-step, step):
		rho, phi = _SFD(cloud[f], cloud[f+step])
		phi_std.append(np.std(phi[:,:,0]))

	# Fit to a lorentzian #
	lorentz = lambda q: 1 / (1 + ((USER.MESH_PHI - q[0])/q[1])**2)
	lorentz3 = lambda q: q[0] * (lorentz([q[1], q[2]]) + \
		lorentz([q[1] + 2*np.pi, q[2]]) + lorentz([q[1] - 2*np.pi, q[2]]))
	bounds = [(0, np.max(csfd)), (-2*np.pi, 2*np.pi), (np.pi/12, np.pi/4)]

	q0 = [np.max(csfd), USER.MESH_PHI[np.argmax(csfd)], np.mean(phi_std)]
	q = sp.optimize.minimize(lambda q: np.sum((csfd - lorentz3(q))**2), q0, bounds=bounds).x

	# Obtain coefficient of determination #
	cod = 1 - np.sum((csfd - lorentz3(q))**2)/np.sum((csfd - np.mean(csfd))**2)
	print(cod)

	# Obtain Jenson-Shannon Distances #
	m = (csfd + lorentz3(q))/2
	jsdiv = (sp.stats.entropy(csfd, m) + sp.stats.entropy(lorentz3(q), m))/2
	jsd_fit = np.sqrt(jsdiv)
	print(jsd_fit)

	m = (csfd + np.mean(csfd))/2
	jsdiv = (sp.stats.entropy(csfd, m) + sp.stats.entropy(np.full(np.shape(csfd), np.mean(csfd)), m))/2
	jsd_avg = np.sqrt(jsdiv)
	print(jsd_avg)
	
	# Visualize #
	plt.figure(figsize=(4,2))
	plt.gca(position=[0,0,1,1])
	plt.plot(USER.MESH_PHI, csfd, color='k', linewidth=6)
	plt.plot(USER.MESH_PHI, lorentz3(q), color=clr, linewidth=3, linestyle='--')
	plt.xlim(-np.pi, np.pi)
	plt.ylim(0, 1)
	plt.xticks([])
	plt.yticks([])
	plt.show()
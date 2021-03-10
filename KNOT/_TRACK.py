#%% --- IMPORTS --- %%#
### External ###
import numpy 			as np
import numpy.linalg		as npl
import scipy.misc		as spm
import scipy.spatial 	as spt
import scipy.sparse 	as sps
import matplotlib.pyplot as plt
import itertools

import time

### Internal ###
from __ENUM			import FILEFORMAT as FMT
import __OPERATION 	as OP
import __VISUALS 	as VIS

from _SEGMENT		import PointCloud 	as PC
from _INITIALIZE	import _MeshLat

import USER

#%% --- EXPOSED METHODS --- %%#
def RUN(clouds, img, *, code='', update=False, visual=False):
	## Update query ##
	if(not OP._CheckFile('%s\\%s_tracks.json' % (code, code)) or update):
		# Track the point clouds #
		if(USER.KER_T == 1):
			traj = _Track(clouds, img, code=code, vis=visual)
			OP._SaveTracks(traj, code)
		else:
			traj = _SubTrack(clouds, img, code=code, vis=visual)

		# Stitch the trajectories together # 
		#traj = _Stitch(traj, code=code)	# Skipping for today due to time constraints & testing #

		# Save data #
		OP._SaveTracks(traj, code)
	else:
		# Load data #
		pass

	## Visualization query ##
	if(visual and USER.KER_T == 1):
		clr = plt.get_cmap('tab10').colors

		# 2D #
		plt.figure(figsize=(6,6))
		ax = plt.axes(position=[0,0,1,1])
		ax.imshow(img[-1,0,:,:], cmap='gray')
		k = 0
		for c in clouds:
			ax.scatter(c.pos[:,0], c.pos[:,1], s=100*c.wgt, linewidth=0, marker='o')
		for p in traj:
			F = len(p.frm)
			if(F < 2): continue
			seg = [(p.res[f,:2].tolist(), p.res[f+1,:2].tolist()) for f in range(F-1)]
			VIS._DispLineGrad(ax, seg, p.frm, clr[np.mod(k, len(clr))], linewidth=6)
			k += 1
		ax.set_xlim(0, np.shape(img)[3])
		ax.set_ylim(0, np.shape(img)[2])
		
		# 3D #
		plt.figure(figsize=(6, 6))
		ax = plt.axes(projection='3d', position=[-0.15,-0.15,1.3,1.3])
		k = 0
		for p in traj:
			F = len(p.frm)
			if(F < 2): continue
			seg = [(p.res[f,:].tolist(), p.res[f+1,:].tolist()) for f in range(F-1)]
			VIS._DispLineGrad(ax, seg, p.frm, clr[np.mod(k, len(clr))], linewidth=6)
			k += 1
		ax.set_xlim(0, np.shape(img)[3])
		ax.set_ylim(0, np.shape(img)[2])
		ax.set_zlim(0, USER.KER_Z)
		plt.show()
	elif(visual and USER.KER_T > 1):
		clr = plt.get_cmap('tab10').colors

		# 2D #
		plt.figure(figsize=(6,6))
		ax = plt.axes(position=[0,0,1,1])
		ax.imshow(np.sum(img, axis=(0,1)), cmap='gray')
		k = 0
		for p in traj:
			F = len(p.frm)
			if(F < 2): continue
			pos = np.empty([0, 3])
			for c in range(len(p.cloud)):
				pos = np.concatenate([pos, p.cloud[c].sres[:,:-1]], axis=0)
			seg = [(pos[f,:2].tolist(), pos[f+1,:2].tolist()) for f in range(np.shape(pos)[0]-1)]
			VIS._DispLineGrad(ax, seg, p.frm, clr[np.mod(k, len(clr))], linewidth=6)
			k += 1

		# 2D + time #
		plt.figure(figsize=(6,6))
		ax = plt.axes(projection='3d', position=[0,0,1,1])
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_zticklabels([])
		k = 0
		for p in traj:
			F = len(p.frm)
			if(F < 2): continue
			pos = np.empty([0, 4])
			for c in range(len(p.cloud)):
				pos = np.concatenate([pos, p.cloud[c].sres], axis=0)
			seg = [(pos[f,[0,1,3]].tolist(), pos[f+1,[0,1,3]].tolist()) for f in range(np.shape(pos)[0]-1)]
			VIS._DispLineGrad(ax, seg, p.frm, clr[np.mod(k, len(clr))], linewidth=6)
			k += 1
		ax.set_xlim(0, np.shape(img)[3])
		ax.set_ylim(0, np.shape(img)[2])
		ax.set_zlim(0, np.shape(img)[0])

		# 3D + time #
		plt.figure(figsize=(6,6))
		ax = plt.axes(projection='3d', position=[0,0,1,1])
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_zticklabels([])
		k = 0
		for p in traj:
			F = len(p.frm)
			if(F < 2): continue
			pos = np.empty([0, 4])
			for c in range(len(p.cloud)):
				pos = np.concatenate([pos, p.cloud[c].sres], axis=0)
			seg = [(pos[f,[0,1,2]].tolist(), pos[f+1,[0,1,2]].tolist()) for f in range(np.shape(pos)[0]-1)]
			VIS._DispLineGrad(ax, seg, p.frm, clr[np.mod(k, len(clr))], linewidth=6)
			k += 1
		ax.set_xlim(0, np.shape(img)[3])
		ax.set_ylim(0, np.shape(img)[2])
		ax.set_zlim(0, USER.KER_Z)
		plt.show()
	## Output ##
	return traj

#%% --- STATIC METHODS --- %%#
def _SFD(p:PC, q:PC, *, units=False):
	## Initialization ##
	N, D = np.shape(p.pos)
	M, _ = np.shape(q.pos)

	if(units):
		p_pos = np.repeat(np.expand_dims(p.abs, 1), M, axis=1)
		q_pos = np.repeat(np.expand_dims(q.abs, 0), N, axis=0)
	else:
		p_pos = np.repeat(np.expand_dims(p.pos, 1), M, axis=1)
		q_pos = np.repeat(np.expand_dims(q.pos, 0), N, axis=0)

	## Compute Displacement ##
	diff = q_pos - p_pos	# Cartesian #

	# Distance #
	rho = np.sqrt(np.sum(diff**2, axis=2)).flatten()

	# Angle #
	phi = np.zeros((N * M, D-1))
	phi[:,0] = np.arctan2(diff[:,:,1], diff[:,:,0]).flatten()
	for d in range(1,D-1):
		num = diff[:,:,d+1]
		den = np.sum(diff[:,:,:d+2]**2, axis=2)

		val = np.full_like(num, np.pi/2)
		val[den > 0] = np.arccos( num[den > 0]/np.sqrt(den[den > 0]) )
			
		phi[:,d] = val.flatten()

	## Outputs ##
	return rho, phi
def _SFDwgt(p:PC, q:PC):
	## Initialization ##
	N = len(p.wgt)
	M = len(q.wgt)

	p_wgt = np.repeat(np.expand_dims(p.wgt, 1), M, axis=1)
	q_wgt = np.repeat(np.expand_dims(q.wgt, 0), N, axis=0)

	## Compute Connection Weight ##
	wgt = q_wgt + p_wgt

	## Outputs ##
	return wgt.flatten()

def _subSFD(pc:PC, *, units=False):
	# Calculates the SFD *within* a point cloud #
	## Initialization ##
	pts = pc.pts[:,:-1]
	N, D = np.shape(pts)

	if(units): pts *= [*USER.RES, USER.DOF[1]/USER.KER_Z, 1]
	frm = np.unique(pts[:,-1])

	## Split point cloud up into sub-point clouds ##
	spts = [None] * len(frm)
	for f in range(len(frm)):
		spts[f] = pts[pts[:,-1] == frm[f],:]

	## Compute the displacement between each set ##
	rho = np.empty(0)
	phi = np.empty([0, D-2])
	for f in range(len(frm)-1):
		# Make a 3D matrix to make life "easier" for the difference below #
		p = np.repeat(spts[f][:,None,:], np.shape(spts[f+1])[0], axis=1)
		q = np.repeat(spts[f+1][None,:,:], np.shape(spts[f])[0], axis=0)
		diff = (q - p)	# All four dimensions! #

		rho = np.concatenate([rho, np.sqrt(np.sum(diff[:,:,:-1]**2, axis=2)).flatten()])
		phi_ = []
		for d in range(0, D-2):
			if(d == 0): phi_.append(np.arctan2(diff[:,:,1], diff[:,:,0]).flatten())
			else:
				num = diff[:,:,d+1]
				den = np.sum(diff[:,:,:d+2]**2, axis=2)

				val = np.full_like(num, np.pi/2)
				val[den > 0] = np.arccos( num[den > 0]/np.sqrt(den[den > 0]) )
			
				phi_.append(val.flatten())
		phi = np.concatenate([phi, np.array(phi_).T])

	## Outputs ##
	return rho, phi
def _subSFDwgt(pc:PC):
	## Initialization ##
	pts = pc.pts
	wgt = pc.pts[:,-1]
	frm = np.unique(pts[:,-2])

	## Split point cloud up into sub-point clouds ##
	swgt = [None] * len(frm)
	for f in range(len(frm)):
		swgt[f] = wgt[pts[:,-2] == frm[f]]

	## Compute the displacement between each set ##
	omega = np.empty(0)
	for f in range(len(frm)-1):
		# Make a 3D matrix to make life "easier" for the difference below #
		p = np.repeat(swgt[f][:,None], np.shape(swgt[f+1])[0], axis=1)
		q = np.repeat(swgt[f+1][None,:], np.shape(swgt[f])[0], axis=0)

		omega = np.concatenate([omega, (q+p).flatten()])

	## Outputs ##
	return omega

def _LKDE(pts, wgt, domain, *, period=None):
	# Stretch the vectors across the domain #
	pts_ = np.repeat(pts.flatten()[:,np.newaxis], len(domain), axis=1)
	wgt_ = np.repeat(wgt.flatten()[:,np.newaxis], len(domain), axis=1)

	# Evaluate Lorentzian parameters #
	pts_mean = np.sum(wgt * pts)/np.sum(wgt)
	pts_var = np.sum(wgt*(pts - pts_mean)**2) / np.sum(wgt)

	if(period is not None):		# Phi #
		gamma2 = (np.sqrt(pts_var) + np.pi/64)/8 #((pts_var) + np.pi/64)/4
	else:			# Rho 	#
		gamma2 = ((pts_var) + USER.TRK_RAD/64)/8 #((pts_var) + USER.TRK_RAD/64)/4

	#gamma2 = pts_var + 1E-1
	differ = pts_ - domain

	# Evaluate all Lorentzians #
	lorentz = 1/(1 + differ**2/gamma2)
	if(period is not None):
		if(period > 0):
			lorentz += 1/(1 + (differ - period)**2/gamma2)
			lorentz += 1/(1 + (differ + period)**2/gamma2)
			lorentz /= 3

	# Aggregate at each point in the domain #
	val = np.sum(wgt_ * lorentz, axis=0)/np.sum(wgt_, axis=0)
	val /= np.max(val)

	## Outputs ##
	return val
def _RANSAC(t, x, w, ord=3, N=20, M=3):
	## Initialize ##
	# Find unique values in t #
	u = np.unique(t)
	U = len(u)

	# Make sure that we're operating over enough points #
	ord_ = min(ord, U-1)

	# Select points #
	if(U < np.power(N, (ord+1)/U)):
		# We have few enough combinations to do an exhaustive search #
		pts = list(itertools.combinations(u, ord_+1))
	else:
		# There's too many combinations to run through them all, randomly select N of them #
		pts = [np.random.choice(u, ord_+1, replace=False) for _ in range(N)]
	N = len(pts)
		
	# Set the upper limit for residuals #
	eps = 2*np.mean([np.std(x[t == v]) for v in u])		# 2 sigma should be fine #

	# Create arrays to hold the model details #
	models = np.zeros((N, ord+1))	# Model parameters #
	nums = np.zeros(N)				# Max points in acceptance region #
	err = np.full(N, np.inf)		# Error of fit #

	# Create the data matrix #
	X = np.power.outer(t, np.arange(ord_+1))

	## Perform Random Sample Consensus ##
	for n in range(N):
		# Initialize #
		flag = False
		num = ord + 1

		# Loop #
		for m in range(M):
			nums[n] = num	# Set the current number of valid points #

			t_ = np.concatenate([t[t == p] for p in pts[n]])	# Select relevant points #
			x_ = np.concatenate([x[t == p] for p in pts[n]])
			w_ = np.concatenate([w[t == p] for p in pts[n]])

			X_ = np.power.outer(t_, np.arange(ord_ + 1))	# Create selected data matrix #
			W_ = np.diag(w_)

			pinv = X_.T @ W_ @ X_	# Write down the will-be inverted part of the moore-penrose inverse #
			if(np.abs(npl.det(pinv)) < 1E-6):			# Check if the matrix is un-invertible #
				flag = True		# OOPS #
				break
			beta = npl.inv(pinv) @ (X_.T @ W_) @ x_		# Polynomial least squares regression #

			res = np.abs(x - X @ beta)			# Evaluate residuals #
			err[n] = np.sum(err)
			pts[n] = np.unique(t[res < eps])	
			num = np.sum(res < eps)

			if(num <= nums[n]): break	# Make sure that we're making progress #

		# Return the best model found #
		if(not flag):
			models[n,:len(beta)] = beta
			nums[n] = num

	## Find Best Model ##
	best_nums = np.nonzero(nums == np.max(nums))[0]	# May be more than one #
	best_err = np.argmin(err[best_nums])			# Get the one with the least error #

	## Output ##
	return models[best_nums[best_err], :]

#%% --- METHODS --- %%#
def _Track(clust, img, *, code='', step=1, vis=False):
	# Go frame by frame #
	frames = np.unique(np.concatenate([clust[c].frames for c in range(len(clust))]))
	F = len(frames)
	C = len(clust)

	stpwch = time.time()
	timers = np.zeros((F))

	traj = list()
	for f in range(0, F, step):
		# Get all clusters belonging to this frame #
		clust_f = [clust[c] for c in range(C) if frames[f] in clust[c].frm]
		Cf = len(clust_f)
		
		# Go through each trajectory to see if any match #
		for p in range(len(traj)):
			if(not traj[p].active): continue

			# Identify clusters near the head of the trajectory #
			cand_acen = np.array([clust_f[c].m1a for c in range(Cf)])
			sqdist = np.sum((traj[p].head.m1a - cand_acen)**2, axis=1)
			cand_idx = np.nonzero(sqdist < USER.TRK_RAD**2)[0]

			if(len(cand_idx) == 0): continue
			cand = [clust_f[c] for c in cand_idx]

			# Link! #
			traj_, best = traj[p].Link(cand)

			if(traj_ is not None):
				clust_f[cand_idx[best]].linked = True
				traj[p] = traj_
		
		# Deactivate stagnant particles #
		for p in range(len(traj)):
			if(not traj[p].active): continue
			traj[p].active = (frames[f] - traj[p].head.frm[0] < USER.TRK_TOL)
		
		# Detect merge events #
		emptyclust = list()
		for p in range(len(traj)-1):
			if(not traj[p].active): continue
			
			for q in range(p+1, len(traj)):
				if(not traj[q].active): continue
				if(traj[p].head.frm[0] != traj[q].head.frm[0]): continue	# Require same frame #

				# Check locality #
				if(np.sum((traj[p].head.ares - traj[q].head.ares)**2) < (2*USER.APR_RAD)**2):
					# They are basically the same particle at this point, but check if the recent histories align #
					rho_sim = np.allclose(traj[p].rho[:,:USER.TRK_MIN], traj[q].rho[:,:USER.TRK_MIN], atol=1E-3)
					if(not rho_sim): continue

					phi_sim = np.allclose(traj[p].phi[:,:USER.TRK_MIN,:], traj[q].phi[:,:USER.TRK_MIN,:], atol=1E-3)
					if(not phi_sim): continue

					# We have an overlap! Absorb the smaller one (q) #
					intersection = np.intersect1d(traj[p].frm, traj[q].frm)
					valid_frames = [f for f in traj[q].frm if f not in intersection]
					if(len(valid_frames) == 0):	emptyclust.append(q);	continue

					q_clouds = [traj[q].cloud[c] for c in range(traj[q].num) if traj[q].cloud[c].frm[0] in valid_frames]
					traj[q] = Particle(q_clouds)
					traj[q].active = False
		for e in np.sort(np.unique(emptyclust))[::-1]:
			traj.remove(traj[e])
		
		# Particle cleanup #
		bad = [p for p in range(len(traj)) if(not traj[p].active and len(traj[p].cloud) < USER.TRK_MIN)]
		for b in bad[::-1]:
			traj.remove(traj[b])

		# Create a new particle for each unlinked cluster #
		traj.extend([Particle([clust_f[c]]) for c in range(Cf) if not clust_f[c].linked])

		# Progress Display #
		timers[f] = time.time() - stpwch
		if(f > 1):
			prefix = '(%s):\t%8.3f sec' % (code, timers[f])
			suffix = '(Remain: %5.0f sec)' % ((F-(f+1)) * np.mean(np.diff(timers[timers > 0])))
			VIS._ProgressBar(f+1, F, prefix=prefix, suffix=suffix)

	## Output ##
	return traj
def _SubTrack(clust, img, *, code='', vis=False):
	## Initialize ##
	frames = np.unique(np.concatenate([np.floor(c.frames) for c in clust]))
	F = len(frames)
	C = len(clust)

	traj = list()

	## Iterate ##
	for f in range(F):
		# Get all clusters part of this frame #
		clust_f = [c for c in clust if (frames[f] in np.floor(c.frm))]
		Cf = len(clust_f)
		if(Cf == 0): continue	# No candidates somehow #

		# Go through each trajectories to see if any match #
		for p in range(len(traj)):
			if(not traj[p].active): continue				# Don't if it's dead #
			if(frames[f] < traj[p].head.head[-1]): continue	# Don't if it's not here yet #

			# Identify cluster tails near the head of the trajectory #
			cand_scen = np.array([c.sres[0,:] for c in clust_f])
			cand_sacen = np.array([c.sares[0,:] for c in clust_f])
			sqdist = np.sum((traj[p].head.ahead[:-1] - cand_sacen[:,:-1])**2, axis=1)
			voxdist = np.sum((traj[p].head.head[:-1] - cand_scen[:,:-1])**2, axis=1)
			fdist = np.abs(traj[p].head.head[-1] - cand_scen[:,-1])

			# Identify viable linking candidates #
			cand_vox = (voxdist < 3)									# Voxel distance within one sample offset #
			cand_abs = (sqdist < (USER.TRK_RAD*np.sqrt(1/4+fdist))**2)	# Real search radius grows with time	#
			cand_idx = np.nonzero(cand_vox | cand_abs)[0]			# Combine together #

			if(len(cand_idx) == 0): continue	# If no candidates, move on #
			cand = [clust_f[c] for c in cand_idx]

			
			plt.figure(figsize=(6, 6))
			ax = plt.gca(projection='3d', position=[-0.05, -0.05, 1.1, 1.1])
			ax.set_xticklabels([])	# 3x3x2 #
			ax.set_yticklabels([])
			ax.set_zticklabels([])
			for c in clust[:4]:
				ax.plot(c.abs[:,0], c.abs[:,1], c.frm, c='k', linewidth=0, marker='o', alpha=0.4, zorder=-1000)
			for c in traj[p].cloud:
				ax.plot(c.abs[:,0], c.abs[:,1], c.frm, c='m', linewidth=0, marker='o', alpha=0.6, zorder=-100)
				
			ax.plot(traj[p].head.sares[:,0], traj[p].head.sares[:,1], traj[p].head.sres[:,3], linewidth=3, zorder=-1)
			ax.scatter(traj[p].head.sares[-1,0], traj[p].head.sares[-1,1], traj[p].head.sres[-1,3], s=100)
			ax.scatter(cand_sacen[:,0], cand_sacen[:,1], cand_scen[:,3])

			ax.view_init(azim=60, elev=30)
			plt.show()
			
			# Link! #
			traj_, best = traj[p].Sublink(cand, fdist[cand_idx])
			if(traj_ is not None):
				clust_f[cand_idx[best]].linked = True

				plt.figure(figsize=(6, 6))
				ax = plt.gca(projection='3d', position=[-0.05, -0.05, 1.1, 1.1])
				ax.set_xticklabels([])
				ax.set_yticklabels([])
				ax.set_zticklabels([])
				for c in clust[:4]:
					ax.plot(c.abs[:,0], c.abs[:,1], c.frm, c='k', linewidth=0, marker='o', alpha=0.4, zorder=-1000)
				for c in [traj_.cloud[-1]]:
					ax.plot(c.abs[:,0], c.abs[:,1], c.frm, c=[0,0.8,0], linewidth=0, marker='o', alpha=0.2, zorder=-50)
				for c in traj[p].cloud:
					ax.plot(c.abs[:,0], c.abs[:,1], c.frm, c='m', linewidth=0, marker='o', alpha=0.6, zorder=-100)
				
				ax.plot(traj[p].head.sares[:,0], traj[p].head.sares[:,1], traj[p].head.sres[:,3], linewidth=3, zorder=-1)
				ax.scatter(traj[p].head.sares[-1,0], traj[p].head.sares[-1,1], traj[p].head.sres[-1,3], s=100)
				ax.scatter(cand_sacen[:,0], cand_sacen[:,1], cand_scen[:,3])

				ax.view_init(azim=60, elev=30)
				plt.show()

				traj[p] = traj_

		# Deactivate stagnant particles #
		for p in range(len(traj)):
			if(not traj[p].active): continue
			traj[p].active = (frames[f] - traj[p].head.head[-1] < USER.TRK_TOL)

		# Add unlinked clusters as trajectories provided that this is the tail frame #
		traj.extend([Particle([c]) for c in clust_f if not c.linked])

	## Output ##
	return traj

def _Stitch(traj, *, code=''):
	## Initialize ##
	P = len(traj)
	links = np.zeros((P, P))

	## Determine Candidates ##
	for p in range(P-1):
		for q in range(p+1, P):
			# Frame distance #
			dist_f = np.abs(traj[q].tail.frm[0] - traj[p].head.frm[0])
			if(USER.TRK_TOL < dist_f): continue

			# l2 distance #
			dist_l2 = np.sum((traj[p].head.m1a - traj[q].tail.m1a)**2)
			if(dist_f * USER.TRK_RAD**2 < dist_l2): continue

			# If they've passed, consider them #
			links[p,q] = np.sqrt(dist_l2)

	## Evaluate Candidates ##
	pairs = np.array(np.nonzero(links)).T
	Q = len(np.unique(pairs[:,0]))
	stpwch = time.time()
	timers = np.zeros((Q))

	for p in np.unique(pairs[:,0]):
		# Initialize #
		ord = 2								# Order of polynomial to fit #
		eps = list()						# Maximum tolerance #
		err = [list(), list(), list()]		# Error for each dimension #

		cand = pairs[pairs[:,0] == p, 1]	# Find all candidates to this base particle #

		# Evaluate Candidates #
		for c in cand:	# <idx in traj>
			# Get points close to the ends of each trajectory #
			rng_p = traj[p].head.frm[0] - traj[p].json[:,-2] <= USER.TRK_LEN
			rng_c = traj[c].json[:,-2] - traj[p].head.frm[0] <= USER.TRK_LEN + USER.TRK_TOL
			pts = np.concatenate((traj[p].json[rng_p,:], traj[c].json[rng_c,:]), axis=0)

			# Evaluate bounds #
			dom = np.power.outer(pts[:,-2], np.arange(ord+1))
			eps_p = np.max(np.std(traj[p].json[rng_p,:3], axis=0))
			eps_c = np.max(np.std(traj[c].json[rng_c,:3], axis=0))
			eps.append(eps_p + eps_c)

			# Random Sample Consensus #
			beta = np.zeros((ord+1, 3))
			for d in range(3):
				beta[:,d] = _RANSAC(pts[:,-2], pts[:,d], pts[:,-1], ord=ord)
				err[d].append(np.std(dom @ beta[:,d] - pts[:,d]))
		best = np.argmax(eps / np.max(err, axis=0))

		# Merge base with best candidate #
		if(np.max(err, axis=0)[best] < eps[best]):
			# Initialze #
			cloud_base = traj[p].cloud
			cloud_cand = traj[cand[best]].cloud
			cloud_merge = list()

			frm_base = np.array([c.frm[0] for c in cloud_base])
			frm_cand = np.array([c.frm[0] for c in cloud_cand])
			frm_merge = np.union1d(frm_base, frm_cand)

			# Merge the base into the best candidate #
			for f in range(int(min(frm_merge)), int(max(frm_merge)+1)):
				if((f in frm_base) and (f in frm_cand)):		# Compound both clouds #
					c_base = cloud_base[np.nonzero(f == frm_base)[0][0]]
					c_cand = cloud_cand[np.nonzero(f == frm_cand)[0][0]]
					c_total = PC(np.concatenate([c_base.pts, c_cand.pts], axis=0))
					cloud_merge.append(c_total)
				elif((f in frm_base) and (f not in frm_cand)):	# Take from base #
					cloud_merge.append(cloud_base[np.nonzero(f == frm_base)[0][0]])
				elif((f not in frm_base) and (f in frm_cand)):	# Take from cand #
					cloud_merge.append(cloud_cand[np.nonzero(f == frm_cand)[0][0]])
				elif((f in rng_p) or (f in rng_c)):				# Run from the model #
					X = np.power.outer(f, np.arange(ord+1))
					pt = [0, 0, 0, f, np.mean([c.wgt for c in cloud_merge])]
					for d in range(3):
						pt[d] = X @ beta[:,d]
					cloud_merge.append(PC(pt))

			# Discard the base and alter the candidate #
			# Note: by modifying only the past of the best candidate, we allow that candidate to be the base for another candidate particle
			traj[p].removed = True
			traj[cand[best]] = Particle(cloud_merge, hist=[traj[cand[best]].rho, traj[cand[best]].phi])

		# Progress Bar #
		q = np.where(np.unique(pairs[:,0]) == p)[0][0]
		timers[q] = time.time() - stpwch
		if(sum(timers > 0) > 1):
			prefix = '(%s):\t%8.3f sec' % (code, timers[q])
			suffix = '(Remain: %5.0f sec)' % ((Q-(q+1)) * np.mean(np.diff(timers[timers > 0])))
			VIS._ProgressBar(1, 1, q, Q+1, prefix=prefix, suffix=suffix)

	## Output ##
	return [traj[p] for p in range(P) if not traj[p].removed]

#%% --- PARTICLE CLASS DEFINITION --- %%#
class Particle:
	### Constructor ###
	def __init__(self, pcs, *, hist=None):
		## Initialization ##
		# Numerical #
		self.num = len(pcs)
		self.cloud = pcs

		self.pos = np.array([pc.res for pc in pcs])
		self.abs = np.array([pc.ares for pc in pcs])
		self.frm = np.array(np.concatenate([np.unique(pc.frm) for pc in pcs]))
		self.wgt = np.array([np.sum(pc.wgt) for pc in pcs])

		self.json = np.concatenate([pcs[c].json for c in range(len(pcs))], axis=0)
		self.res = np.array([pc.res for pc in pcs])
		self.ares = np.array([pc.ares for pc in pcs])

		# Boolean States #
		self.active = True		# Whether or not this particle is still moving #
		self.removed = False	# Whether or not this particle doesn't exist #

		# History #
		if(hist is None):	# Use nearest neighbors #
			nn_line = np.linspace(1, 0.7, USER.TRK_KDE)	# Normalize to [max = 1] #
			self.rho = np.repeat(nn_line[:,np.newaxis], USER.TRK_LEN, axis=1)
			self.phi = np.ones((USER.TRK_KDE, USER.TRK_LEN, np.shape(pcs[0].pos)[1] - 1))
		else:				# Load the input #
			self.rho = hist[0]
			self.phi = hist[1]

		# Sub-History #
		if(USER.KER_T > 1):	# We have sub-frame #
			self.sub_rho = []
			self.sub_phi = []

		# Keep track of the first and last point cloud appended to this particle #
		frm_min = [np.min(pc.frm) for pc in pcs]
		frm_max = [np.max(pc.frm) for pc in pcs]

		self.tail = pcs[np.argmin(np.abs(frm_min - np.min(self.frm)))]
		self.head = pcs[np.argmin(np.abs(frm_max - np.max(self.frm)))]
	
	### Methods ###
	def Link(self, cand, score_min=0.01):
		## Initialize ##
		C = len(cand)
		D = 3 - 1
		
		rho = [None] * C
		phi = [None] * C
		wgt = [None] * C
		frm = np.zeros((C))
	
		score_rho = np.zeros(C)
		score_phi = np.zeros((C, D))

		## Evaluate Candidates ##
		for c in range(C):
			# Get the single frame displacements #
			rho[c], phi[c] = _SFD(self.head, cand[c], units=True)
			wgt[c] = _SFDwgt(self.head, cand[c])
			frm[c] = abs(self.head.frm[0] - cand[c].frm[0])
		
			# Compare against the current history #
			wgt_ = wgt[c].flatten()
			hist_rho = np.mean(self.rho, axis=1)
			hist_phi = np.mean(self.phi, axis=1)

			# Rho history #
			rho_ = np.repeat(rho[c][:,np.newaxis], USER.TRK_KDE, axis=1) / frm[c]
			rho_min = np.min(abs(rho_ - USER.MESH_RHO), axis=1)
			rho_thr = rho_min < USER.TRK_RAD
			rho_idx = np.argmin(abs(rho_[rho_thr] - USER.MESH_RHO), axis=1)
			if(sum(rho_idx) > 0):
				score_rho[c] = np.sum(wgt_[rho_thr] * hist_rho[rho_idx])/np.sum(wgt_[rho_thr])

			# phi history #
			phi_ = np.repeat(phi[c][:,0][:,np.newaxis], USER.TRK_KDE, axis=1)
			phi_idx = np.argmin(abs(phi_ - USER.MESH_PHI), axis=1)
			if(sum(phi_idx) > 0):	# ??? #
				score_phi[c,0] = np.sum(wgt_ * hist_phi[phi_idx,0])/np.sum(wgt_)

			# All other phi histories #
			for d in range(1, D):
				phi_ = np.repeat(phi[c][:,d][:,np.newaxis], USER.TRK_KDE, axis=1)
				phi_idx = np.argmin(abs(phi_ - USER.MESH_PHI), axis=1)
				if(sum(phi_idx) > 0):
					score_phi[c,d] = np.sum(wgt_ * hist_phi[phi_idx,d])/np.sum(wgt_)
		
		# Give the final score to each candidate #
		score = score_rho + np.sum(score_phi, axis=1)
		#score_ang = np.prod(score_phi, axis=1)
		#score = (score_rho + score_ang)/2 + np.sqrt(score_rho * score_ang)
		
		if(any(score > score_min * frm**2)):
			# Get the best candidate #
			best = np.argmax(score)

			# Evaluate its characteristics #
			best_rho = _LKDE(rho[best] / frm[best], wgt[best], USER.MESH_RHO)

			best_phi = np.zeros((USER.TRK_KDE, D))
			best_phi[:,0] = _LKDE(phi[best][:,0], wgt[best], USER.MESH_PHI, period=2*np.pi)
			for d in range(1, D):
				best_phi[:,d] = _LKDE(phi[best][:,d], wgt[best], USER.MESH_PHI, period=0)

			# Adjust for small rho #
			#rho_adj = np.exp(-2*np.sum(wgt[best]*rho[best])/np.sum(wgt[best]*frm[best]))
			#best_phi = np.sqrt(best_phi**2 + rho_adj**2)	# renormalize #
			#best_phi /= np.max(best_phi)

			# Modify the history #
			hist_rho_ = np.concatenate((best_rho[:,np.newaxis], self.rho[:,:-1]), axis=1)
			hist_phi_ = np.concatenate((np.expand_dims(best_phi, 1), self.phi[:,:-1,:]), axis=1)

		## Outputs ##
		if(any(score > score_min * frm**2)):
			return Particle([*self.cloud, cand[best]], hist=[hist_rho_, hist_phi_]), best
		else:
			return None, -1
	def Sublink(self, cand, fdist, err_min=0.1):
		# The principle here is that the candidates are close enough that if you find another candidate that shares a similar self-SFD than you, it's probably your continuation. This is primarily to remove links to the side spirals. #
		## Initialize ##
		C = len(cand)
		D = 3 - 1
	
		score_rho = np.zeros(C)
		score_phi = np.zeros((C, D))

		## Evaluate Current Head subSFDs ##
		rho_h, phi_h = _subSFD(self.head, units=True)
		wgt_h = _subSFDwgt(self.head)
		if(len(wgt_h) == 0): return None, -1
		
		# Draw Lorentzian KDEs #
		rho_hL = _LKDE(rho_h, wgt_h, USER.MESH_RHO)
		phi_hL = _LKDE(phi_h[:,0], wgt_h, USER.MESH_PHI, period=2*np.pi)
		theta_hL = _LKDE(phi_h[:,1], wgt_h, USER.MESH_PHI, period=0)

		"""
		plt.figure(1)
		plt.plot(USER.MESH_RHO, rho_hL, c='k')

		plt.figure(2)
		plt.plot(USER.MESH_PHI, phi_hL, c='k')

		plt.figure(3)
		plt.plot(USER.MESH_PHI, theta_hL, c='k')
		"""

		## Evaluate Candidates' subSFDs ##
		rho_cL = np.zeros([USER.TRK_KDE, C])
		phi_cL = np.zeros([USER.TRK_KDE, C])
		theta_cL = np.zeros([USER.TRK_KDE, C])
		for c in range(C):
			rho_c, phi_c = _subSFD(cand[c], units=True)
			wgt_c = _subSFDwgt(cand[c])
			if(len(wgt_c) == 0): continue	# This is a bad #

			rho_cL[:,c] = _LKDE(rho_c, wgt_c, USER.MESH_RHO)
			phi_cL[:,c] = _LKDE(phi_c[:,0], wgt_c, USER.MESH_PHI, period=2*np.pi)
			theta_cL[:,c] = _LKDE(phi_c[:,1], wgt_c, USER.MESH_PHI, period=0)

			# Draw Lorentzian KDEs #
			"""
			plt.figure(1)
			plt.plot(USER.MESH_RHO, rho_cL[c])

			plt.figure(2)
			plt.plot(USER.MESH_PHI, phi_cL[c])

			plt.figure(3)
			plt.plot(USER.MESH_PHI, theta_cL[c])
			"""

		## Evaluate each candidate ##
		err = np.zeros([C, 3])
		# Get errors #
		for c in range(C):

			err[c,0] = np.mean((rho_hL - rho_cL[:,c])**2)
			err[c,1] = np.mean((phi_hL - phi_cL[:,c])**2)
			err[c,2] = np.mean((theta_hL - theta_cL[:,c])**2)

		## Determine which candidate has the least error and link! ##
		best = np.argmin(np.sum(err, axis=1) * fdist)
		err_best = np.min(np.sum(err, axis=1) * fdist)
		print(np.sum(err, axis=1) * fdist)

		## Outputs ##
		if(err_best < err_min):
			return Particle([*self.cloud, cand[best]]), best
		else:
			return None, -1

		## Visualize ##
		"""
		plt.figure()
		ax = plt.gca(projection='3d')
		# Current head #
		ax.scatter(self.head.pts[:,0], self.head.pts[:,1], self.head.pts[:,3], s=100*self.head.wgt, c='k')
		ax.plot(self.head.sares[:,0], self.head.sares[:,1], self.head.sres[:,3], c='k')

		# Candidates #
		for c in range(C):
			ax.scatter(cand[c].pts[:,0], cand[c].pts[:,1], cand[c].pts[:,3], s=100*cand[c].wgt)
			ax.plot(cand[c].sares[:,0], cand[c].sares[:,1], cand[c].sres[:,3])
		plt.show()
		"""

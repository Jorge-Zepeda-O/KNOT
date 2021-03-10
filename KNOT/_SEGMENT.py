#%% --- IMPORTS --- %%#
### External ###
from sklearn.cluster	import KMeans	
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011. #

import numpy 				as np
import numpy.fft			as npf
import scipy.spatial 		as spt
import scipy.io				as spi
import matplotlib.pyplot	as plt
from mpl_toolkits.mplot3d	import Axes3D
from matplotlib.collections import PolyCollection

import time

### Internal ###
from __ENUM			import FILEFORMAT as FMT
import __OPERATION 	as OP
import __VISUALS 	as VIS

from _INITIALIZE	import _MeshLat, _MeshMeta

import USER

#%% --- EXPOSED METHODS -- %%#
def RUN(pos, wgt, img, *, code='', update=False, visual=False):
	## Update query ##
	if(not OP._CheckFile('%s\\%s_clouds.json' % (code, code)) or update):
		# Perform the segmentation #
		clouds = _Identify(pos, wgt, img, code=code, vis=visual)

		# Save data #
		clouds_json = [clouds[c].json for c in range(len(clouds))]
		OP._SaveJSON(clouds_json, '%s\\%s_clouds' % (code, code))
	else:
		# Load data #
		clouds_json = OP._LoadJSON('%s\\%s_clouds' % (code, code))
		clouds = []
		for c in range(len(clouds_json)):
			cloud = np.array(clouds_json[c])
			clouds.append(PointCloud(cloud))

	## Visualization query ##
	if(visual):
		plt.figure(figsize=(6,6))
		ax = plt.gca(projection='3d', position=[-0.05, -0.05, 1.1, 1.1])
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_zticklabels([])
		for c in clouds[:2]:
			ax.scatter(c.abs[:,0], c.abs[:,1], c.frm, s=400*c.wgt, c='k')	# 2x2
		ax.view_init(azim=60, elev=15)
		plt.show()

	## Output ##
	return clouds

#%% --- STATIC METHODS --- %%#
def _BFS(graph, visit, n):
	## Initialization ##
	queue = [n]
	visit[n] = True

	## Graph Search ##
	while(queue):
		n = queue.pop(0)
		if(not n in graph): continue
		queue.extend([i for i in graph[n] if not visit[i]])
		visit[graph[n]] = True

	## Outputs ##
	return visit
def _DelaunayCluster(pts, *, voxthr=USER.SEG_RAD, vis=False):
	## Initialize ##
	P, D = np.shape(pts)
	pos = pts[:,:-1].copy()
	pos[:,-1] *= np.sqrt(USER.KER_T)	# Inflate the time axis a bit - it's sub-frame after all #

	## Triangulate ##
	try:
		# Triangulate based on the absolute positions #
		planar = np.array([np.ptp(pos[:,d]) == 0 for d in range(D-1)], bool)	# Coplanar? #
		tri = spt.Delaunay(pos[:,~planar])	# Triangulate #
	except spt.qhull.QhullError:
		tri = None	# We'll brute force it #
	except ValueError:
		tri = None	# We'll brute force it #

	## Compute Segmentation ##
	# Initialize #
	graph = dict(list())
	visit = np.zeros(np.shape(pts)[0], dtype=bool)
	seg = list()

	if(tri is not None):
		# Check the simplicies for valid links #
		S, V = np.shape(tri.simplices)
		for s in tri.simplices:
			for u in range(V-1):
				for v in range(u+1, V):	
					voxdist = np.sum((pos[s[u],~planar] - pos[s[v],~planar])**2)
					if(voxdist <= voxthr**2):	# A link has been established #
						if(s[u] not in graph): graph[s[u]] = list()
						if(s[v] not in graph): graph[s[v]] = list()
						graph[s[u]].append(s[v])
						graph[s[v]].append(s[u])
	else:
		# Try all possible points for valid links #
		for u in range(P-1):
			for v in range(u+1, P):
				voxdist = np.sum((pos[u,:] - pos[v,:])**2)
				if(voxdist <= voxthr**2):	# A link has been established #
					if(u not in graph): graph[u] = list()
					if(v not in graph): graph[v] = list()
					graph[u].append(v)
					graph[v].append(u)

	## Traverse Graph ##
	while(not all(visit)):
		item = np.nonzero(~visit)[0][0]	# Get the first free point 					#
		if(item not in graph):			# If it has no edges, it's a free stander 	#
			visit[item] = True
			seg.append([item])
		else:							# Else explore the subgraph and add it		#
			visit_0 = visit.copy()
			visit = _BFS(graph, visit, item)
			dvisit = np.logical_xor(visit, visit_0)
			seg.append(np.nonzero(dvisit)[0].tolist())

	## Sanity Check ##
	# Remove "empty" points and clusters #
	for c in range(len(seg)):	
		seg[c] = np.array(seg[c])[np.nonzero(pts[seg[c],-1])[0]].tolist()
	seg = [seg[c] for c in range(len(seg)) if len(seg[c]) > 0]

	## Split Up Into Clusters ##
	clust = [PointCloud(np.array([pts[s[q],:] for q in range(len(s))])) for s in seg]

	## Conflate nearby clusters with known problems ##
	# Check which direction to be looking in #

	## Visualization ##
	if(False):
		plt.figure()
		ax = plt.axes(projection='3d')
		for c in clust:
			ax.scatter(c.pos[:,0], c.pos[:,1], c.frm)
		plt.show()

	## Output ##
	return clust

def _CloudThr(clust, tru=[], *, vis=False):
	## Initialization ##
	C = len(clust)

	# Moments #
	m0 = np.array([c.m0 for c in clust])
	m1 = np.array([c.m1 for c in clust])
	m2 = np.array([c.m2 for c in clust])
	m3 = np.array([c.m3 for c in clust])
	
	# Other metrics #
	m0x = np.array([np.max(c.wgt) for c in clust])
	area = np.sqrt(np.array([c.num for c in clust]))
	size = np.maximum(np.array([np.sqrt(np.sum(np.ptp(c.pos, axis=0)**2)) for c in clust]), 1)

	## Calculate Moment Radii ##
	m2r = np.sqrt(np.sum(m2, axis=1))
	m3r = np.cbrt(np.sum(np.abs(m3), axis=1))

	## Classify ##
	# 0.99 **2, 0.33 **2
	classify = lambda x, y: np.maximum(np.minimum(2*((x/1.04)**2 + (y/0.47) - 1), 1), 1E-3)**2
	good = (classify(m0x/m0, m2r/area) < m3r/size)# | (classify(m2r/area, m0x/m0) < m3r/size)
	
	# Redemption #
	if(sum(~good) > 0):
		m0_thr = np.mean(np.log10(m0[~good])) + 2*np.std(np.log10(m0[~good]))
		okay = np.zeros(len(good), dtype=bool)
		#okay = (m0_thr < np.log10(m0)) & (~good)
	else:
		m0_thr = 0
		okay = np.zeros(len(good), dtype=bool)

	# Verdict #
	safe = good | okay

	## Visualization ##
	if(vis):
		if(np.size(tru) > 0):
			dist = np.zeros((np.shape(tru)[0], len(clust)))
			for i in range(np.shape(tru)[0]):
				for c in range(len(clust)):
					dist[i,c] = np.sum((tru[i,:] - clust[c].m1)**2)
			close = np.min(dist, axis=0) < 5**2
		else:
			close = np.zeros(len(clust), dtype=bool)

		# On image #
		for c in range(C):
			if(close[c]):
				plt.scatter(clust[c].pos[:,0], clust[c].pos[:,1], \
					s=25*(np.log10(clust[c].wgt) + 3), c='b', alpha=0.5/np.sqrt(clust[c].num))
			elif(good[c]):
				plt.scatter(clust[c].pos[:,0], clust[c].pos[:,1], \
					s=25*(np.log10(clust[c].wgt) + 3), c='g', alpha=0.5/np.sqrt(clust[c].num))
			elif(okay[c]):
				plt.scatter(clust[c].pos[:,0], clust[c].pos[:,1], \
					s=25*(np.log10(clust[c].wgt) + 3), c='y', alpha=0.5/np.sqrt(clust[c].num))
			else:
				plt.scatter(clust[c].pos[:,0], clust[c].pos[:,1], \
					s=25*(np.log10(clust[c].wgt) + 3), c='r', alpha=0.5/np.sqrt(clust[c].num))
		plt.title('%4.1f%%' % (100 * sum(good & close)/len(close)))
		plt.xlim(0, USER.CHIP[0])
		plt.ylim(0, USER.CHIP[1])

		# Classification #
		ax = plt.axes(projection='3d', position=[0.5,0.33,0.5,0.67])
		ax.scatter(m0x[good & ~close]/m0[good & ~close],m2r[good & ~close]/area[good & ~close], \
			(m3r[good & ~close]/size[good & ~close]), c='g')
		ax.scatter(m0x[~good & okay]/m0[~good & okay],	m2r[~good & okay]/area[~good & okay], \
			(m3r[~good & okay]/size[~good & okay]), c='y')
		ax.scatter(m0x[~safe]/m0[~safe],				m2r[~safe]/area[~safe], \
			(m3r[~safe]/size[~safe]), c='r')

		ax.scatter(m0x[good & close]/m0[good & close],	m2r[good & close]/area[good & close], \
			(m3r[good & close]/size[good & close]), c='b')
		ax.scatter(m0x[~good & close]/m0[~good & close],m2r[~good & close]/area[~good & close], \
			(m3r[~good & close]/size[~good & close]), c='m')

		xlim = 1
		ylim = 1
		zlim = 1
		xx, yy = np.meshgrid(np.linspace(0, xlim, 41), np.linspace(0, ylim, 41))
		zs = classify(xx, yy)
		ax.plot_surface(xx, yy, zs, alpha=0.2)
		
		ax.set_xlabel('m0x/m0')
		ax.set_ylabel('m2r/area')
		ax.set_zlabel('m3r/size')
		ax.set_xlim(0, xlim)
		ax.set_ylim(0, ylim)
		ax.set_zlim(0, zlim)

		ax = plt.axes(position=[0.5,0.03,0.5,0.30])
		ax.hist(np.log10(m0[~good]), bins=np.linspace(-2, 2, 41))
		ax.plot(m0_thr*np.ones(2), [0, 10], color='k')
		ax.set_xlabel('m0')
		#plt.show()

	## Output ##
	return [clust[c] for c in range(C) if safe[c]]
def _Separate(clust, *, vis=False):
	## Initialize ##
	C = len(clust)
	clen = np.array([clust[c].num for c in range(C)])
	cwgt = np.array([clust[c].m0 for c in range(C)])
	sig = 3

	## Find "Big" Clusters #
	# Filter out things that are obviously too big or small #
	m1 = np.sum(cwgt * clen)/np.sum(cwgt)
	m2 = np.sqrt(np.sum(cwgt * (clen - m1)**2)/np.sum(cwgt))
	sel = (m1 - sig*m2 <= clen) & (clen < m1 + sig*m2)

	# Refine the judgments #
	if(m2 == 0):
		smol = []
		bigg = []
	else:
		m1_ = np.sum(cwgt[sel] * clen[sel])/np.sum(cwgt[sel])
		m2_ = np.sqrt(np.sum(cwgt[sel] * (clen[sel] - m1_)**2)/np.sum(cwgt[sel]))

		smol = np.nonzero(clen <= m1_ - sig*m2_)[0]
		bigg = np.nonzero((clen > m1_ + sig*m2_) & (clen/2 > m1_ - sig*m2_))[0]

	if(False):
		for c in range(C):
			if(c in smol):
				plt.scatter(clust[c].pos[:,0], clust[c].pos[:,1], c='r', alpha=0.2)
			elif(c in bigg):
				plt.scatter(clust[c].pos[:,0], clust[c].pos[:,1], c='y', alpha=0.2)
			else:
				plt.scatter(clust[c].pos[:,0], clust[c].pos[:,1], c='g', alpha=0.2)
			plt.text(clust[c].m1[0], clust[c].m1[1], c, color='w')
		plt.xlim(0, USER.CHIP[0])
		plt.ylim(0, USER.CHIP[1])
	
	## Split Clusters ##
	clust_ = list()
	for c in range(C):
		if(c in bigg):
			# Split and recluster #
			k = int(clen[c] // m1_ + 1)
			kmeans = KMeans(n_clusters=k).fit(clust[c].pos)
			idx = kmeans.labels_

			# If any cluster is too small, reduce the number of clusters and try again #
			while(any([sum(idx == i) <= m1_ - sig*m2_ for i in range(k)]) and k > 0):
				k -= 1
				kmeans = KMeans(n_clusters=k).fit(clust[c].pos)
				idx = kmeans.labels_

			# Append each cluster to the big list #
			for i in range(k):
				if(sum(idx == i) > m1_ - sig*m2_):
					PC = PointCloud(clust[c].pts[idx == i,:])
					clust_.append(PC)
					if(vis): plt.scatter(clust[c].pos[idx == i,0], clust[c].pos[idx == i,1])
		elif(c not in smol):
			clust_.append(clust[c])

	for c in range(len(clust_)):
		if(clust_[c].removed): continue
		for q in range(len(clust_)):
				
			if(c <= q): continue
			if(clust_[q].removed): continue

			# Evaluate the distance between c and q #
			dist = np.sqrt(np.sum((clust_[c].ares - clust_[q].ares)**2))
			#print(c, q, dist)
			if(dist > USER.SEG_SEP): continue

			# Remove the cluster that is less confident #
			if(np.max(clust_[c].wgt) > 1*np.max(clust_[q].wgt)):
				clust_[q].removed = True
				#print('q removed')
			elif(np.max(clust_[q].wgt) > 1*np.max(clust_[c].wgt)):
				clust_[c].removed = True
				#print('c removed')
			else:
				pass
				#print('')
	clust = [clust_[c] for c in range(len(clust_)) if(not clust_[c].removed)]

	if(vis):
		for c in range(len(clust)):
			plt.scatter(clust[c].pos[:,0], clust[c].pos[:,1], c='r', alpha=0.5)
			plt.text(clust[c].m1[0], clust[c].m1[1], c, color='w')
		plt.xlim(0, USER.CHIP[0])
		plt.ylim(0, USER.CHIP[1])

	## Visualization ##
	if(vis):
		ax = plt.axes(position=[0.5,0,0.5,1])
		ax.hist(clen, weights=cwgt, bins=range(40))
		ax.plot((m1) * np.ones(2), [0, 80], color='k', linestyle='-')
		ax.plot((m1 - sig*m2) * np.ones(2), [0, 80], color='k', linestyle='--')
		ax.plot((m1 + sig*m2) * np.ones(2), [0, 80], color='k', linestyle='--')
		if(m2 > 0):
			ax.plot((m1_) * np.ones(2), [0, 40], color='r', linestyle='-')
			ax.plot((m1_ - sig*m2_) * np.ones(2), [0, 40], color='r', linestyle='--')
			ax.plot((m1_ + sig*m2_) * np.ones(2), [0, 40], color='r', linestyle='--')

	## Output ##
	return clust

#%% --- METHODS --- %%#
def _Identify(pos, wgt, img, *, code, vis=False):
	## Initialization ##
	F = len(wgt)
	clouds = list()

	# Progress #
	stpwch = time.time()
	timers = np.zeros((F))
	tru = OP._LoadTruth(code)			

	## Cluster points in each frame ##
	for f in range(F):
		if(len(wgt[f]) == 0): continue	# Check if there are any points in this frame #

		# Create a point cloud based on the points #
		pts = np.concatenate([pos[f], wgt[f][:,None]], axis=1)

		# Segment the point cloud to find clusters #
		cloud = PointCloud(pts, seg=True)

		# Why?? # vvv #
		# Weight threshold #
		clust = _CloudThr(cloud.clust, vis=vis)
		#		# ^^^ #
		clust = _Separate(clust)

		# Append new clusters to the batch #
		clouds.extend(clust)

		## Visualize ##
		"""
		if(vis):
			pts = np.zeros([0, 3]);
			for p in range(len(tru)):
				if(f in tru[p].frm):
					idx = np.nonzero(f == tru[p].frm)[0]
					pts = np.concatenate((pts, tru[p].res[idx,:]), axis=0)

			plt.figure()
			if(USER.KER_Z > 1):
				ax = plt.axes(position=[0,0,1,1], projection='3d')
				imclr = np.repeat((img/np.max(img))[f,0,:,:,None], 3, axis=2)
				xx, yy = np.meshgrid(range(np.shape(img)[2]), range(np.shape(img)[3]))
				ax.plot_surface(xx*USER.RES[0], yy*USER.RES[1], 0*xx, facecolors=imclr, rstride=8, cstride=8, zorder=-100000)
				for c in clust:
					ax.plot(c.abs[:,0], c.abs[:,1], c.abs[:,2], marker='o', linewidth=0, zorder=1000)
			else:
				ax = plt.axes(position=[0,0,1,1])
				ax.imshow(img[f,0,:,:], cmap='gray')
				for c in clust:
					ax.scatter(c.pos[:,0], c.pos[:,1], s=400*c.wgt, c='b')
				ax.plot(pts[:,0], pts[:,1], color='r', marker='o', linewidth=0, fillstyle='none')
			plt.show()
		"""

		# Progress Display #
		timers[f] = time.time() - stpwch
		if(f > 0):
			prefix = '(%s):\t%8.3f sec' % (code, timers[f])
			suffix = '(Remain: %5.0f sec)' % ((F-(f+1)) * np.mean(np.diff(timers[timers > 0])))
			VIS._ProgressBar(f+1, F, prefix=prefix, suffix=suffix)

	## Output ##
	return clouds

#%% --- POINTCLOUD CLASS DEFINITION --- %%#
class PointCloud:
	### Constructor ###
	def __init__(self, pts, *, seg=False):
		## Initialization ##
		self.pts = pts
		self.num, self.dim = np.shape(pts)
		self.json = pts.tolist()

		# Positions #
		self.pos = pts[:,:-2]													# Point positions		#
		self.abs = pts[:,:-2] * np.array([*USER.RES, USER.DOF[0]/USER.KER_Z])	# Absolute positions	#

		self.frm = pts[:,-2]
		self.wgt = pts[:,-1]

		# Tracking Flags #
		self.linked = False
		self.removed = False
		self.frames = np.unique(self.frm)

		# Weighted Moments #
		self.m0  = np.sum(self.wgt)

		self.m1  = np.sum(self.wgt * self.pos.T, 				axis=1).T / self.m0
		self.m1a = np.sum(self.wgt * self.abs.T, 				axis=1).T / self.m0

		self.m2  = np.sum(self.wgt *(self.pos - self.m1 ).T**2, axis=1).T / self.m0
		self.m2a = np.sum(self.wgt *(self.abs - self.m1a).T**2, axis=1).T / self.m0

		self.m3  = np.sum(self.wgt *(self.pos - self.m1 ).T**3, axis=1).T / self.m0
		self.m3a = np.sum(self.wgt *(self.abs - self.m1a).T**3, axis=1).T / self.m0

		# Localization #
		self.res = self.m1
		self.ares = self.m1a

		# Sub-frame localization #
		if(USER.KER_T > 1):
			units = np.array([*USER.RES, USER.DOF[1]/USER.KER_Z, USER.FRATE/USER.KER_T])
			self.sres = np.empty([0, 4])
			self.sares = np.empty([0, 4])
			for sf in np.unique(pts[:,-2]):
				spts = pts[pts[:,-2] == sf,:]
				sm1 = np.sum(spts[:,-1] * spts[:,:-1].T, axis=1).T / np.sum(spts[:,-1])
				sm1a = sm1 * units
				self.sres = np.concatenate([self.sres, sm1[None,:]], axis=0)
				self.sares = np.concatenate([self.sares, sm1a[None,:]], axis=0)

			self.tail = self.sres[np.argmin(self.sres[:,-1]),:]
			self.head = self.sres[np.argmax(self.sres[:,-1]),:]
			self.atail = self.tail * units
			self.ahead = self.head * units

		## Triangulation ##
		# Perform the triangulation in voxel space #
		self.clust = _DelaunayCluster(self.pts) if(seg) else None

	### Class Methods ###
	@classmethod
	def _Merge(cls, *pcs, tri=False):
		# Merge multiple clouds together #
		return cls(np.concatenate([pc.pts for pc in pcs], axis=0), tri=tri)
	@classmethod
	def _Absorb(cls, pc, pts, *, tri=False):
		# Add points to an existing cloud #
		return cls(np.concatenate([pc.pts, pts], axis=0), tri=tri)
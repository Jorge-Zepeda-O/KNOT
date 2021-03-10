import numpy as np
import scipy.io as scipio
import matplotlib.pyplot as plt
import pandas as pd
import json
import h5py
import re
import sys
import unicodedata
from collections import defaultdict
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import math
import time
import os
import timeit
from timeit import default_timer as timer

class particle:
		
	def __init__(self, xs = None, ys = None, from_json = False, x_pos = None, y_pos = None, start = None, num_frames = None, msd = None, jump_dist = None, hops = None, area_covered = None, polar_length = 0, polar_angle = 0, num_blinks = 0, wait_times = None):
		
		self.x_pos = list()
		self.y_pos = list()
		self.raw_x_pos = list()
		self.raw_y_pos = list()
		self.start = 0
		self.num_frames = 0
		self.msd = 0
		self.hops = 0
		self.jump_dist = None
		self.area_covered = None
		self.num_blinks = 0
		self.polar_angle = 0
		self.polar_length = 0
		self.wait_times = list()

		# Reconstitute a particle from a stored json_file
		if from_json is True:
			self.x_pos = x_pos
			self.y_pos = y_pos
			self.start = start
			self.num_frames = num_frames
			self.msd = msd
			self.hops = hops
			self.jump_dist = jump_dist
			self.area_covered = area_covered
			self.wait_times = wait_times
		# Load a new trajectory from a data file
		else:
			frames = np.nonzero(xs)
			self.num_frames = len(frames[0])
			self.start = frames[0][0]
			#print frames
			self.x_pos = np.trim_zeros(xs).tolist()
			self.y_pos = np.trim_zeros(ys).tolist()	
	
	def calcMSD(self):
		# TODO Check if MSD is supposed to be an incremental value calculated at each step.
		start_x = self.x_pos[0]
		start_y = self.y_pos[0]
		displacement = 0
		for i in range(self.num_frames):
			dist = (start_x-self.x_pos[i])**2 + (start_y-self.y_pos[i])**2
			displacement = displacement + dist
		self.msd = displacement/self.num_frames
		return self.msd
	
	def rollCall(self):
		print 'I am a particle starting at ' + str(self.start) + " and go for " + str(self.num_frames) + ' frames. I jump ' + str(len(self.hops)) + ' times. '
	
	def getStart(self):
		return self.start
	
	def getNumFrames(self):
		return self.num_frames

	def getNumHops(self, jump_dist = 1):
		self.calcJumpDists(jump_dist = jump_dist)
		if self.hops is None:
			return 0
		return len(self.hops)

	def getJumpDistances(self):
		if self.hops is None:
			return []
		return self.hops
	
	def plotParticle(self, fig_handle = None, color = 'b'):
		if fig_handle is None:
			fig_handle = plt.figure()
		plt.plot(self.y_pos, self.x_pos, color)

	def calcJumpDists(self, jump_dist = 1):
		# Catch if there is only one frame, so no hops
		self.hops = []
		self.jump_dist = jump_dist
		if self.num_frames is 1:
			self.hops = None
		# If there is more than one frame, calculate distances
		else:
			self.wait_times = list()
			self.wait_times.append(1)
			wait_index = 0
			for i in range(len(self.x_pos)-1):
				x1 = self.x_pos[i]
				y1 = self.y_pos[i]
				x2 = self.x_pos[i+1]
				y2 = self.y_pos[i+1]
				# Detect blinking in the trajectory and adjust accordingly
				if x2 == 0.0 and y2 == 0.0:
					#print 'blink'
					# If a blink is detected, take the coordinates of the previous point; Increment the number of blinks and adjust the original trajectory. This is a destructive process.
					self.num_blinks = self.num_blinks + 1
					self.x_pos[i+1] = self.x_pos[i]
					self.y_pos[i+1] = self.y_pos[i]
					x2 = self.x_pos[i]
					y2 = self.y_pos[i]

				dist = np.linalg.norm([x1-x2, y1-y2])
				if dist >= jump_dist:
					self.wait_times.append(1)
					wait_index += 1
					self.hops.append(dist)
				else:
					self.wait_times[wait_index] += 1
			if self.x_pos[-1] == 0 and self.y_pos[-1] == 0:
				self.x_pos[-1] = self.x_pos[-2]
				self.y_pos[-1] = self.y_pos[-2]
	def calcPolarCoords(self):
		if self.hops is None:
			hops = 0
		else:
			hops = len(self.hops)
		# The old measure which calculated tradition polar. We want to adjust it for frame numbers
		# self.polar_length = np.sqrt(hops**2 + self.num_frames**2)
		self.polar_length = self.num_frames
		self.polar_angle = np.arctan(float(hops)/self.num_frames)

	def calcAreaCovered(self):
		return 0
	
	def isSingleFrame(self):
		return self.num_frames == 1

	def isImmobile(self):
		if self.hops is None:
			return True
		else:
			return len(self.hops) == 0

	def toJSON_old(self, aggregate = True, directory = 'example.json'):

		json_data = self.__dict__
		if aggregate:
			return json_data
		else:
			with open(directory, 'w') as fp:
				print json_data
				json.dump(json_data, fp, sort_keys = True, indent = 4)
				fp.close()
			return 'Stored'
	
	def toJSON(self, directory = './example.json'):
		with open(directory, 'w') as fp:
			json.dump(self.__dict__, fp)
			fp.close

"""
The wideFieldData class leverages the data stored in a hdf5 data file and organizes it in terms of particles.
"""
class WideFieldData:
	
	def __init__(self, file_handle = '', verbose = False, cutoff = None, name = 'sample', from_json = False, particle_json = '', frame_json = '', bins = 18):
		
		# This is the first time the WideFieldData has been processed
		self.particles = dict()
		self.frames = defaultdict(list)
		self.name = name
		self.cdf = None
		self.bins = bins
		self.pdf = None
		self.verbose = verbose
		self.num_filters = 0
		self.filt_type = list()
		self.filt_params = list()
		self.single_incident_pop = list()
		self.immobile_pop = list()

		if not from_json:
			data = h5py.File(file_handle)
			
			# Old version that worked with Nick's data
			# trjR = data['trjR']
			
			# Chayan saved the value under a different name so we have to check the column
			trjR = data[data.keys()[0]]

			self.name = name
			
			if cutoff is not None:
				stop_point = cutoff
			else:
				stop_point = trjR.shape[0]
			
			if verbose:
				print trjR.shape
				print np.nonzero(trjR)[0]

			for i in range(stop_point):
				try:
					self.particles[i] = particle(trjR[i,0,:], trjR[i,1,:])
				except IndexError:
					print "Particle at " + str(i) + " indexed out!"
					continue
				if verbose and i%50 is 0:
					print "On particle " + str(i)
				if i%500 is 0:
					self.dumpToJSON()
				start = self.particles[i].getStart()
			self.makeFrameList(trjR.shape[2])		
		# We are rebuilding the structure from pre-built JSON data            
		else:
			self.reloadParticle(particle_json)
			self.reloadFrames(frame_json)
	
	def setVerbose(self, verb = None):
		if verb is None:
			self.verbose = not self.verbose
		else:
			self.verbose = verb

	def graphFrame(self, frame_num, give_data = False):
		# Given a frame number, graph all particles present on the surface
		particle_list = self.filterParticles(this_range = self.frames[frame_num])
		x_list = []
		y_list = []
		for i in particle_list:
			particle_index = frame_num - self.particles[i].getStart()
			x_list.append(self.particles[i].x_pos[particle_index])
			y_list.append(self.particles[i].y_pos[particle_index])
		
		if not give_data:
			fig1 = plt.figure()
			plt.plot(x_list, y_list, 's')
			return plt.gca()
		else:
			return x_list, y_list, particle_list
	
	def makeFrameList(self, full_range = 10000):
		# Correctly build listing of the different particles within a given frame number
		# TODO: Make it to where I don't need to pass in the number of frames.
		self.frames = {key:[] for key in range(full_range)}
		for i in self.particles:
			index = self.particles[i].getStart()
			for j in range(self.particles[i].getNumFrames()):
				self.frames[index + j].append(i)
			
	def graphMultipleFrames(self, frame_nums = [], give_data = False):
		# Leverages graphFrames to gather data/graph multiple frames of data
		data_sets_x = []
		data_sets_y = []
		particle_list = []
		j = 0
		for i in frame_nums:
			x, y, part_list = self.graphFrame(i, give_data = True)
			data_sets_x.append(x)
			data_sets_y.append(y)
			particle_list.append(part_list)
			j = j + 1
			
		if not give_data:
			fig1 = plt.figure()
			for i in range(j):
				plt.plot(data_sets_x[i], data_sets_y[i], 's')
			return plt.gca()
		else:
			return data_sets_x, data_sets_y, particle_list

	def dumpToJSON(self):
		all_particles = []
		for i in self.particles:
			all_particles.append(self.particles[i].__dict__)
		with open(self.name+'_particles.json', 'w') as fp1:
			json.dump(all_particles, fp1)
			fp1.close()
			
		all_frames = []
		for i in self.frames:
			all_frames.append(self.frames[i])
		with open(self.name+'_frames.json', 'w') as fp2:
			json.dump(all_frames, fp2)
			fp2.close()
			
	def reloadParticle(self, particle_dir):
		# This method takes a dirctory to a particle json file and rebuilds the particle listing
		with open(particle_dir, 'r') as fp:
			this_data = json.load(fp)
			fp.close()
			
		# print this_data
		i = 0
		for item in this_data:
			self.particles[i] = particle(from_json = True, **item)
			i = i + 1
			
	def reloadFrames(self, frame_dir):
		# This method takes a dirctory to a particle json file and rebuilds the particle listing
		with open(frame_dir, 'r') as fp:
			this_data = json.load(fp)
			fp.close()
			
			i = 0
		for item in this_data:
			self.frames[i] = item
			i = i + 1
	
	def calcHopDistribution(self, distance = 1, exclude = False):
		num_of_hops = []
		hop_data = []
		for i in self.particles:
			self.particles[i].calcJumpDists(jump_dist = distance)
			hop_list = self.particles[i].getJumpDistances()
			if hop_list is None:
				continue
			hop_data.append(hop_list)
			num_of_hops.append(len(hop_list))
		return num_of_hops, hop_data

	def getWaitTimeDistribution(self, distance = 1):
		# Returns the raw data for all wait times in the system
		wait_times = list()
		self.calcHopDistribution(distance = distance)
		for i in self.particles:
			w_t = self.particles[i].wait_times
			if w_t is None:
				w_t = []
			wait_times.append(w_t)

		# Fun list comprehension to unpack a list of lists
		if wait_times is None:
			return []
		else:
			return [item for sublist in wait_times for item in sublist]
	
	def getHopDFs(self, bins = 19):
		if self.cdf is not None and self.pdf is not None:
			return self.pdf, self.cdf
		else:
			num_hops, hop_data = self.calcHopDistribution()
			self.pdf, self.bins = np.histogram(num_hops, bins = range(bins))
			self.cdf = np.cumsum(self.pdf)/float(np.sum(self.pdf))
			return self.pdf, self.cdf

	def rollCall(self, step):
		for i in range(len(self.particles.keys())/step):
			print self.particles[i*step].rollCall()
	
	def graphCDF(self, percLine = 0.99):
		# Quick and dirty method to graph a CDF for a dataset.
		plt.figure()
		ax = plt.gca()
		print self.cdf
		if self.cdf is None:
			self.getHopDFs()
		x = range(self.cdf.shape[0])
		y = self.cdf
		ax.plot(x, y, label = 'CDF', linewidth = 3)

		# Draws a horizontal line to denote a vertical cut off along an axis
		ax.plot([5,18], [percLine,percLine], color = 'black', linestyle = '--', linewidth = 4)
		
		plt.ylim([0.95, 1.0])
		plt.xlim([5, 18])
		plt.yticks([0.95, percLine, 1])
		plt.xticks([5, 10, 15, 18])
		return ax

	def getFrameDistribution(self):
		num_frames = []	
		for i in self.particles:
			num_frames.append(self.particles[i].getNumFrames())
		return num_frames
		
	def getParticle(self, part_num):
		return self.particles[part_num]

	def calcPolarCoords(self):
		for i in self.particles:
			self.getParticle(i).calcPolarCoords()

	def getPolarDistribution(self, limit = None):
		# Gets all the polar angles/lengths from all particles in the widefield constuct
		# Check if all the particles have polar coordinates calculated.
		if self.getParticle(0).polar_angle is 0:
			self.calcPolarCoords()

		polar_coords  = np.zeros((len(self.particles)))
		polar_lengths = np.zeros((len(self.particles)))
		for i in range(polar_coords.shape[0]):
			polar_coords[i]  = self.getParticle(i).polar_angle
			polar_lengths[i] = self.getParticle(i).polar_length

		return polar_coords, polar_lengths

	def setParticleFilter(self, filt_type = 'polar_angle', filt_params = [10, 35]):
		# Defines a filter that will limit the intake of certain particles
		
		self.filt_type.append(filt_type)
		self.filt_params.append(filt_params)
		self.num_filters += 1

	def adjustParticleFilter(self, filt_index, remove = False, filt_type = None, filt_params = None):
		if remove:
			del self.filt_params[filt_index]
			del self.filt_type[filt_index]
			self.num_filters -= 1
		else:
			if filt_type is not None:
				self.filt_type[filt_index] = filt_type
			if filt_params is not None:
				self.filt_type[filt_index] = filt_params
		if self.verbose:
			self.listFilters()

	def clearFilters(self):
		self.filt_type = None
		self.filt_params = None
		self.num_filters = 0

	def listFilters(self):
		for i in range(self.num_filters):
			print 'Filter ', i, ' : ', self.filt_type[i], ' : ', self.filt_params

	def getImmobilePopulation(self):
		
		if len(self.immobile_pop) is not 0:
			return self.immobile_pop, self.single_incident_pop
		else:
			for i in self.particles.keys():
				immobile = self.particles[i].isImmobile()
				single_incident = self.particles[i].isSingleFrame()
				if immobile and not single_incident:
					self.immobile_pop.append(i)
				elif single_incident:
					self.single_incident_pop.append(i)
		return self.immobile_pop, self.single_incident_pop


	def filterParticles(self, this_range = 0):
		# Works as the actual filter for the information
		# Range is the range of particles we are interested in getting

		if type(this_range) is int:
			prospective_particles = range(this_range)
		else:
			prospective_particles = this_range
		interesting_particles = []
		
		if self.verbose:
			print len(prospective_particles), 'Prospective Particles: ', prospective_particles
			print 'Filter: ', self.filt_type, ', Parameters: ', self.filt_params

		if self.filt_type is None:
			return prospective_particles
		
		for i in prospective_particles:
			this_particle = self.particles[i] # The particle we are including
			inclusion_bool = True			  # A boolean to dictate that the particle is 'interesting'
			for j in range(self.num_filters):
				if self.filt_type[j] == 'polar_angle' and inclusion_bool is True:
					# Filtering particles by polar angle.
					# If the filter params is 2, then we have a lower bound and upper bound and want the items in between.
					p_a = self.particles[i].polar_angle
					if type(self.filt_params[j][1]) is int:
						if  p_a > self.filt_params[j][0] and p_a < self.filt_params[j][1]:
							inclusion_bool &= True
						else:
							inclusion_bool &= False

					elif self.filt_params[j][1] is 'ge':
						if p_a >= self.filt_params[0]:
							inclusion_bool &= True
						else:
							inclusion_bool &= False

					elif self.filt_params[j][1] is 'le':
						if p_a <= self.filt_params[0]:
							inclusion_bool &= True
						else:
							inclusion_bool &= False

				if self.filt_type[j] == 'num_frames' and inclusion_bool is True:
					n_frames = self.particles[i].getNumFrames()
					if type(self.filt_params[j][1]) is int:
						if n_frames > self.filt_params[j][0] and n_frames < self.filt_params[j][1]:
							inclusion_bool &= True
					
					elif self.filt_params[j][1] is 'ge' and n_frames >= self.filt_params[j][0]:
						inclusion_bool &= True
					
					elif self.filt_params[j][1] is 'le' and n_frames <= self.filt_params[j][0]:
						inclusion_bool &= True
					
					else:
						inclusion_bool &= False 
				
				if self.filt_type[j] == 'num_hops' and inclusion_bool is True:
					n_hops = self.particles[i].getNumHops()
					if type(self.filt_params[j][1]) is int:
						if n_hops > self.filt_params[j][0] and n_hops < self.filt_params[j][1]:
							inclusion_bool &= True
					
					elif self.filt_params[j][1] is 'ge' and n_hops >= self.filt_params[j][0]:
						inclusion_bool &= True
					
					elif self.filt_params[j][1] is 'le' and n_hops <= self.filt_params[j][0]:
						inclusion_bool &= True
					
					else:
						inclusion_bool &= False 
			if inclusion_bool:
				interesting_particles.append(i)

		if self.verbose and len(interesting_particles) == 0:
			print 'There are no particles that match the given parameters'

		if self.verbose:
			print len(interesting_particles),' Interesting Particles: ', interesting_particles
		return interesting_particles

"""
	v, w = linalg.eigh(cov)
	u = w[0] / linalg.norm(w[0])
	angle = np.arctan(u[1] / u[0])
	angle = 180 * angle / np.pi  # convert to degrees
	# filled Gaussian at 2 standard deviation
	ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
	                      180 + angle, facecolor=color,
	                      edgecolor='yellow',
	                      linewidth=2, zorder=2)
	ell.set_clip_box(splot.bbox)
	ell.set_alpha(0.5)
	fig.add_artist(ell)
	fig.set_xticks(())
	fig.set_yticks(())
"""
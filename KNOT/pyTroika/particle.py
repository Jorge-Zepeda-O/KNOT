# Import block
import numpy as np

# Set up logging
import logging
###############################################################################################################################################################################################
################################################################################## Particle Class #############################################################################################
###############################################################################################################################################################################################

class particle:
		
	def __init__(self, xs = None, ys = None, from_json = None, x_pos = None, y_pos = None, sigma = None, start = None, frames = None, dict_key = None, msd = None, jump_dist = None, hops = None, area_covered = None, polar_length = 0, polar_angle = 0, num_blinks = 0, wait_times = None, is3D = False, verbose = False):
		
		"""
		Values needed/given at instantiation
		"""
		self.raw_x_pos = np.insert(np.array([]),0,x_pos)		# List of x-localization coordinates from initial recognition
		self.raw_y_pos = np.insert(np.array([]),0,y_pos)		# List of y-localization coordinates from initial recognition
		self.init_sigma = np.insert(np.array([]),0,sigma)		# List of sigmas from initial localization
		self.start_frame = start 								# First frame where this particle was localized
		self.frames = np.insert(np.array([]),0,start)			# Listing of frames that the particle is active; Should be linear unless we expand how long a particle can disappear/reappear				
		self.rejection = np.array([])							# List of frames where the particle was rejected by super resolution methods.
		
		"""
		Values updated later
		"""
		self.x_pos = np.array([])				# Super-localized x-positions
		self.y_pos = np.array([])				# Super-localized y-positions
		self.super_res_sigma = np.array([])		# Sigma/radii as found by blob-finder
		self.dict_key = dict_key				# The dictionary key as stored in troika
		self.verbose = verbose 					# Verbosity Flag
		
		"""
		Calculated values
		"""
		self.msd = 0							# Mean squared displacement
		self.hops = 0							# Number of hops detected in the trajectory
		self.jump_dist = None 					# Distances of each jump
		self.area_covered = None 				# The area explored by a particle during the trajectory
		self.num_blinks = 0 					# Number of frames 
		self.polar_angle = 0					# Polar angle for hop/length detection
		self.polar_length = 0					# Polar length for hop/length detection
		self.wait_times = np.array([])			# The array of wait times between hops

		self.startLogger()						# Create the logger for the system.
		"""
		Reconstitute a particle from a stored json_file
		"""
		if from_json is not None:
			self.fromJSON(from_json)

		if is3D:
			self.raw_z_pos = np.insert(np.array([]),0,x_pos)	# Raw localized coordinates for 3D tracking
			self.z_pos = np.array([])							# Super localized coordinates for 3D tracking

###########################################################################################
################################ Setter Methods  ##########################################
###########################################################################################

	def setSuperResCoords(self, x, y, sigma, frame_num):
		"""
		Method to update the listing of super resolution centers
		
		Inputs:
		x - The x coordinate of the super-resolved center
		y - The y coordinate of the super-resolved center
		sigma - The sigma radius of hte localization
		frame_num - The frame number where this particle is localized.

		Outpus:
		None
		"""
		if self.frames[-1] != frame_num:
			self.logger.error('particle.setSuperResCoords: there is something off with the frames.')
		self.x_pos = np.insert(self.x_pos, len(self.x_pos), x)
		self.y_pos = np.insert(self.y_pos, len(self.y_pos), y)
		self.super_res_sigma = np.insert(self.super_res_sigma, len(self.super_res_sigma), sigma)

	def setColor(self, color):
		"""
		Setter method to give a particle a certain color to use when graphing. Color must be compliant
		with a call to matplotlib.

		Inputs:
		color - The color that should be used for graphing this particle.
		"""
		self.color = color

	def startLogger(self, level = 'DEBUG'):
		"""
		Establish error logging for the system. Used the following link https://docs.python.org/3.7/howto/logging-cookbook.html#multiple-handlers-and-formatters
		"""
		# Need to create if statement to switch between logging levels.
		"""
		logging.basicConfig(level=logging.DEBUG,
							format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
							datefmt='%m-%d %H:%M')
		"""
		# define a Handler which writes INFO messages or higher to the sys.stderr
		self.logger = logging.getLogger('particle')
		self.logger.setLevel(logging.DEBUG)
		"""
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		ph.setFormatter(formatter)
		"""

		# Stops the info from going to root. Will want to make this an option at some point.
		self.logger.propagate = False
###########################################################################################
################################ Getter Methods  ##########################################
###########################################################################################

	def getRawCoords(self, frame_num):
		"""
		This method returns the _raw_ coordinates at a given frame number
		Note: The user should default to getCoordPairs. This function is deprecated
		
		Inputs:
		frame_num - return the coordinates of the particle in a specific frame number.

		Outputs:
		The raw x,y coordinates of the particle in that specific frame.
		"""
		frame_index = self.getFrameIndex(frame_num)
		if frame_index < 0:
			if self.verbose:
				self.loggere.error('The particle does not exist in that frame')
			return -1
		return self.raw_x_pos[frame_index], self.raw_y_pos[frame_index]

	def getSuperResCoords(self, frame_num):
		"""
		This method returns the _raw_ coordinates at a given frame number
		Note: The user should default to getCoordPairs. This function is deprecated
		
		Inputs:
		frame_num - return the coordinates fo the particle in a specific frame number

		Outputs:
		The super-resolved x,y coordinates of the particle in that specific frame.
		"""
		frame_index = self.getFrameIndex(frame_num)
		if frame_index < 0:
			if self.verbose:
				self.logger.error('particle.getSuperResCoords: The particle does not exist in that frame')
			return -1
		elif len(self.x_pos) == 0:
			if self.verbose:
				self.logger.error('particle.getSuperResCoords: The particle has not been super-localized yet.')
			return -1
		return self.x_pos[frame_index], self.y_pos[frame_index]

	def getXCoords(self):
		"""
		This method returns the current x-coordinates of the system, preferentially choosing the super-res coordinates when available.

		Inputs:
		None

		Outputs:
		Returns a copy of the most accurate x-coordinates available to the system.
		"""
		if len(self.x_pos) != 0:
			return np.copy(self.x_pos)
		elif len (self.raw_x_pos) != 0:
			return np.copy(self.raw_x_pos)
		else:
			self.logger.error('No x-coordinates have been assigned')
			return -1

	def getYCoords(self):
		"""
		This method returns the current y-coordinates of the system, preferentially choosing super-res coordinates when available.

		Inputs:
		None

		Outputs:
		Returns a copy of the most accurate y-coordinates available to the system.
		"""
		if len(self.y_pos) != 0:
			return np.copy(self.y_pos)
		elif len(self.raw_y_pos) != 0:
			return np.copy(self.raw_y_pos)
		else:
			self.logger.error('No y-coordinates have been assigned')
			return -1

	def getZCoords(self):
		"""
		This method will return the most accurate z coordinates available assuming z-coordinates are available.

		Inputs:
		None

		Outputs:
		Returns a copy of the z coordinates
		"""
		try:
			if len(self.z_pos) != 0:
				return np.copy(self.z_pos)
			elif len(self.raw_z_pos) != 0:
				return np.copy(self.raw_z_pos)
			else:
				self.logger.error('No z-coordinates have been assigned')
				return -1
		except:
			pass

	def getCoordPairs(self, frame_num = None, jitter = False):
		"""
		Method to automatically return an nx2(3) vector of the particle coordinates. This is the default method and should be used in all cases.
		
		We currently assume that frame_num defines the last point that we want. Is it a bad
		idea to have getCoordPairs do the separation or should that be handled on the other
		side?

		Inputs:
		frame_num - Dictates that last frame of the trace. The array is from the particle beginning up to that frame.
		jitter - Dictates whether slight jitter should be added to particle centers if they are too close to see on a grid.

		Outputs:
		A coordinate array including all x,y, and z coordinates (if available)
		"""
		if frame_num is None:
			frame_num = (0, self.getNumFrames()-1)
		elif len(frame_num) == 1:
			frame_index = self.getFrameIndex()
			if frame_index != -1:
				frame_num = (0, frame_index)
			else:
				return -1
		if jitter:
			x = rand_jitter(self.getXCoords())
			y = rand_jitter(self.getYCoords())
		else:
			x = self.getXCoords()
			y = self.getYCoords()

		if self.getZCoords() is not None:
			coords = np.array([self.getXCoords(), self.getYCoords(), self.getZCoords()]).T
		else:
			coords = np.array([x, y]).T

		if coords.shape[0] == 0:
			return np.array([0,0])
			if self.verbose:
				self.logger.debug('particle.getCoordPairs: The particle has no associated coordinates.')

		return coords[frame_num[0]:frame_num[1]]

	def getStart(self):
		"""
		Return the first frame that this particle is active.

		Inputs:
		None

		Outputs:
		The number of the first frame where this particle is identified.
		"""
		return self.start
	
	def getNumFrames(self):
		"""
		Return the number of frames that this particle is active

		Inputs:
		None

		Outputs:
		Return the number of frames the particle is active.
		"""
		return len(self.frames)

	def getNumHops(self, jump_dist = 1):
		"""
		Return the number of times the particle is detected as moving some number of pixels away.

		Inputs:
		jump_dist - the minimum distance that can be considered a 'jump'

		Outputs:
		The number of times a particular particle jumps a distance further than jump_dist
		"""
		self.calcJumpDists(jump_dist = jump_dist)
		if self.hops is None:
			return 0
		return len(self.hops)

	def getJumpDistances(self):
		"""
		Return the set of all jump distances.

		Inputs:
		None

		Outputs:
		Array of all jump distances. Should be an array/list of len(frame_number)-1 entries.
		"""
		if self.hops is None:
			return []
		return self.hops

	def getWindowDims(self, frame_num, n_sigma = 4):
		"""
		Returns the dimensions of the window around the raw particle center with n_sigma of variance

		Inputs:
		frame_num - Directs what frame you want to pull the window from.
		n_sigma - number of sigma to use as height/width of window. Assumed to be a symmetric square

		Outputs;
		frame_dims - tuple of form [min_height, max_height, min_width, max_width]
		"""
		frame_index = self.getFrameIndex(frame_num)
		sigma = self.init_sigma[frame_index]
		min_height = self.raw_x_pos[frame_index] - n_sigma * sigma
		max_height = self.raw_x_pos[frame_index] + n_sigma * sigma
		min_width = self.raw_y_pos[frame_index] - n_sigma * sigma
		max_width = self.raw_y_pos[frame_index] + n_sigma * sigma

		# Round all indices up then type cast them to ints so they can be used in coordinate systems.
		dims = np.ceil(np.array([min_height, max_height, min_width, max_width])) # I don't know why these came in as matrices but they did.
		return dims.astype(int)

	def getFrameIndex(self, frame_num):
		"""
		Find the list index that corresponds to the requested frame in the stored particle data.

		Inputs:
		frame_num - The frame_number that the user wants to look at.

		Outputs:
		The index of the requested frame number in the x,y coordinate arry.
		"""
		try:
			return np.where(self.frames==frame_num)[0][0]
		except:
			self.logger.error('getFrameIndex: The particle does not exist in that frame')
			return -1

	def getKey(self):
		"""
		Method returns the key that indicates where this particle is stored.

		Inputs:
		None

		Outputs:
		The troikaExperiment assigned dictionary key for the particle dictionary in troikaExperiment.
		"""
		return self.dict_key

	def getFirstFrame(self):
		"""
		Returns the first frame that a particle is detected to appear

		Inputs:
		None

		Outputs:
		Return the frame number of the first frame that the particle is active
		"""
		return self.frames[0]

	def getLastFrame(self):
		"""
		Returns the last frame in the list. Ideally, this should be the only frame that can be 
		attached to an older particle.

		Inputs:
		None

		Outputs:

		"""
		return self.frames[-1]

	def getFrameSet(self):
		"""
		Return a set of all frames where this particle is present.

		Inputs:
		None

		Outputs:
		The set of all frames where the particle is present.
		"""
		return set(self.frames)

	def getParticleTrace(self, frame_num = None):
		"""
		Return the coordinate data up to the value requested by frame_num

		Inputs:
		frame_num - The last frame of the coordinates as dictated by the user. 

		Outputs:
		The coorindate trace

		Note:
		This method could be modified as getCoordPairs() already has this trace abridging functionality. This method is likely
		a better default but should become a wrapper for the getCoordPairs method. Also, it does not currently support referencing
		of a 3D trace.
		"""
		if frame_num is None:
			frame_num = -1
		else:
			frame_index = self.getFrameIndex(frame_num)

		if frame_index == -1:
			self.logger.error('The particle does not exist in frame'+frame_num)
			return None

		return self.getCoordPairs()[:, :frame_num]

	def getLastPoint(self, origin_adjusted = False):
		"""
		Get the last position of the trace

		Inputs:
		origin_adjusted - Boolean dictating if the last point needs to be origin adjusted.

		Outputs:
		The final point in the selected coordinate system
		"""
		if not origin_adjusted:
			return self.getCoordPairs()[-1,:]
		else:
			return self.calcOriginCoordinates()[0][-1,:]

	def getFirstPoint(self, origin_adjusted = False):
		"""
		Get the last position of the trace

		Inputs:
		origin_adjusted - Boolean dictating if the last point needs to be origin adjusted.

		Outputs:
		The first point in the selected coordinate system
		"""
		if not origin_adjusted:
			return self.getCoordPairs()[0,:]
		else:
			return self.calcOriginCoordinates()[0][0,:]
###########################################################################################
################################ Increment Methods ########################################
###########################################################################################
	def markRejection(self, frame_num):
		"""
		Count the number of times the particle has been rejected by the localization standard

		Incomplete method. Meant to help isolate the particles that are experiencing the most
		difficulty in localization.
		"""
		np.insert(self.rejection,len(self.rejection),frame_num)
		pass

###########################################################################################
################################ Particle Collapse Methods ################################
###########################################################################################
	def collapse(self):
		"""
		Return all initialized values so that we can combine two particles together. This method
		squashes all important values from our particle and returns them as a dictionary that
		can be directly consumed by the adsorb method of a second host particle.

		Inputs:
		None

		Outputs:
		ret_dict - A dictionary of all the important values in a particle instance.
		"""
		ret_dict = dict()
		ret_dict['raw_x_pos'] = self.raw_x_pos
		ret_dict['raw_y_pos'] = self.raw_y_pos
		ret_dict['init_sigma'] = self.init_sigma
		ret_dict['frames'] = self.frames
		ret_dict['x_pos'] = self.x_pos
		ret_dict['y_pos'] = self.y_pos
		ret_dict['super_res_sigma'] = self.super_res_sigma
		ret_dict['rejection'] = self.rejection

		try:
			ret_dict['raw_z_pos'] = self.raw_z_pos
			ret_dict['z_pos'] = self.z_pos
		except:
			pass

		return ret_dict

	def absorb(self, ret_dict):
		"""
		Receive the information from an old particle and concatenate its contents produced from a
		call to the collapse function. Currently assumes that an older particle is only consuming
		a younger particle.

		These methods should still be called when doing a back trace to combine trajectories; However,
		it will be necessary to check the _order_ of particles so traces are added "tip-to-tail"

		Inputs:
		ret_dict - A dictionary provided by another collapsed particle.

		Outputs:
		None
		"""
		# There need to be many, many checks to make sure that everything gets added correctly. This code assumes that the new particles get added to older particles.
		axis_opt = 0
		self.raw_x_pos = np.concatenate((self.raw_x_pos, ret_dict['raw_x_pos']), axis = axis_opt)
		self.raw_y_pos = np.concatenate((self.raw_y_pos, ret_dict['raw_y_pos']), axis = axis_opt)
		self.init_sigma = np.concatenate((self.init_sigma, ret_dict['init_sigma']), axis = axis_opt)
		self.frames = np.concatenate((self.frames, ret_dict['frames']), axis = axis_opt)
		self.rejection = np.concatenate((self.rejection, ret_dict['rejection']), axis = axis_opt)
		self.x_pos = np.concatenate((self.x_pos, ret_dict['x_pos']), axis = axis_opt)
		self.y_pos = np.concatenate((self.y_pos, ret_dict['y_pos']), axis = axis_opt)
		self.super_res_sigma = np.concatenate((self.super_res_sigma, ret_dict['super_res_sigma']), axis = axis_opt)

		if "z_pos" in ret_dict.keys():
			self.z_pos = ret_dict['z_pos']
			self.raw_z_pos = ret_dict['raw_z_pos']

###########################################################################################
############################# Blink Detection Methods #####################################
###########################################################################################

	"""
	I pulled and converted these methods directly from the original code as provided by Wenxiao.
	As of now, they have seen no use in the original code.
	"""
	def blinkingDetection(self, dT = 1):
		"""
		This method is a recreation of our blinking code seen in the MLE github. It utilizes
		information theory to account for blinking in the diffusion calculation.
		"""
		R = 0
		data = self.getParticleTrace()
		output = np.zeros(3)
		D_i = self.indpblink(data)/4/dT
		sigma2_i = D_i
		if True: # Original method anticipates other possible arguments and compensates for them.
			# Optimizer/iterator options are set here to open the ceiling of optimization
			likelihoodfunc = lambda b: -self.likelihoodSubFunction(data, b[0], b[1], dT, R)
			x_opt = fmin(func = likelihoodfunc, x0 = [D_i, sigma2_i])
			output[0] = x_opt[0]
			output[1] = x_opt[1]
			output[2] = self.likelihoodSubFunction(data, output[0], output[1],dT,R)
		else: # Open up to use variable arguments later
			pass
		return output

	def likelihoodSubFunc1D(self, dXX, D, sigma2, dT, R):
		"""
		A sub function of the blink detection method where dXX is the square distance of diff
		of trajectories with mean subtracted and k_th element multiplied by (-1)^k
		"""

		if D<0 or sigma2<0:
			return np.NINF
		else:
			ncero = np.where(dXX != 0)
			sigma = np.zeros((ncero.size-1,ncero.size-1))
			delta = np.zeros(ncero.size-1)
			for i in range(ncero.size-1):
				inc = ncero[i+1] - ncero[i]
				delta[i] = (dXX[ncero[i+1],1]-dXX[ncero[i],1])
				sigma[i,i] = (2*D*dT*inc+2*sigma2)

				if i+1 < ncero.size:
					sigma[i, i+1] = -sigma2
					sigma[i+1, i] = -sigma2
			L = -0.5*np.log(det(sigma))-0.5*delta/(sigma)*inv(delta)
		
		return L

	def likelihoodSubFunction(self, dXX, D, sigma2, dT, R):
		"""
		Sub function to run for all entries in the array dXX
		"""
		L = 0
		for i in range(dXX.size[1]):
			L += likelihoodSubFunc1D(dXX[:,kk])
		return L

	def indpblink(self):
		ncero = np.where(data[:,0] != 0)
		gain = 0
		for i in range(ncero.size-1):
			gain += norm(data[ncero[i+1]],data[ncero[i]])**2/(ncero[i+1]-ncero[i])
		return gain/(ncero.size-1) 

###########################################################################################
################################ Visulization Methods #####################################
###########################################################################################

	def plotParticle(self, fig_handle = None, color = 'b', movie_len = None, cmap = 'viridis', use_origin_coords = False, jitter = False):
		"""
		This method is designed to plot a particle trace/return the information necessary to plot a particle's trace
		
		Inputs:
		fig_handle - A valid matplotlib figure handle for plotting to an already generated figure.
		color - A non-gradient color to make the lines TODO: This need to be improved as this does not work with line collection
		movie_len - The number of line segments to have in the line collection
		cmap - The colormap to use for the color gradient
		use_origin_coords - Boolean to dictate if the system should calculate a line collection based on teh origin coordinates
		jitter - Boolean to add jitter to line segments if they overlap too much.

		Outputs:
		lc - A line collection containing all the necessary information to plot a color gradient trace.
		"""
		if fig_handle is None:
			fig_handle = plt.figure()

		"""
		Line segements require both a startpoint and an endpoint. This language guarantees
		that the proper points exist to form the line segments that we want. We slice into the coords
		to make sure the coordinates return correctly for matplotlib image drawing.
		"""
		if not use_origin_coords:
			coordinates = self.getCoordPairs(jitter = jitter)[:,[1,0]].reshape(-1,1,2)
		else:
			coordinates = self.calcOriginCoordinates(jitter = jitter)[0][:,[1,0]].reshape(-1,1,2)
		coordinates = np.concatenate([coordinates[:-1], coordinates[1:]], axis=1)

		"""
		In order to get segments with different colors, we need to use matplotlib collections (https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/multicolored_line.html)
		The first segment is defining how we want the colors to look. We want to open up several different normalization options. One would be how that segment relates to the rest
		of the movie, at which point we need to use movie_len. Otherwise,  we might want it to be self comparative where every molecule uses the same colormap and the color is assigned
		based on the frame of the line segment.
		"""
		if movie_len is None:
			norm = plt.Normalize(0, self.getNumFrames())
		else:
			norm = plt.Normalize(0, movie_len)
		lc = LineCollection(coordinates, cmap = cmap, norm = norm)
		lc.set_array(np.arange(self.getNumFrames()))
		return lc

###########################################################################################
################################ Calculation Methods ######################################
###########################################################################################
	
	def calcMSD(self, tau = 1, style = 0):
		"""
		This method calculates the MSD of the particle given a trajectory. 
		TODO: Error check for any minor problems.

		Inputs:
		tau - The 'lag' factor that the user wants to use.
		style - Dictates if the user just wants the first difference or to recurse through all taus.

		Outputs:
		Returns the MSD that is appropriately matched to the style
		"""

		if tau > self.getNumFrames():
			if self.verbose:
				self.logger.error("The time lag is greater than the frame count")
			return -1

		displacement = 0
		i = tau
		if style == 0:
			firstX = self.getXCoords()[:-tau]
			lagX = self.getXCoords()[tau:]
			firstY = self.getXCoords()[:-tau]
			lagY = self.getXCoords()[tau:]
			displacement = np.sum((lagX-firstX)**2 + (lagY-firstY)**2)

		# TODO: Error check to make sure this is returning good numbers.
		elif style == 1:
			coordinates = self.getCoordPairs()
			while i < self.getNumFrames():
				first = np.array((coordinates[i-tau,0], coordinates[i-tau,1]))
				lag = np.array((coordinates[i], coordinates[i]))
				displacement += np.square(np.linalg.norm(lag-first))
				i += 1
		
		# Need to create a dictionary of tau values. Maybe build in a diffusion coefficient method that calculates all possible lags?
		#self.MSD = displacement/self.getNumFrames()
		return displacement/(self.getNumFrames()-tau)

	def calcDiffusion(self, fit = False):
		"""
		Calculate a diffusion constant and report deviations from the line.

		Inputs:
		fit - If fit is true, the system tries to fit the system to a line and returns fit params
			  else, the system just returns the msds and taus so the user can graph them.
		"""
		msds = np.zeros((self.getNumFrames()-1))
		taus = np.arange(1, self.getNumFrames())
		for i in taus:
			msds[i-1] = self.calcMSD(tau = i)

		if fit:
			D = lambda taus, Diff_con: 4 * Diff_con * taus
			popt, pcov = curve_fit(D, taus, msds)
			return popt, [msds, taus], pcov

		return msds, taus

	def calcJumpDists(self, jump_dist = 1):
		"""
		This method calculates the distance a particle jumps between each coordinate and stores
		it in an instance variable.

		Inputs:
		jump_dist - The minimum distance to qualify a motion as a jump.

		Outputs:
		None, assignment is done locally.
		"""

		"""
		Catch if there is only one frame, so no hops
		"""
		self.jump_dist = jump_dist
		if self.isSingleFrame():
			self.hops = 0
		else:
			"""
			If there is more than one frame, calculate distances
			"""
			firstX = self.getXCoords()[:-1]
			lagX = self.getXCoords()[1:]
			firstY = self.getYCoords()[:-1]
			lagY = self.getYCoords()[1:]
			self.hops = np.sqrt((lagX-firstX)**2 + (lagY-firstY)**2)

	def calcPolarCoords(self):
		"""
		This calculates the polar angles between jumps per frames. The idea is that
		the polar angle will provide a new measure for mobility/diffusivity.

		Inputs:
		None

		Outputs:
		None, values are stored as an instance variables.
		"""
		if self.hops is None:
			hops = 0
		else:
			hops = len(self.hops)
		# The old measure which calculated tradition polar. We want to adjust it for frame numbers
		# self.polar_length = np.sqrt(hops**2 + self.num_frames**2)
		self.polar_length = self.num_frames
		self.polar_angle = np.arctan(float(hops)/self.num_frames)

	def calcRadiusOfGyration(self, frame_step = None):
		"""
		Calculation of the radius of gyration value detailed in DOI: 10.1039/C0CP01805H (Paper) Phys. Chem. Chem. Phys., 2011, 13, 4326-4334
		# TODO: Alias this for 3D motion
		
		Inputs:
		frame_step - dictates the last value to be used in the tensor grid

		Outputs:
		The 'radius of gyration' values in x and y. Needs to be aliased for 3D motion.
		"""

		xs = self.getXCoords()[:frame_step]
		ys = self.getYCoords()[:frame_step]
		N = xs.shape[0]
		Ex = np.mean(xs)
		Ey = np.mean(ys)

		tensor = np.array([[np.sum(np.square(xs-Ex))/N, np.sum((xs-Ex)*(ys-Ey))/N],
						   [np.sum((xs-Ex)*(ys-Ey))/N,  np.sum(np.square(ys-Ey))/N]])

		eig_vals, eig_vectors = np.linalg.eig(tensor)

		return np.sqrt(eig_vals[0]**2 + eig_vals[1]**2)
	
	def calcTortuosity(self, log = False):
		"""
		Calculation of the tortuosity of a particles trajectory. As discussed in:
		https://pubs.acs.org/doi/abs/10.1021/acsami.7b15335

		Inputs:
		log - Boolean whether to return the log of tortuosity or the raw tortuosity value.

		Outputs:
		tort - The calculated tortuosity of the trajectory.
		"""
		if self.calcFinalDistance() == 0:
			tort = self.calcTotalDistanceTraveled()/0.1
		else:
			tort = self.calcTotalDistanceTraveled()/self.calcFinalDistance()

		if log:
			tort = np.log10(tort)
		return tort

	def calcTotalDistanceTraveled(self):
		"""
		Calculate the total end-to-end distance that the particle travels along its trajectory
		"""
		self.calcJumpDists()
		return np.sum(self.hops)

	def calcFinalDistance(self):
		"""
		Calculate the distance from the startpoint to the end point.
		"""
		coordinates = self.getCoordPairs()
		if coordinates.shape[0] <= 1:
			return 0
		return np.linalg.norm(coordinates[0,:]-coordinates[-1,:])

	def calcOriginCoordinates(self, origin_coords = None, jitter = False):
		"""
		Calculates an origin corrected set of coordinates as well as classifies how many 
		quadrants that the trajectory migrates through. 

		Inputs:
		origin_coords - Give a secondary set of coordinates to reference when normalizing
		jitter - Add jitter to the origin coordinates

		Outputs:
		norm_coords - The coordinates adjusted to start at the origin
		angles - Agglomeration of all the quadrants a particle explores
		distances - Overall distance traveled by the particle.
		"""

		coords = self.getCoordPairs(jitter = jitter)
		
		# TODO: Find out why some particles have empty coordinates
		if coords.shape[0] == 0:
			return np.array([0,0]), [0], [0]
		
		# TODO: Make this take a user given point and normalize all traces to this point
		if origin_coords == None:
			norm_coords = coords[0,:]

		rad2deg = 360/(2*np.pi)

		norm_coords = coords - np.ones((coords.shape[0],2)) * norm_coords
		dx = coords[1:,0]-coords[:-1,0]
		dy = coords[1:,1]-coords[:-1,1]

		# Need to add my own angle calculated that catches for dx == 0
		with np.errstate(divide='ignore'):
			angles = np.arctan(dy/dx)

		adj = 0
		for i in range(dx.shape[0]):
			# Check if arctan is returning a value that should be in II, or III
			if dx[i] < 0:
				# We are in II but are reported in four
				if dy[i] > 0:
					adj = np.pi
				else:
					adj = -np.pi
			else:
				adj = 0

			if angles[i] < 0:
				angles[i] += 2*np.pi

			angles[i] + adj
		distances = np.sqrt(dx**2 + dy**2)
		return norm_coords, np.nan_to_num(angles,copy=False) * rad2deg, distances

###########################################################################################
################################### Booleann Methods ######################################
###########################################################################################

	def isSingleFrame(self):
		"""
		Easy boolean method to check if the particle only exists for only one frame.

		Inputs:
		None

		Outputs:
		None
		"""
		return self.getNumFrames() == 1

	def isImmobile(self):
		"""
		Boolean method to check if the particle fails to move during it's time in frame.

		Inputs:
		None

		Outputs:
		None
		"""
		if self.hops is None:
			return True
		else:
			return len(self.hops) == 0

###########################################################################################
#################################### Saving Methods #######################################
###########################################################################################
	
	def dumpToJSON(self, directory = './example.json', TECalled = True):
		"""
		Dump all the particle information into a json file.

		Inputs:
		directory - directory to save the particle json
		TECallled - Marks if this method was called by an instance of troikaExperiment. If so, return the
		JSON as a dictionary so it can be combined in the troikaInstance JSON dump.
		"""
		j_dict = dict()

		# Need to dynamically parse through the class dictionary. If the value is 0/None, skip it. Else,
		# we need to convert it from a numpy array to a list.
		for i in self.__dict__.keys():
			val = self.__dict__[i]

			# Exclude uninteresting/none values for conciseness
			if val is None or i == 'verbose' or (type(val) is np.ndarray and len(val) == 0):
				continue
			if type(val) is np.ndarray:
				val = val.tolist()
			j_dict[i] = val

		# If called by TroikaExperiment, return a valid dictionary of all the important value
		if TECalled:
			return j_dict
		# If called independently, save to current directory by oneself
		else:
			with open(directory, 'w') as fp:
				json.dump(j_dict, fp)
				fp.close()

	# Need to reconstitute the particle from JSON here.
	def fromJSON(self, j_dict):
		# Iterate through the keys in self.__dict__ assign appropriately. Note, the number of keys
		# in the paced in dictionary should always be less than or equal to the keys in __dict__S

		# Inserts any values matched with keys in both dictionaries 
		for i in self.__dict__.viewkeys() & j_dict.viewkeys():
			if type(j_dict[i]) is list:
				self.__dict__[i] = np.array(j_dict[i], dtype='float16')
			else:
				self.__dict__[i] = j_dict[i]

	def testMethod(self, args = None):
		"""
		Test method to be implemented and changed on the fly.
		"""
		pass

	def dumpTrajectoryToCSV(self, filename = None, origin_corrected = False):
		"""
		Method to dump the trajectory of a particle into a CSV file for processing outside of the troikaExperiment environment

		Inputs:
		filename - The name of the csv file for the particle.
		origin_corrected - A boolean indicating whether the returned trajectory needs to be corrected for the origin coordinates

		Outputs:
		None
		"""
		if filename == None:
			filename = str(self.key)+'.csv'

		if origin_corrected:
			coords = part.calcOriginCoordinates()[0]
		else:
			coords = part.getCoordPairs()

		out_frame = pd.DataFrame({'x':coords[:,1], 'y': coords[:,0]})
		out_frame.to_csv(filename)

"""
Need to write a quick and easy way to dump particle trajectories into an excel/csv file. Below is some code I used for the collaborator.

cell = 3
for i in range(3):
    part = t_e.getParticle(middle_set_keys[i])
    coords = part.calcOriginCoordinates()[0]
    plt.figure()
    plt.plot(coords[:,1],coords[:,0])
    this = pd.DataFrame({'x':coords[:,1], 'y': coords[:,0]})
    this.to_csv('cell_'+str(cell)+'_middle_'+str(i)+'.csv')

    part = t_e.getParticle(last_set_keys[i])
    coords = part.calcOriginCoordinates()[0]
    plt.figure()
    plt.plot(coords[:,1],coords[:,0])
    this = pd.DataFrame({'x':coords[:,1], 'y': coords[:,0]})
    this.to_csv('cell_'+str(cell)+'_last_'+str(i)+'.csv')
"""
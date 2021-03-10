# Base scientific imports
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection # Necessary for time coloration on particle tracks.

# Numpy Imports
from numpy.linalg import norm
from numpy.linalg import det
from numpy.linalg import inv

# Scipy imports
from scipy.ndimage.filters import convolve
from scipy.optimize import fmin
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

# skimage imports
from skimage.feature import blob_log
from skimage import io
from skimage.segmentation import random_walker
from skimage.exposure import rescale_intensity
from skimage.exposure import equalize_hist
from skimage import img_as_float
from skimage.morphology import reconstruction
from skimage.morphology import opening
from skimage.morphology import white_tophat
from skimage.morphology import square
from skimage.morphology import disk

# Writer for directory
import os

# Debugging writer
import logging

# GIF Writer
import imageio

# Timing packages
import time

# Self-made codes
from particle import *
from helperFuncs import *

class troikaExperiment:
	"""
	troikaExperiment is the master class that holds the original image stack, performs image cleanup, and performs particle linking duties. It should be designed such that methods can be outward facing and 
	easiliy extrapolated to a GUI. The particles themselves are stored as separate instances of a particle class and are retained in a master key list by troikaExperiment.
	"""
	def __init__(self, data_dir =  None, name = 'myTroikaExperiment', eager = True, verbose = False, particle_collapse_verbose = False, check_frame_compatibility_verbose = False, frame_num = 0, get_frame_verbose = False, map_particle_verbose = False, is_linking = False, frame_tolerance = 1, track_time = False, from_json = None):
		"""
		Initialization method for troikaExperiment.

		Inputs:
		data_dir			Machine URL to find the original image stack that is the experimental data to be analyzed.
		eager 				Initialized Boolean that initializes settings for data manipulation. Can be set to False if user wants to institute different settings.
		verbose 			Boolean that turns checkpoint statements on/off
		frame_num 			The first frame that the system should look at. Necessary whenever eager is set to True.
		get_frame_verbose	Boolean for debugging the getFrame method
		is_linking			Informs the system that it should immediately start linking particles between frames.
		frame_tolerance 	The blinking tolerance for the linking process
		track_time 			Track the amount of time the system has spent identifying/linking particles.
		from_json 			A url pointing to a folder containing a previous instance of troikaExperiment. This contains all particles/particle information but only stores the data
							url for the original image stack.

		Outputs:
		None
		"""

		"""
		Necessary for data pulling and saving
		"""
		self.data_dir = data_dir					# Directory URL to find the image stack
		self.name = name 							# The 'name' of the class. Necessary for saving the system to JSON as it decides what the folder string will be

		"""
		Booleans for debuggin
		"""
		self.verbose = verbose 														# Boolean flag to enable debugging statements
		self.get_frame_verbose = get_frame_verbose									# Boolean flag to enable getFrame debugging statements
		self.track_time = track_time 												# Boolean flag to enable tracking of wall clock time as the system performs the requested actions.
		self.map_particle_verbose = map_particle_verbose 							# Boolean flag to enable print statements during the particle mapping process
		self.particle_collapse_verbose = particle_collapse_verbose 					# Boolean flag to enable print statements during the particle collapse process
		self.check_frame_compatibility_verbose = check_frame_compatibility_verbose 	# Boolean flag to enable print statements during the checkFrameCompatibility function

		"""
		Image data storage
		"""
		self.data = None 							# Stored raw data. This chunk is never altered so that the system can be reset to the original values whenever necessary.
		self.adj_data = None 						# Storage space for denoised or altered frame values.
		
		"""
		Particle identification information
		"""
		self.particles = dict() 					# Dictionary that stores identified particles. The key is a tuple (first frame ID'd, part_ID_tag) that pairs to either
													# an instance of the particle class or another key that leads to the base particle class. The second case indicates that
													# the original particle at that key has been collapsed into a previous particle.
		self.part_in_frame = None 					# List of the number of particles in each frame
		
		"""
		State tracking for the system
		"""
		self.curr_frame = None 						# A pointer to the current frame that is being altered/tracked. Necessary as this variable assigns the first value in the particle key.
		self.data_state = 'raw' 					# Establish which data state we are looking at. Determines what output the system is giving. No alterations should be made in the raw state.

		"""
		Frame-to-frame linking information
		"""
		self.linking_map = dict() 					# A dictionary of Tuple key'd dictionaries that aggregates points in the same space so that we can better link them together.
		self.is_linking = is_linking 				# Option to have the system start linking particles
		self.frame_tolerance = frame_tolerance 		# The number of blinking frames a particle is allowed to have before being matched.
		self.frame_tolerance_counter = 0 			# The number of particles that have been assigned due to the laxness of the frame_tolerance value.
		self.particle_intensities = list()
		"""
		Signal processing storage
		"""
		self.mask = None							# n x n array defining the _universal_ mask for the data set. Could be extrapolated to an n x n x i stack if multiple images are analyzed.

		"""
		Begin debugging information.
		"""
		self.startLogger()
		"""
		Load a previous instance from JSON
		"""
		if from_json is not None:
			self.fromJSON(from_json)

		elif eager:
			"""
			Start a fresh instance using new data.
			"""
			self.setData(data_dir)
			self.setFrame(frame_num = frame_num)
			self.part_in_frame = np.zeros((self.getDimensions()[0])) - 1 # Initialized at -1 so we can differentiate frames that have 0 particles vs. those that haven't been initiliazed.

###########################################################################################
################################ Getter Methods  ##########################################
###########################################################################################

	def getData(self):
		"""
		This method returns the image stack as assigned by setData(). It assumes that a stack of images has been handed over.
		"""
		if self.data is None:
			return 'getData: Experiment has no data file'
		else:
			return self.data
	
	def getAdjustedData(self):
		"""
		This method returns the adjusted image stack. This prevents crossover between the original images and those that have been altered.
		"""
		if self.adj_data is None:
			return 'getAdjustedData: Experiment has no adjusted data'
		else:
			return self.adj_data

	def getDimensions(self):
		"""
		Returns the dimensions of the data stack.
		"""
		if self.data is None:
			return 'getADimensions: No data has been loaded'
		else:
			return self.data.shape

	def getFrame(self, frame_num = None):
		"""
		getFrame returns the requested frame and sets the system state such that it is pointing
		at the requested frame. Note that this means that the system frame pointer switches to
		whatever frame num is given.

		Inputs:
		frame_num 	The number of the frame in the image stack that the user is requesting.

		Outpus:
		curr_frame 	Returns the image as describedd by the curr_frame_num or the user given frame_num
		"""

		# Evaluates the current frame number and assigns it if the variable is defined or creates it if it is not.
		try:
			if self.curr_frame_num is None:
				self.curr_frame_num = 0
			elif frame_num is not None:
				self.curr_frame_num = frame_num
		except AttributeError:
			self.curr_frame_num = 0

		# Debugging statement to check what frame is being selected and what statck it is being selected from.
		if self.verbose and self.get_frame_verbose:
			self.logger.info('getFrame: Selecting',  self.curr_frame_num, 'from ' + self.data_state)
		
		# Chooses what image set to select from depending on the system state.	
		if self.data_state == 'raw':
			self.curr_frame = self.data[self.curr_frame_num]
		elif self.data_state == 'adjusted' and self.adj_data is not None:
			self.curr_frame = self.adj_data[self.curr_frame_num]
		else:
			self.logger.error('getFrame: There is no current frame selected')
			return 0

		return self.curr_frame
		

	def getFrameNum(self):
		"""
		Returns the current frame number that the system is pointing to.

		Inputs:
		None

		Outputs:
		The current frame number (if assigned)

		# TODO turn the first return statement into an error message.
		"""
		if self.curr_frame_num is None:
			self.logger.error('getFrameNum: No frame has been selected')
		else:
			return self.curr_frame_num

	def getParticle(self, particle_num):
		"""
		Get the particle matched to the key particle_num

		Inputs:
		particle_num	The particle key for the user requested particle.

		Outpus:
		An instance of the particle class.
		"""

		# Raises an error message if there is no matching instance of particle.
		if particle_num not in self.particles.keys():
			self.logger.error('getParticle: The key '+ str(particle_num) + ' has no matching particle entry.')

		 # If particle num is just a singular, integer value, then want a particle existing in the current frame.
		if type(particle_num) is int:
			if (self.curr_frame_num, particle_num) in self.particles:
				particle = self.particles[(self.curr_frame_num, particle_num)]

		# If there is more than one integer, we assume it is a properly formed tuple key.
		elif type(particle_num) is not int: 
			if particle_num in self.particles:
				particle = self.particles[particle_num]

		# If we collapse a particle, we mark it's entry in the dictionary with a key to the original particle and recurse until we get an instance of particle.
		if type(particle) is tuple:
			particle = self.getParticle(particle)

		return particle

	def getParticleWindow(self, particle_num):
		"""
		This method grabs the particle window as defined by the particle center and the sigma localizaiton
		and returns the bounds so it can be graphed.

		Input:
		particle_num	tuple key that identifies the desired particle.
		
		Output:
		part_window 	Listing of the coordinates that define a window around the particle based on the localization sigma.
		"""
		if type(particle_num) is int:
			part_key = (self.curr_frame_num, particle_num)
		else:
			part_key = particle_num

		this_part = self.getParticle(part_key)
		part_window = this_part.getWindowDims(part_key[0])
		return part_window
	
	def getRealParticles(self):
		"""
		This method iterates through the particle dictionary and counts the number of real particle objects.

		Input:
		None

		Output:
		active_particles	List of all keys in the particle dictionary that return a particle instance.

		TODO: Consider if this particle list should be stored or if it needs to be rerun every time. Should be linear currently.
		"""	
		count = 0
		active_particles = list()
		for i in self.particles:
			if type(self.particles[i]) is not tuple:
				count += 1
				active_particles.append(i)
		return active_particles

	def nextFrame(self):
		"""
		This is an iterative extension of getFrame that is utilized when the system is iterating frame by frame. Rather than
		pass back another image, it instead checks if another frame exists and returns a boolean if that is the case.

		TODO: In hindsight, this method may not need to exist or could instead be achieved by checking if curr_frame_num is
		greater than the last dimension of the data stack. This method is used enough that it should be removed.
		"""
		self.curr_frame_num += 1

		# Debugging flag.
		if self.verbose:
			self.logger.info("nextFrame: Incrementing to frame number " + str(self.curr_frame_num))
		# Try to grab a frame. If you get an error, that means no frame exists.
		try:
			self.getFrame()
			return True
		except:
			self.logger.info('nextFrame: There are no more frames to pass forward!')
			return False

	def getParticleIntensities(self):
		"""
		Return the intensity listing of particles to the user.
		# TODO need to make this a per frame method.
		
		Inputs:
		None

		Outputs:
		Intensity listing in the system
		"""
		return self.particle_intensities

###########################################################################################
################################ Setter Methods  ##########################################
###########################################################################################
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
		self.logger = logging.getLogger('troikaExperiment')
		self.logger.setLevel(logging.DEBUG)

		ch = logging.StreamHandler()
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		ch.setFormatter(formatter)
		self.logger.addHandler(ch)
				# Stops the info from going to root. Will want to make this an option at some point.
		self.logger.propagate = False
		
	def setData(self, data_dir = None):
		"""
		This setter method takes the data directory and instantiates a copy of the data as well as an adjusted version of the data.

		Inputs:
		data_dir	Directory URL for where the image data is stored.

		Outpus:
		None
		"""

		# This assumes that you are resetting the system based on the data dir that was handed when troikaExperiment was initialized.
		if data_dir is None:
			self.data = io.imread(self.data_dir) 
			self.adj_data = io.imread(self.data_dir)
			self.curr_frame_num = 0

		# This assumes that you have been handed a _new_ directory that points to a new data set.
		else:
			self.data = io.imread(data_dir)
			self.adj_data = io.imread(self.data_dir)
			self.data_dir = data_dir
			self.curr_frame_num = 0

	def setSavePath(self, save_dir = None):
		"""
		This setter method creates a directory to make a folder where all data will be saved

		Inputs:
		save_dir	Directory where the JSON files that form the troika experiment should be saved.
		"""

		# If the user gives a directory, use that.
		if save_dir is not None:
			self.save_dir = save_dir

		# If not, try to create a unique one using the date and the name.
		else:
			self.save_dir = './'+time.strftime("%Y_%m_%d")+'_'+self.name+'/'

		# Check if the directory actually exists in the file system or if you need to create a new one.	
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)
			if self.verbose:
				self.logger.info('setSavePath: New save directory has been created at ' + self.save_dir)

		elif self.verbose:
			self.logger.info('setSavePath: Save directory has been set to '+self.save_dir)


	def setVerbose(self, verbose = None, get_frame_verbose = None, particle_collapse_verbose = None, map_particle_verbose = None):
		"""
		Switch on global verbosity for the system as well as set other verbosity flags.

		Inputs:
		verbose 			change the state of global verbosity
		get_frame_verbose	change the state of the getFrame verbosity
	
		Outputs:
		None
		"""

		if verbose is None and get_frame_verbose is None: 					# If not output are given, simply flip the grand verbose switch
			self.verbose = not (self.verbose)

		if verbose is not None:												# Update the global verbosity flag
			self.verbose = verbose

		if get_frame_verbose is not None:									# Update the getFrame verbosity flag
			self.get_frame_verbose = get_frame_verbose

		if map_particle_verbose is not None:								# Update map particle verbosity flag
			self.map_particle_verbose = map_particle_verbose

		if particle_collapse_verbose is not None:							# Update particleCollapse verbosity flag
			self.particle_collapse_verbose = particle_collapse_verbose

	def setFrame(self, frame_num = 0):
		"""
		This setter resets the frame at a current position if no data directory has been
		given. It is a 'single frame' method, available for a user who wishes to do processing
		on a single image.

		Inputs:
		frame_num 	the number of frame the user wants the system to be set to.

		TODO: This method is only used once in the initializer. It might be possible to collapse it into getFrame.
		"""
		if self.data is not None:
			self.curr_frame = self.data[frame_num]
			self.curr_frame_num = frame_num
		else:
			self.logger.debug('setFrame: There is no data to pull a frame from.')

	def setDataState(self, state = None):
		"""
		Conditional switch to alter whether the system is pulling from the raw data stack or pulling from the adjusted data stack.

		Inputs:
		state 	Switch to the user specified state. Otherwise perform a switch flip.
		"""
		if state is not None:
			self.data_state = state
		elif self.data_state == 'raw':
			self.data_state = 'adjusted'
		else:
			self.data_state = 'raw'

	def setAdjustedFrame(self, frame, frame_pointer):
		"""
		This method inserts a new frame of data into the adjusted data set. Rather, this
		method updates the adjusted data stack after some form of image processing has been performed.

		A common error for this method is that signal processing methods pass in
		arrays that are real valued between 0 and 1. However, adj_data is a stack
		of integer arrays. This method assumes that the frame that is being passed
		in has been rescaled to integer values.

		Inputs:
		frame 			the actual data array that the user is pulling.
		frame_pointer	a pointer to the image being pulled.

		Outputs:
		None
		"""
		if self.verbose:
			self.logger.info('setAdjustedFrame: Setting new frame at ' + str(frame_pointer))
		
		self.adj_data[frame_pointer,:,:] = frame

	def setMask(self, mask):
		"""
		This method stores the universal mask used for the whole data set. This assumes
		that one mask is good enough for everything.

		Inputs:
		mask 	the n x n array of ones and zeros that is the image mask

		Outputs:
		None
		"""
		if self.verbose:
			self.logger.info('setMask: Setting new mask.')
		self.mask = mask

	def setLinking(self, is_linking = None):
		"""
		This method alters the Boolean for initiating the linking protocols. If no boolean is given
		the system simply treats this as a switch flip.

		Inputs:
		is_linking 	new Boolean for the linking map.
		"""
		if is_linking is None:
			self.is_linking = not (self.is_linking)
		else:
			self.is_linking = is_linking

	def setAdjustedParameters(self, mask = False, histogram_normalization = False, SNRBooster = False):
		"""
		TODO: Do something with this...
		"""
		pass

	def resetData(self):
		"""
		This method resets the adjusted data stack back to the original image files.

		Inputs:
		None

		Outputs:
		None
		"""
		self.adj_data = io.imread(self.data_dir)

	def addNewParticle(self, particle_pos = None, particle_sigma = None, j_dict = None):
		"""
		This method adds a new particle to the particle listing and increments the counts of particle in that frame

		Inputs:
		particle_pos - Tuple of raw particle coordinates
		particle_sigma - Sigma value dictating the "width" of the spot
		j_dict - Dictionary from a reconstituted particle
		
		Outputs:
		None
		"""
		# Reconstituting a particle from a previous experiment set
		if j_dict is not None:
			part = particle(from_json = j_dict)
			key = tuple(part.getKey().astype(int)) # This is a lot of casting to stop an unhashable type warning

		# Building the particle for the first time from a localized position.
		else:
			key = (self.curr_frame_num, int(self.part_in_frame[self.curr_frame_num]))
			part = particle(x_pos = particle_pos[0], y_pos = particle_pos[1], sigma = particle_sigma, start = self.curr_frame_num, dict_key = key)
			self.part_in_frame[self.curr_frame_num] += 1		
		self.particles[key] = part

		# We are entering into this table while linking. Once here, add the key to the particle to the linking map.
		if self.is_linking:
			map_key = (particle_pos[0], particle_pos[1], self.curr_frame_num) # Three part key: Current frame number is included for pruning purposes.
			self.linking_map[map_key] = key
			
			"""
			# Add particles to linking map for mapping later
			if map_key in self.linking_map:
				self.linking_map[map_key].append(key)
			else:
				self.linking_map[map_key] = [key]
			"""

###########################################################################################
########################### Boolean and Check Methods #####################################
###########################################################################################

###########################################################################################
############################# Static Frame Methods ########################################
###########################################################################################

	def identifyParticles(self, method = 'blobs_log', method_opts = None, adj_frame = None, make_particles = True, is_linking = None, intensity_quartile = None, intensity_cutoff = None):
		"""
		Use some programmatic method for identifying particles on an intensity map
		
		Inputs:
		method_opts - Dictionary of options that dictate how the particle identification algorithm will work.
		adj_frame - States if the system should be using some form of frame adjustment
		make_particles - Build particle objects as they are identified.
		is_linking - Boolean stating if the system should eagerly begin making traces.
		intensity_quartile - Optional cutoff to take the top n brightest pixels
		intensity_cutoff - User specified cutoff line for intensity.

		Output:
		out_blobs - Listing of rough particle positions as well as a sigma radius for that particle.
		"""

		out_blobs = list()
		# Set the frame number to zero as we want to A. Denote that the method has been called and B. Make sure we start over if the method is called twice.
		self.part_in_frame[self.getFrameNum()] = 0 
		
		# Create a list for the intensity of each particle we find in frame
		intensity_list = list()

		"""
		Perform any frame adjustments. Feel free to add more options here.
		"""
		if adj_frame is not None:
			self.setDataState('adjusted')
		if adj_frame == 'morphologically':
			self.morphologically()
		# TODO: Enable other frame adjustments before particle identification.
		if adj_frame == 'histNorm':
			self.histogramNormalization()

		"""
		This first block of code goes through the effort of finding particles and returning the coordinates and sigma
		"""
		if method == 'blobs_log':

			
			if method_opts is None:
				method_opts = {'min_sigma':1.2, 'max_sigma':1.5, 'threshold':.02, 'num_sigma':10} 	# If not method options are given, these are our default options for blobs_log			
			id_blobs = blob_log(self.getFrame(), **method_opts)

		# TODO: Add other particle identification methods
		else:
			self.logger.error('No valid method selected')

		if self.verbose:																			# If requested, report how many particles have been found.
			self.logger.info("identifyParticles: There have been ", len(id_blobs), " particles identified.")


		"""
		Look at intensity values for preprocessing in the frame. This assumes that all intensity values should be scaled based on the readout of the current frame.
		"""
		for part in id_blobs:
			intensity = self.getFrame()[int(part[0]),int(part[1])]
			intensity_list.append(intensity)

		if intensity_quartile is not None:
			intensity_cutoff = np.percentile(intensity_list, intensity_quartile)
		elif intensity_cutoff is None:
			intensity_cutoff = 0

		"""
		This block calls the particle generation method to create particles from the particle class. At this point, all particle identification has concluded.
		We can expand this area by adding conditionals to decide if a particle should be added.
		"""
		if make_particles is True:
			ind = 0
			for part in id_blobs:
				if intensity_list[ind] > intensity_cutoff:
					self.addNewParticle([part[0], part[1]], part[2])
					out_blobs.append(part)
				else:
					pass
				ind += 1

		# Option for people to change is_linking when calling identifyParticles
		if is_linking is not None:
			self.is_linking = is_linking

		# Proc a call to link particles together eagerly. Can be delayed until later if desired.
		if self.is_linking and self.getFrameNum() != 0 and self.part_in_frame[self.getFrameNum()-1] != -1: 
			self.mapParticles() 																	# Just call map particles and handle the heuristics later.

		# Add the intensity of all particles form this frame to the master list
		self.particle_intensities.append(intensity_list)
		return out_blobs

	def superResolve(self, local_thd = True, gauss_width = 3, wide2 = 2, num_std = 3, instrumental_response = 0.204):
		"""
		This is an updated Super Resolution that operates on a particle by particle basis rather than scanning through each image. It pre-supposes that particles have been identified rather
		than try and identify the particles at the same time that they are super-resolved.\

		# The magic number 0.204 is the average value per pixel averaged by 1000 trials.
		"""
		if self.part_in_frame[self.curr_frame_num] == -1:
			self.identifyParticles()

		if self.part_in_frame[self.curr_frame_num] == 0:
			self.logger.info('superResolve: There are no particles to localize in this frame.')
			return 0
		match_r = 2 * gauss_width 			# The size of the fitting region is match_r*2+1
		sig_thd = instrumental_response*(2*match_r+1)*0.9 	# Threshold based on the width of the noise. Width of a real particle must be smaller than 90% of the width of the noise.
											
		rejection_count = 0

		for i in range(int(self.part_in_frame[self.curr_frame_num])):
			key = (self.curr_frame_num, i)
			part_window = self.getParticleWindow(key)
			im = self.getFrame(frame_num = key[0])[part_window[0]:part_window[1],part_window[2]:part_window[3]]
			xc, yc, sigma = self.radialCenter(im) # Need to create an optional way to switch the way we perform super-resolution


			if xc is False and self.verbose:
				self.logger.info('Particle at key ', str((self.curr_frame_num, i)), ' did not localize correctly.')
			# Check if the localization is good enough

			if sigma < sig_thd:
				if True or self.verbose:
					how_much_off = sigma - sig_thd
					self.logger.debug('superResolve: Particle ' + str(i)+ ' rejected. Outside by ', str(how_much_off))
					self.particles[key].markRejection(self.curr_frame_num)
					self.particles[key].setSuperResCoords(part_window[0] + xc, part_window[2]+yc, sigma, self.curr_frame_num)
					rejection_count += 1
			else:
				self.particles[key].setSuperResCoords(part_window[0] + xc, part_window[2]+yc, sigma, self.curr_frame_num)
		self.logger.info('superResolve: ',str(rejection_count), ' of ', str(self.part_in_frame[self.curr_frame_num]), ' particles rejected.')


	def old_findSuperResolutionCenters(self, blob_log_opts = None, local_thd = True, gauss_width = 3, wide2 = 2, num_std = 3):
		"""

		WENXIAO'S/BO'S old code
		This is the original way Troika identified particle centers in an image.
		Identifies particles using pre-processed image data

		Input: 
		im - input (single) frame to localize particles within
		blob_log_opts - dictionary of options for blob_log. See skimage.feature.blob_log for details
		local_thd - determines if local threshholding is to be used.
		gauss_width - estimated Gaussian standard deviation. Used to determine the size of the fitting region
		wide2 - the width threshold
		num_std - how many standard deviations to add up as a threshold

		Output:
		params - Nx3 matrix of [x, y, Gwidth] row vectors for identified particles

		TODO:
		Need to refactor this function to be listed as super-resolve particles and handle particle identification elsewhere.
		"""

		im = self.getFrame()
		
		"""
		For now, I am going to utlize the LoG method of finding particles as it has worked for the data previously. I can implement our other items at a later date.
		"""
		raw_particles = self.identifyParticles(**blog_log_opts) 	# n x 3 matrix where n is the number of particles, and the three entries are rough particle coordinates
		h, v = im.shape
		n_part = raw_particles.shape[0] 							# extracting number of particles

		match_r = 1 * gauss_width 									# The size of the fitting region is match_r*2+1
		final_particles = zeros(n_part,3) 							# Matrix that will hold all the particles we identify; replaces 'params' in original Matlab code
		sig_thd = 0.204*(2*match_r+1)*0.9 							# Threshold based on the width of the noise. Width of a real particle must be smaller than 90% of the width of the noise.
																	# The magic number 0.204 is the average value per pixel averaged by 1000 trials.

		"""
		Check boundary conditions for the region of interest.
		"""
		for i in range(len(n_part)):
			row1 = np.max(q[0]-match_r, 0)
			row2 = np.min(q[0]+match_r, h)
			col1 = np.max(q[1]-match_r, 0)
			col2 = np.min(q[1]+match_r, v)

			"""
			Call the localization function
			"""
			xc, yc, sigma = self.radialCenter(im[row1:row2, col1:col2]) # This is the default method. We will want to change this in the future.

			if xc is False and self.verbose:
				self.logger.error('Particle at key ', str((self.curr_frame_num, i)), ' did not localize correctly.')

			"""
			Check if the localization is good enough
			"""
			if sigma < sig_thd:
				final_particles[i,:] = (xc+row1-1, yc+clm-1, sigma)
				if self.verbose:
					how_much_off = sigma - sig_thd
					self.logger.info('superResolve: Particle ',str(i),' rejected. Outside of threshold by ', str(how_much_off))

###########################################################################################
############################# Super Resolution Methods ####################################
###########################################################################################

	def radialCenter(self, im, smooth_flag = True):
		"""
		Calculate the center of a 2D intensity distribution using a radial symmetry function

		Inputs:
		I - 2D gray scale intensity map

		Outputs:
		xc, yc - Radial center coordinates
		sigma  - Rough measure of distribution width (sqrt of the second moment of I-min(I))

		Adapted from Raghuveer Parthasarathy's original matlab code
		Copyright 2011-2012, Raghuveer Parthasarathy, The University of Oregon 
		"""

		"""
		This block creates a matrix of "mid-point" coordinates
		"""
		nx, ny = im.shape # Assuming that I is a 2 dimensional image

		if nx == 0 or ny == 0 or ny != nx:
			self.logger.error('radialCenter: Something did not work correctly. nx = ', str(nx), ' ny = ', str(ny))
			return False, False, False

		xm_onerow = np.linspace(-(nx - 1)/2+0.5, (nx-1)/2-0.5, nx-1)
		xm = np.tile(xm_onerow, (ny-1, 1))
		if ny == nx:
			ym = (np.ones((ny-1, ny-1)) * xm_onerow).T
		else:
			ym_onerow = np.linspace(-(ny - 1)/2+0.5, (ny-1)/2-0.5, ny-1)
			ym = (np.ones((ny-1, ny-1)) * ym_onerow).T

		"""
		Calculating the derivatives along the 45-degree shifted coordinates u and v.
		"""
		dimdu = im[0:ny-1, 1:nx] - im[1:ny, 0:nx-1] # The way this translates could be a problem I have a feeling that the indices need to be switched around.
		dimdv = im[0:ny-1, 0:nx-1] - im[1:ny, 1:nx] # 

		"""
		Optional smoothing of the region via 2D convolution with a 3x3 of ones
		"""
		if smooth_flag:
			h = np.ones((3,3))/9
			fdu = conv2(dimdu, h, mode = 'same')
			fdv = conv2(dimdv, h, mode = 'same')
			dImag2 = np.multiply(fdu, fdu) + np.multiply(fdv, fdv)
	
		"""
		Calculation of gradient slope; Note that a 45 degree rotation of u,v components is needed to express the slope in the x-y coordinate system.
		This negative sign "flips" the array to account for y increasing "downward"
		"""
		try:
			m = np.divide(-(fdv+fdu), (fdu-fdv))
		except:
			"""
			Testing for non-real results
			"""
			NaN_array = np.isnan(m)

			if np.sum(np.sum(NaN_array)) > 0: # If the denominator is 0, try again w/o smoothing
				unsmoothm = np.divide((dimdv + dimdu), (fdu-fdv))
				m[NaN_array] = unsmoothm[NaN_array]
			
			NaN_array = np.isnan(m)
			n_NaN_in_m = np.sum(np.sum(NaN_array))

			if n_NaN_in_m > 0: # If m still contains NaNs, replace them with 0s
				m[NaN_array] = 0


		# TODO: Need to add some catches for infinite values. The old code handles these but in a very inelegant way.

		"""
		Calculating the "y-intercept" which should be straight forward
		"""
		b = ym - np.multiply(m, xm)

		"""
		Now we want to weight all the values by the square of the gradient magnitude and inverse distance to gradient intensity centroid
		"""
		sdI2 = np.sum(dImag2[:])
		xcentroid = np.divide(np.sum(np.sum(np.multiply(dImag2, xm))), sdI2)
		ycentroid = np.divide(np.sum(np.sum(np.multiply(dImag2, ym))), sdI2)
		w_div = np.sqrt(np.multiply((xm-xcentroid),(xm-xcentroid)) + np.multiply((ym-ycentroid),(ym-ycentroid)))
		w = np.divide(dImag2, w_div)

		"""
		Calculate the sub-pixel coordinates using the radial center fit
		"""
		xc, yc = lsradialcenterfit(m, b, w)

		"""
		Adjust outputs to be relative to the upper left coordinate
		"""
		xc = xc + (nx+1)/2 # Possibly need to check the value because we are in a 0 indexed system versus a 1 indexed system.
		yc = yc + (ny+1)/2

		"""
		Calculating the sigma values
		TODO: Need to test how well this segment works.
		"""

		Isub = im - np.min(np.min(im))
		px, py = np.meshgrid(np.linspace(1,nx,nx), np.linspace(1,ny,ny))
		xoffset = px - xc
		yoffset = py - yc

		r2 = np.multiply(xoffset, xoffset) + np.multiply(yoffset, yoffset)
		sigma = np.sqrt(np.sum(np.sum(np.multiply(Isub, r2)))/np.sum(np.sum(Isub)))/2

		return xc, yc, sigma

###########################################################################################
############################# Particle Tracking Methods ###################################
###########################################################################################

	def mapParticles(self, frame_tolerance = None, distances = 5):
		"""
		This is the master map particles method. 

		Inputs: 
		frame_tolerance - User specified frame tolerance
		distances - The maximum distance allowed for linking a particle.
		"""

		if self.verbose and self.map_particle_verbose:
			self.logger.info('mapParticles: Particle mapping has begun.')
		if self.track_time:
			start = time.time()
		if frame_tolerance is not None:
			self.frame_tolerance = frame_tolerance

		"""
		Set variables for analysis purposes.
		"""		
		tracking_count = 0																								# Number of particles that have been added to a trajectory on this go around.
		distances = np.arange(distances+1)																				# The set of valid pixel distances that we are going to consider.

		"""
		Begin radial tracking for each point. We want to iterate through radius first, then point list.
		"""
		for dist in distances:
			for i in list(self.linking_map.keys()): 																			# Iterate over all active keys in the linking_map
				
				particle_collapsed = False 																				# Marks that our current particle has been mapped

				
				if self.verbose and self.map_particle_verbose:															# State the particle that is being mapped
					self.logger.debug('mapParticles: map key: '+i)
				
				if i[2] != self.curr_frame_num:																			# Check if the particle we are mapping exists in the current frame.
					if self.verbose and self.map_particle_verbose:
						self.logger.debug('mapParticles: key '+i+' skipped as it is not the current frame')
					continue
				
				if dist == 0:																							# Set the candidate particle list depending on our search radius; Could likely collapse this for conciseness
					particle_matching_list = self.createSameCoordList((i[0],i[1]))
				else:
					particle_matching_list = self.createSurroundingCoordList((i[0],i[1]), pixel_radius = dist)
				
				for j in particle_matching_list:

					if self.verbose and self.map_particle_verbose and len(particle_matching_list) > 0:					# Report what my possible candidate coordinates are.
						self.logger.info('mapParticles: Candidate particle keys for '+i+' is '+particle_matching_list)
					
					frame_diff = self.checkFrameCompatibility(self.linking_map[i], self.linking_map[j], frame_tolerance) # Check frame compatibility and return correct values for particle collapse
					
					if self.verbose and self.map_particle_verbose:														# Report what the frame_diff looks like
						self.logger.debug('mapParticles, frame_diff: '+frame_diff)
					
					if type(frame_diff) is not tuple:																	# Frame diff is False, iterate to the next particle
						continue
					else:																								# Frame diff returned a tuple so we can call particle collapse.
						if self.verbose and self.map_particle_verbose and dist == 0:
							self.logger.debug('mapParticles: Same coords, key '+j)
						particle_collapsed = self.particleCollapse(*frame_diff, l_map_key_1 = i, l_map_key_2 = j, distance_order = dist)
						tracking_count += 1

					if particle_collapsed:
						break
			"""
			Prune linking_map of all dead keys after each radial run *should be ancillary as particle collapse annhilates entries.
			"""
			self.pruneLinkingMap()
		"""
		Final print statement mentioning how long it took to process through the frame as well as list the number of particles succesfully linked.
		"""
		if self.verbose or self.track_time:
			end = time.time() 
			self.logger.info('mapParticles: Frame '+str(self.curr_frame_num)+' of '+str(self.getDimensions()[0])+' mapped in '+str(np.round(end-start,2))+' seconds. '+str(tracking_count)+' of '+str(int(self.part_in_frame[self.curr_frame_num]))+' particles mapped.')
	
	def particleCollapse(self, p_key1, p_key2, l_map_key_1 = None, l_map_key_2 = None, distance_order = None):
		"""
		This method will be used to collapse particles together to generate trajectories. This method returns all the necessary information to update another particle and assumes that all conditions for particle adsorption have been met.

		Inputs:
		p_key1			Key to the first particle to be collapsed
		p_key2			Key to the second particle that is to be host
		l_map_key_1 	Linking map key of the first particle
		l_map_key_2		Linking map key of the second particle
		distance_order	How far away the two particle centers are; Utilized for error checking primarily.

		Outputs:
		None
		"""

		if self.verbose and self.particle_collapse_verbose:
			self.logger.debug('particleCollapse: Collapsing '+str(p_key2)+' into '+str(p_key1)+'. Distance order '+str(distance_order))

		"""
		Absorb the sacrificial particle into the host and clean up the particle list and linking map.
		"""
		self.getParticle(p_key1).absorb(self.getParticle(p_key2).collapse())						# Combine particles
		self.particles[p_key2] = p_key1																# Remove redundant particle from dictionary. Place key to root particle in the position
		self.linking_map[l_map_key_1] = p_key1														# Remove redundant particle from linking_map and replace it with a key to particle 1

		if self.verbose and self.particle_collapse_verbose:											# Self pruning for linked particles
			self.logger.debug('particleCollapse: Destroying '+l_map_key_2)
		del self.linking_map[l_map_key_2]

		if self.verbose and self.particle_collapse_verbose:											# Announce that the particle absorbtion was a success.
			self.logger.debug("particleCollapse: Particle "+str(p_key1)+" absorbed "+str(p_key2)+"!")

	def checkFrameCompatibility(self, p_key1, p_key2, frame_tol = 1):
		"""
		Helper method that takes two particle keys and sees if they are within frame_tol of each other
		Inputs:
		p_key1 - key for the first particle
		p_key2 - key for the second particle
		frame_tol - number of frames of separation allowed for the particles

		Outputs:
		key_tuple - False if the two particle keys are not compatible. If true, returns a tuple with the key of
					structure (collapse receiver, collapse target).

		TODO: Make this method check if there are frame conflicts e.g. two particle trails are share common frames.
		"""
		
		# Check if the two particles we are checking exist in the same frame
		if p_key1[0] == p_key2[0]:
			if self.verbose and self.checkFrameCompatibility_verbose:
				self.logger.debug('checkFrameCompatibility: particles '+str(p_key1)+' and '+str(p_key2)+' occur in the same frame.')
			return False

		elif p_key1 not in self.particles.keys():
			if self.verbose and self.checkFrameCompatibility_verbose:
				self.logger.debug('checkFrameCompatibility: particle '+str(p_key1)+'does not exist.')
			return False

		elif p_key2 not in self.particles.keys():
			if self.verbose and self.checkFrameCompatibility_verbose:
				self.logger.debug('checkFrameCompatibility: particle '+str(p_key2)+'does not exist.')
			return False

		# Check if the two particles have any common frames in their past. If so, do not combine
		if len([value for value in self.getParticle(p_key2).getFrameSet() if value in self.getParticle(p_key1).getFrameSet()]) > 0:
			if self.verbose and self.checkFrameCompatibility_verbose:
				self.logger.debug('checkFrameCompatibility: particles '+str(p_key1)+' and '+str(p_key2)+' have overlapping frames.')
			return False
		
		frame_diff = self.getParticle(p_key1).getLastFrame() - self.getParticle(p_key2).getLastFrame()
		
		if np.abs(frame_diff) <= self.frame_tolerance:
			if frame_diff > 0: # Particles are in correct order in time
				return (p_key2, p_key1)
			else: # Particles are out of order in time
				return (p_key1, p_key2)
		else:
			if self.verbose and self.checkFrameCompatibility_verbose:
				self.logger.debug('checkFrameCompatibility: particles different fell outside of frame compatibility. Frame_diff: '+frame_diff)
			return False

	def createSurroundingCoordList(self, pixel_center, pixel_radius):
		"""
		Helper method that generates a listing of tuples for all pixels at a given radius away with closest pixel coordinates
		listed first, followed by far corner pixels.

		Inputs:
		pixel_center - the center pixel that the radius is built around
		pixel_radius - the radius around the particle center

		Outputs:
		A listing of the valid keys in the linking_map that fit the distance/frame tolerance parameters.
		"""

		"""
		Define the corners of the box that is surrounding pixels. As of now, this makes a misrepresented circle as we don't trim the corners. 
		TODO: Normalize to a greater extent.
		"""
		border_coords = list()
		min_x = int(pixel_center[0]) - int(pixel_radius)
		max_x = int(pixel_center[0]) + int(pixel_radius)
		min_y = int(pixel_center[1]) - int(pixel_radius)
		max_y = int(pixel_center[1]) + int(pixel_radius)

		"""
		Horizontal components
		"""
		for j in range(self.curr_frame_num - self.frame_tolerance, self.curr_frame_num):
			for i in range(min_x, max_x+1):
				border_coords.append((i, max_y, j))
				border_coords.append((i, min_y, j))
			"""
			Verticle components
			"""
			for i in range(min_y+1, max_y):
				border_coords.append((min_x, i, j))
				border_coords.append((max_x, i, j))

		
		border_coords = list(dict.fromkeys(border_coords))	# The wrappers at the end help to remove any duplicate coordinates present.		
		
		"""
		This final list comprehension returns the intersection of the surrounding values
		with the entries in linking list.
		"""
		return [x for x in border_coords if x in self.linking_map.keys()]

	def createSameCoordList(self, pixel_center):
		"""
		Helper method that creates a list of valid linking map keys for particles occuring in the same position.

		Inputs:
		pixel_center - the center coordinate of the pixel in question

		Outputs:
		same_coords - list of valid linking map keys for stationary particles
		"""
		createSameCoordList_verbose = False
		same_coords = list()
		for i in range(self.curr_frame_num-self.frame_tolerance, self.curr_frame_num):
			same_coords.append((pixel_center[0], pixel_center[1], i))

		same_coords = [x for x in same_coords if x in self.linking_map.keys()]

		if self.verbose and createSameCoordList_verbose:
			self.logger.debug('createSameCoordList: pixel_center ', pixel_center, ' has the following coordinates ', same_coords)
		return [x for x in same_coords if x in self.linking_map.keys()]

	def getClosestParticle(self, local_keys = None):
		"""
		This method is intended to take a series of linking_map keys that surround a particle of interest in space.
		It then seeks out nearby particles in the radius and finds the closest particle. It is not intuitively
		useful until we get super-resolution working.

		Inputs;
		local_keys - listing of the linking_map keys containing particles as identified by super-res methods.

		Outputs:
		p_key - key of the closest particle.
		"""
		pass

	def pruneLinkingMap(self):
		"""
		Our pruning is a forward looking method. i.e. you cannot rewind the system to restart particle linking
		as the data has already been discarded. We do this as it reduces the bloat in the linking map which 
		subsequently reduces the particle mapping time. Further, it is structured to remove particles that have
		not been linked. Particle references that have been linked are removed during particle collapse.

		Inputs:
		None

		Outputs:
		None
		"""
		frame_cutoff = self.curr_frame_num - 1 - self.frame_tolerance

		if frame_cutoff < 0:
			return 0
		
		if self.verbose:
			self.logger.info('pruneLinkingMap: Entering in with '+len(self.linking_map)+' entries. Removing all frames at or below '+frame_cutoff)
		

		for i in list(self.linking_map.keys()):
			if i[2] <= frame_cutoff:
				del self.linking_map[i]

		if self.verbose:
			self.logger.info('pruneLinkingMap: Ending with '+len(self.linking_map)+' entries.')



###########################################################################################
############################# Signal Changing Methods #####################################
###########################################################################################

	def SNRBooster(self, frame = None):
		"""
		This method is highly inspired by Arnauld Serge's code for particle
		tracking, for more information about their method, look up: "Dynamic
		multiple-target tracing to probe spatiotemporal cartography of cell
		membranes" and their online code.
		There is a easier function in matlab: conv2, which can also perform 2-D
		convolution of two images. It is even faster. However, that function
		suffering from the boundary problem, that's why we don't use it here. Here
		we use a Fourier transform method. Petersen 1993, Eqn. 12 - Fourier method

		Inputs:
		frame - 2x2 matrix of values that represents an image

		Outputs:
		boosted_frame - convoluted image that has boosted signal
		"""
		self.setDataState(state = 'adjusted')

		if frame is None:
			frame = self.getFrame()
			frame_pointer = self.curr_frame_num
		elif type(frame) is int:
			frame = self.getFrame(frame)
			frame_pointer = frame

		v, h = frame.shape		# Dimensions of the frame
		ex_w = 3				# Dimensions of the convolutional matrix

		mask = np.zeros((v,h))
		boosted_frame = np.zeros((v,h))

		v_half = int(np.ceil(v/2))
		h_half = int(np.ceil(h/2))
		mask[v_half:(v_half+ex_w), h_half:(h_half+ex_w)] = np.ones((ex_w, ex_w))

		ft_mask = np.fft.fft2(mask)

		ft_im = np.fft.fft2(frame[:,:])

		mid_frame = np.fft.fftshift(np.fft.ifft2(np.multiply(ft_mask,ft_im)))
		boosted_frame[:,:] = np.real(mid_frame)/ex_w**2


		self.setAdjustedFrame(boosted_frame, frame_pointer)

		return boosted_frame

	def histogramNormalization(self, frame = None, from_adj = False):
		"""
		Uses the histogram normalization feature from the exposure package to raise contrast
		in the image. Note that usage of this method automatically updates the altered frame
		stack of the system.

		Inputs:
		frame - Either an n x n picture representing a frame from the data set or an integer
				value that is valid using getFrame()
		from_adj - boolean to declare if frame is being renewed from raw data or in addition
				   to other adjustments

		Outputs:
		altered_frame - the frame output from equalize_hist
		"""
		if from_adj:
			self.setDataState(state = 'adjusted')
		else:
			self.setDataState(state = 'raw')

		if frame is None:
			frame = self.getFrame()
			frame_pointer = self.curr_frame_num

		else:
			frame_pointer = self.curr_frame_num

		if self.verbose:
			self.logger.info('histogramNormalization: Beginning processing on frame '+frame_pointer)

		altered_frame = equalize_hist(frame) * np.max(frame)

		self.setAdjustedFrame(altered_frame, frame_pointer)

		return altered_frame

	def innerOuterSegmentation(self, frame = None, floor = -0.95, ceiling = 0.8, diff_param = 20, set_mask = True):
		"""
		innerOuterSegmentation does a two part segmentation using the random_walker algorithm
		from the skimage package. This segmentation assumes that only two segments are wanted,
		thereby getting an inner cell vs. an outer cell. Further, it is a wrapper for the
		segmentation algorithm. It automatically multiplies the adjusted frame by the segmentation.

		Inputs:
		frame - the specified frame to process
		floor - the base line for the adjusted intensity to be zeroed
		ceiling - the top line for the adjusted intensity to be set to one.
		diff_param - kwarg for the random_walker method
		set_mask - Create a system mask based on this inner/outer segmentation.

		Outputs:
		masked_image - Masked version of the frame based on the inner/outer segmentation.
		maske - The 1/0's mask generated by the inner/outer segmentation.
		"""
		if frame is None:
			frame = self.getFrame()
			frame_pointer = self.curr_frame_num
		elif type(frame) is int:
			frame_pointer = frame
			self.curr_frame_num = frame
			frame = self.getFrame(frame)

		"""
		Create a similar image with a rescaled intensity range. NOTE: You need to perform
		histogram normalization. Failing to do so leads to system crashing memory bloat.
		"""
		data = self.histogramNormalization()
		data = rescale_intensity(data, in_range = (0, np.max(data)), out_range = (-1, 1))
		markers = np.zeros(data.shape, dtype = np.uint)

		markers[data<floor] = 1 
		markers[data>ceiling] = 3
		mask = random_walker(data, markers, beta=diff_param, mode='bf')
		
		masked_image = (mask) * self.getFrame() #* np.max(frame)
		self.setAdjustedFrame(masked_image, frame_pointer)

		if set_mask and self.mask is None:
			self.setMask(mask-1)

		return masked_image, mask

	def nSegmentation(self, seg_dictionary = None, frame = None):
		"""
		nSegmentation segments the image based on the output of the random_walker algorithm
		from the skimage package.

		This method is meant to allow n different levels of segmentation in a frame. However, I never ended up completing it as the random walker method has
		some odd effects that I never sorted out.
		"""
		if frame is None:
			frame = self.getFrame()
		elif type(frame) is int:
			frame = self.getFrame(frame)

		# Create a similar image with a rescaled intensity range.
		data = rescale_intensity(frame, in_range = (0, np.max(frame)), out_range = (-1, 1))
		
		markers = np.zeros(data.shape, dtype = np.uint)
		markers[data<floor] = 1
		markers[data<ceiling] = 2
		pass

	def filterRegionalMaxima(self):
		"""
		The guide for this method can be found at https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_regional_maxima.html
		"""
		if self.data_state is not 'adjusted':
			self.setDataState('adjusted')

		image = img_as_float(self.getFrame())
		image = gaussian_filter(image, 1)

		seed = np.copy(image)
		seed[1:-1, 1:-1] = image.min()
		mask = image

		dilated = reconstruction(seed, mask, method = 'dilation')

		final_image = image - dilated
		self.setAdjustedFrame(final_image, self.curr_frame_num)

		return image - dilated

	def morphologically(self, operator = 'white_tophat', element = 'disk', radius = 3):
		"""
		Apply a morphological operator to clean up portions of the image. Right now the base method is meant to 
		work with a disk element and utilize a white_tophat operator.

		Inputs:
		operator - The morphological operator of choice.
		element - The shape of the kernel used to morph the existing image.
		radius - The size of the kernel being generated.

		Outputs:
		None, all changes are stored in the adjusted image stack.
		"""

		if self.data_state is not 'adjusted': # Not sure if it is just quicker to call the set state method rather than evaluate an if.
			self.setDataState('adjusted')
		
		"""
		Setting up the image for morphological processing
		"""
		image = img_as_float(self.getFrame())
		rescale = 100000 # Image values must be >1 to be stored in an integer matrix
		
		"""
		Setting up the element for the morphological operator.
		"""
		if element == 'disk':
			element = disk(radius)
		
		"""
		Perform the morphological operation that will generate the final image.
		"""
		if operator == 'white_tophat':
			filtered_image = white_tophat(image, element)
		elif operator == 'opening':
			filtered_image = opening(image, element)

		"""
		Update the adjusted frame stack and return our new image.
		"""
		self.setAdjustedFrame(filtered_image*rescale, self.curr_frame_num)
		return filtered_image*rescale

	def applyMask(self):
		"""
		Apply mask uses the system calculated mask to remove low level background noise. This method
		assumes that the mask has been calculated before being called. It could be useful to have have
		this method eagerly calculate a mask if one does not exist.
		"""
		if self.data_state is not 'adjusted':
			self.setDataState('adjusted')

		if self.mask is None and self.verbose:
			self.logger.error('applyMask: No mask has been generated.')
		else:
			self.adj_data[self.curr_frame_num] = self.getFrame()*self.mask

###########################################################################################
###################### Plotter and Visualization Methods ##################################
###########################################################################################

	def colorAssignment(self, args):
		"""
		The purpose of color assignment is to use some series of methods to assign colors to
		particles based on a set of criteria. The splitting of that criteria should be decided
		by args. This is supposed to serve as a wrapper around other color setting methods.
		"""

		# Need a switch case-esque listing for each method that can be used.

		# Based on length of trace
		"""
		Need assignment of trace color by distribution of trace length.
		User should give some set of trace lengths and colors. 
		"""		

		# Based on diffusion/motion
		"""
		Assignment of color based on RMSE. User must provide a dictionary with number/color matches.
		"""
		# Based on position
		"""
		User must provide a metric for a particle center/particle region. Consider using segmentation maps for this.
		"""
		pass

	def visualizeParticle(self, particle_key, frame_number = None):
		"""
		This method allows the user to visualize a particles local region given the sigma of the raw fitting. It assumes that the particle has already been identified and pulls from
		an instance of particle referenced from the particle dictionary using particle key.

		Inputs:
		particle_key - The key for the particle instance stored in troikaExperiments particle dictionary.
		frame_number - The fthe frame of the particle to be plotted.

		Outputs:
		None, could be extended to hand back a figure handle w/ axes.
		"""
		fig, ax = plt.subplots(figsize = (4,4))
		part_window = self.getParticleWindow(particle_key)
		
		if frame_number is None:
			frame_number = particle_key[0]
		"""
		Set Background
		"""
		im = self.getFrame(frame_num = frame_number)

		"""
		Plot scale bar
		"""
		scale_x, scale_y, text = self.plotScaleBar(sb_coords = [part_window[2]+5, part_window[1]-1], bar_dist_in = 1000, just_data = True, no_label = True) # Default to a 1 um scale bar
		plt.plot(scale_x, scale_y, color = 'white', linewidth = 5)
		ax.imshow(im)

		"""
		Isolate the window where we want to see the particle and set matplotlib to show that region.
		"""
		ax.set_xlim([part_window[3],part_window[2]])
		ax.set_ylim([part_window[1],part_window[0]])

		"""
		Plot the particle or fire off an error message if the particle does not exist in the requested frame.
		"""
		coords = self.getParticle(particle_key).getCoordPairs(frame_number)
		if coords != -1:
			c = plt.Circle((coords[1], coords[0]), 0.1, color='red', linewidth=2, fill=False)
			plt.gca().add_patch(c)
		elif self.verbose:
			self.logger.debug('visualizeParticle: No coordinates were found for particle '+str(particle_key)+' frame '+str(frame_number))

	def plotScaleBar(self, just_data = True, pix2nm = 193.8, fontsize = 20, sb_coords = [20, 240], bar_dist_in = 4000, no_label = False):
		"""
		Plot a scale bar or return the necessary info to plot a scale bar

		Inputs:
		just_data - Return only the necessary items to plot the bar. If false, actually plot the bar on the current figure.
		pix2nm - conversion factor for how many nms there are in a pixel
		fontsize - size of the font for the text object.
		sb_coords - The coordinates for the left-most point of the bar.
		bar_dist_in - The number of nms the bar is supposed to extend.
		no_label - Boolean dictating if the bar should have a label generated.

		Outputs
		x_coords - X-coordinates necessary to plot the scale bar.
		y_coords - Y-coordinates necessary to plot the scale bar.
		text_info - Real text object that acts as a label to the scale bar.

		TODO: Need to add a way to change where the scale bar should be appearing
		"""
		scale_bar_x = sb_coords[0]
		scale_bar_y = sb_coords[1]
		bar_dist = bar_dist_in/pix2nm
		x_coords = [scale_bar_x, scale_bar_x+bar_dist]
		y_coords = [scale_bar_y, scale_bar_y]
		label = str(bar_dist_in/1000.0) + 'um'
		text_info = [(scale_bar_x+bar_dist/2), scale_bar_y-5, label, fontsize]
		
		if not just_data:
			plt.plot(x_coords, y_coords, color = 'white')
			if not no_label:
				plt.text((scale_bar_x+bar_dist/2-4),scale_bar_y-3,label, color = 'white', fontsize=fontsize)

		else:
			return x_coords, y_coords, text_info

	def plotFrameNumber(self, just_data = True, fontsize = 20):
		"""
		Adds a frame counter to the bottom corner

		Inputs:
		just_data - Boolean that determines if an actual plot object is generated or if only the information to make the bar is given.
		fontsize - Fontsize of the label generated for the frame number.
		"""
		framecount_x = 200
		framecount_y = 240
		framestatement = str(self.curr_frame_num+1) + '/' + str(self.getDimensions()[0])

		text_info = [framecount_x, framecount_y, framestatement, fontsize]

		if not just_data:
			plt.text(text_info[0],text_info[1],text_info[2], color = 'white', fontsize = text_info[3])

		else:
			return text_info
	
	def plotCurrImage(self, is_GIF = False, figsize = (6,6), scalebar = True, framecount = True, dpi = 100, adjusted = False, add_particles = False):
		"""
		This method plots the current image. Adjustments is a dictionary that details what transformations need to be included/done to make the new GIF.

		Inputs:
		is_GIF - Boolean to indicate if this is being used recursively to make a GIF.
		figsize - Size of the figure being generated.
		scalebar - Boolean indicating if a scalebar should be drawn.
		framecount - Boolean indicating if the frame number should be plotted in the bottom.
		dpi - Set the dpi for the image.
		adjusted - Boolean whether plotted frames should come from the adjusted stack or the raw stack.
		add_particles - Boolean for adding localized particle centers to the image.

		Outputs:
		image - If this is supposed to be a GIF, the canvas object will be returned as an image.
		"""
		fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
		# An explantion on how to eliminate excess whitespace can be found here https://stackoverflow.com/questions/49693560/removing-the-white-border-around-an-image-when-using-matplotlib-without-saving-t
		fig.subplots_adjust(0,0,1,1)
		
		if not adjusted: # If there are no adjustments to be made, we end here
			ax.imshow(self.getFrame())

		else: # TODO: Implement adjusted image graphing
			pass

		ax.set_xticks([])
		ax.set_yticks([])
		
		# Formatting ifs
		if scalebar:
			self.plotScaleBar(just_data = False)
		
		if framecount:
			self.plotFrameNumber(just_data = False)

		# Decorator ifs
		if add_particles:
			self.plotIDParticles()

		# Output ifs
		if is_GIF: #This is an important bit of code that saves the original chart as an image. All modifications/additions must be done before this call.
			fig.canvas.draw()
			image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
			image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
			return image
		
		if not is_GIF:
			ax.imshow(self.curr_frame)

	def plotIDParticles(self, args = None):
		"""
		Make a call to identify particles in the current frame then overlay these particles on an image of the cell.
		"""
		parts = self.identifyParticles(**args)
		for i in parts:
			y,x,r = i
			c = plt.Circle((x, y), 0.5, color='red', linewidth=2, fill=False)
			plt.gca().add_patch(c)

	def simpleGIFofRawData(self, gif_dir, start = 0):
		"""
		This is meant to be a largely extensible method that can make .gifs for use in PowerPoint presentations. Currently it is a quick and dirty method.

		Inputs:
		gif_dir - The directory where the gif needs to be saved.
		start - The frame that should be the start of the gif

		Outputs:
		None
		"""
		is_next_frame = True # Is next frame is a boolean that can get changed inside of the preceding while loop. It is initialized as True so the loop can run once at least.
		self.getFrame(0)
		with imageio.get_writer(gif_dir, mode='I') as writer:
			while is_next_frame:
				writer.append_data(self.curr_frame)
				is_next_frame = self.nextFrame()

	def makeFancyGIF(self, gif_dir, mode = 'raw', update_flag = 50, frame_start = 0, frame_end = None, add_particles = False):
		"""
		More advanced GIF making method meant to extend simple raw GIF by allowing us to graph and alter frames as necessary
		Use this address for guidance https://ndres.me/post/matplotlib-animated-gifs-easily/
		
		Inputs:
		gif_dir - Directory to save the GIF
		mode - Used to dictate if the image with particle centers (raw) or the particles after linking (tracking) should be performed.
		update_flag - How many frames to process before reporting progress to the user.
		frame_start - What frame to start on.
		frame_end - Last frame to process.
		add_particles - Boolean dictating if particles should be plotted on the image.
		"""
		self.getFrame(frame_start) 		# Set the system to the correct first frame.
		is_next_frame = True 			# Boolean to start the while loop
		gif_data = [] 					# List where we will store our frame/image information
		progress_bar = 0 				# Start for our progress

		"""
		Decide the mode to be used.
		"""
		if mode == 'raw':
			while is_next_frame:
				gif_data.append(self.plotCurrImage(is_GIF = True, add_particles = add_particles))
				plt.close()
				is_next_frame = self.nextFrame()

				"""
				Update progress bar.
				"""
				progress_bar += 1
				if progress_bar % update_flag == 0:
					self.logger.info(str(progress_bar) + ' frames of ' + str(self.getDimensions()[0]) + ' completed!')
				if frame_end is not None and self.curr_frame_num > frame_end:
					break
			"""
			Finalize image and save output
			"""
			imageio.mimsave(gif_dir, gif_data, fps = 32)
		
		"""
		TODO: Implement tracking as a gif style.
		"""
		if mode == 'tracking':
			while is_next_frame:
				pass

		"""
		Save the GIF we have made and clean up all the figure handles that have been generated.
		"""
		self.logger.info('Finished processing. Saved at: ', gif_dir)
		plt.close('all')

	def dumpToJSON(self, directory = None):
		"""
		This method dumps all necessary parameters for troikaExperiment into a JSON file and triggers a JSON dump for every instance of particle
		contained within the particle dictionary.

		Inputs:
		directory - Optional directory parameter to reset where Troika experiment is going to be saving.

		Outputs.
		None
		"""
		
		"""
		Alter the save path if necessary.
		"""
		self.setSavePath(save_dir = directory)

		"""
		Writing all particle info into a giant list
		"""
		directory = self.save_dir+'_particles.json'
		all_particles = []
		active_particles = self.getRealParticles()
		for i in active_particles:
			all_particles.append(self.particles[i].dumpToJSON())

		"""
		Write all particles to file
		"""
		with open(directory, 'w') as fp1:
			json.dump(all_particles, fp1)
			fp1.close()

		"""
		Writing TE info to dictionary to file
		"""
		t_e_dict = dict()
		t_e_dict['data_dir'] = self.__dict__['data_dir']
		t_e_dict['frame_tolerance'] = self.__dict__['frame_tolerance']
		t_e_dict['part_in_frame'] = self.__dict__['part_in_frame'].tolist()

		with open(self.save_dir+'t_e_settings.json', 'w') as fp1:
			json.dump(t_e_dict, fp1)
			fp1.close()

		"""
		Write adjusted frames to file? It might be better to add a "replay" function that contains the order
		of frame adjustments to recreate adjusted frames rather than save them.
		"""
	def fromJSON(self, json_dir):
		"""
		This method reconstitutes a troikaExperiment from a directory containing the necessary JSON files

		Inputs:
		json_dir - A directory URL leading to all the necessary information to reconstitute a troikaExperiment
		"""

		"""
		Reload and rebuild all stored particles.
		"""
		with open(json_dir+'_particles.json') as json_file:
			particles = json.load(json_file)
			for p in particles:
				self.addNewParticle(j_dict = p)

		"""
		Rebuilding the troikaExperiment settings from file.
		"""
		with open(json_dir+'t_e_settings.json') as json_file:
			j_dict = json.load(json_file)
			for i in self.__dict__.viewkeys() & j_dict.viewkeys():
				if type(j_dict[i]) is list:
					self.__dict__[i] = np.array(j_dict[i], dtype='float16')
				else:
					self.__dict__[i] = j_dict[i]

		"""
		Reset the new troikaExperiment to the given data directory
		"""
		self.setData(self.data_dir)


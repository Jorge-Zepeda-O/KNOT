#%% --- IMPORTS --- %%#
### External ###
from enum 		import Enum

#%% --- FILEFORMAT ENUM DEFINITION --- %%#
class FILEFORMAT(Enum):
	### Static Parameters ###
	FXN  = '.fxn'	# For simulation only			#
	XML  = '.xml'	# For evaluation only			#
	MAT  = '.mat'	# For Troika compatibility		#
	JSON = '.json'	# For strictly data				#

	PNG  = '.png'	# For static images				#
	TIF  = '.tif'	# For layered images or movies	#
	GIF  = '.gif'	# For movies and visualization 	#

	### Overrides ###
	def __str__(self):
		return str(self.value)

#%% --- FUNCTIONCODE ENUM DEFINITION --- %%#
class FUNCTIONCODE(Enum):
	### Static Parameters ###
	## Linear Functions ##
	POINT = 0		# b
	LINE = 1		# A*t + b
	POLY = 2		# A*(c_n t^n + c_{n-1} t^{n-1} + ... + c_1 t) + b

	## Periodic Functions ##
	SINE = 10		# A * sin(2pi * f * t + phi) + b
	COSINE = 11		# A * cos(2pi * f * t + phi) + b

	## Transcedental Functions ##
	EXP = 20		# A * exp((t - mu)/k) + b

	## Distributions ##
	GAUSS = 30		# A * exp(-1/2 * ((t - mu)/sigma)^2) + b
	LORENTZ = 31	# A / (1 + ((t - mu)/gamma)^2) + b
	LEVY = 32		# A * exp(-1/2 * (c/(t - mu))) / (x - mu)^(3/2) + b

	## Processes ##
	WIENER = -1		# A * sum(sigma * N(0,1) * sqrt(dt) + mu * dt) + b
	CONFINED = -2	# A * (rho - sigma*N(0,1)) + b

	### Static Methods ###
	@staticmethod
	def __str__():
		return __name__

	@classmethod
	def _GetArgs(cls, code):
		## Linear Functions ##
		if(code is cls.POINT):		return []		# <No arguments> #
		elif(code is cls.LINE):		return []		# <No arguments> #
		elif(code is cls.POLY):		return [1,0]	# Coefficients #

		## Periodic Functions ##
		elif(code is cls.SINE):		return [1,0]	# Linear Frequency, Phase #
		elif(code is cls.COSINE):	return [1,0]	# Linear Frequency, Phase #

		## Transcedental Functions ##
		elif(code is cls.EXP):		return [0,1]	# Shift, Decay #

		## Distributions ##
		elif(code is cls.GAUSS):	return [0,1]	# Shift, Standard Deviation #
		elif(code is cls.LORENTZ):	return [0,1]	# Shift, Full-Width at Half-Max #
		elif(code is cls.LEVY):		return [0,1]	# Shift, Scale #

		## Processes ##
		elif(code is cls.WIENER):	return [0,1]	# Drift, Diffusion Coefficient #
		elif(code is cls.CONFINED):	return [1,1]	# Radius, Diffusion Coefficient #

#%% --- PHASE MASK ENUM DEFINITION --- %%#
class PHASEMASK(Enum):
	### Static Parameters ##
	NONE = 0	# No phase mask - Accessible through HELIX	#
	ASTIG = 1	# Astigmatism	- Not implemented yet		#
	HELIX = 2	# Double Helix								#
	TETRA = 3	# Tetrapod		- Not implemented yet		#

	### Overrides ###
	def __str__(self):
		if(self.value == 0):	return 'NONE'
		elif(self.value == 1):	return 'ASTIG'
		elif(self.value == 2):	return 'HELIX'
		elif(self.value == 3):	return 'TETRA'

#%% --- LOCALIZATION ENUM DEFINITION --- %%#
class LOCALIZATION(Enum):
	### Static Parameters ###
	XY = 0		# No phase mask requested
	XYZ = 1		# Stationary double-helix phase mask
	XYT = 2		# Rotating double-helix phase mask 			- Implemented, but sub-par performance
	XYZT = 3	# Stretching-lobe double-helix phase mask	- Implemented, but not working yet
	
	### Accessors ###
	@classmethod
	def _GetState(cls, loc_z=False, loc_t=False):
		## Output ##
		if(loc_z and loc_t):			return cls.XYZT		# Stretching-lobe 			#
		elif(loc_z and (not loc_t)):	return cls.XYZ		# Stationary double helix 	#
		elif((not loc_z) and loc_t):	return cls.XYT		# Rotating double helix 	#
		else:							return cls.XY		# No phase mask 			#
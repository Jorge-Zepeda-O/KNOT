#%% --- IMPORTS --- %%#
### External ###
import numpy as np
from __ENUM import PHASEMASK as PM

#%% --- INSTRUMENT --- %%#
### Detector ###
CHIP = 128 * np.ones(2, dtype=int)	# <int> Pixels along each dimension in the camera (May get updated as needed)
RES = 0.0685 * np.ones(2)			# <um> Lateral resolution		(67 nm ISBI | 68.5 nm ours)
DOF = np.array([4.000, 1.600])		# <um> Axial depth of field		(3D DH PSF, Stretching Lobe)
FRATE = 0.030						# <s> Frame rate of the camera	(30 ms ours)

### Phase Mask ###
KER_TYPE = PM.HELIX					# <PHASEMASK> Enum dictating which phase mask to use. #

## Samples in meta-dimensions ##
KER_Z = 16	# <int> Number of depth samples
KER_T = 1	# <int> Number of sub-frame samples

## Aperture details ##
WAVELENGTH = 0.560	# <nm> Excitation wavelength
NA = 1.46			# <#> Numerical Aperture

# NB: Pavani's doctoral thesis states:
# - The DH-PSF base width is 1.7x that of the Airy disk
# - The lobes are sparated by 3x the diffraction limit 
# https://pdfs.semanticscholar.org/cbd2/59fdecf5e08f63963d097be05d3913891490.pdf
APR_SHP = 'cir'				# <''> The shape of the PSF ('cir', 'elp', 'squ', 'rec')#
APR_RAD = WAVELENGTH/(2*NA)	# <um> The width of a singular lobe						#
if((KER_Z > 1 or KER_T > 1) and (KER_TYPE == PM.HELIX)): APR_RAD *= 1.7

## Double Helix Kernel ##
KER_LOOP = 1								# <int> How many half-turns does the phase mask go through per frame? #
KER_ROT = 1.000								# <# pi> How far the DH rotates at the ends #
KER_SEP = 4*WAVELENGTH/(2*NA)				# <um> How far apart are the two DH lobes?	#
KER_RNG = np.array([2/3*KER_SEP, KER_SEP])	# <um> How much does the DH lobes stretch?	#

# From calibration #
KER_SEP = 1.000
APR_RAD = 0.380
KER_RNG = [1.500]*2

#%% --- SIMULATION --- %%#
SIM_KER = True		# <bool> Simulate the phase mask kernel?	#
SIM_IMG = False		# <bool> Simulate the image?				#
SIM_ART = False		# <bool> Simulate artifacts?				#

SIM_FRAMES = 100		# <int> Number of frames to simulate		#
SIM_SUB = 1			# <int> Number of sub-frame samples to simulate (if applicable) #
FPS = 40			# <int> Frames per Second when simulating	#

#%% --- PROCESSING --- %%#
### Parallelization ###
PAR_FLAG = False	# <bool> Paralellize? (Not yet implemented)	 #
PAR_CORES = 6		# <int> Number of paralell processes to open #

### Preprocessing ###
PRE_BG = APR_RAD*2/3	# <#> Gaussian standard deviation for BG Sub filter				(4/3 R_apr) # 
PRE_NS = APR_RAD*1/6	# <#> Lorentzian HWHM for noise suppression filter				(2/3 R_apr)	#
PRE_LT = APR_RAD*3/2	# <#> Gaussian standard deviation for local threshold filter	(9/3 R_apr)	#
PRE_TS = 0				# <int> Temporal smoothing length - set to zero if no smoothing (0 frames)	#

PRE_EPS = 0.75			# <#> Local threshold strength - vary depending on SNR for best results (0.5) #

### Recovery ###
REC_CHUNK = False		# <bool> Chunk up the image when deconvolving? #
REC_ITER = 80 * (3 if(KER_Z>1) else 1) * (6 if(KER_T>1) else 1)
#REC_ITER = 80 * (6 if(KER_Z>1) else 1) * (12 if(KER_T>1) else 1)	# It really shouldn't take this long #

### Identification ###
SEG_RAD = np.sqrt(3)	# <#> Maximum distance between connected points - in *voxels*	#
SEG_SEP = 0.200			# <um> Maximum distance to merge clusters (Not implemented)		#
SEG_MERGE = 5			# <int> Number of iterations to merge clusters					#

### Tracking ###
TRK_RAD = 0.500		# <um> Search radius for tracking candidates			(0.700 um)	#
TRK_TOL = 3			# <fr> Number of frames not present before termination	(3 frames)	#

TRK_LEN = 6		# <fr> Maximum number of frames to keep in history	(6 frames)	#
TRK_MIN = 5		# <fr> Minimum number of frames to keep trajectory	(5 frames)	#
TRK_KDE = 180	# <int> Number of samples for KDEs					(180 samp)	#

#%% --- DERIVED PARAMETERS --- %%#
### Instrument ###
FOV = CHIP * RES									# <um> Field of Vision #
EXT = np.array([FOV[0],-FOV[0],-FOV[1],FOV[1]])/2	# "Extent" parameter in matplotlib's `imshow` #

### Processing ###
MESH_RHO = np.linspace(0, TRK_RAD, TRK_KDE)		# [#] Array of locations to evaluate rho			#
MESH_PHI = np.linspace(-np.pi, np.pi, TRK_KDE)	# [#] Array of locations to evaluate phi and theta	#
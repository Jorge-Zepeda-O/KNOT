#%% --- IMPORTS --- %%#
### External ###
import numpy 			as np
import scipy.special 	as sp

### Internal ###
from __ENUM 	import FUNCTIONCODE as FXNCODE

#%% --- STATIC PARAMETERS --- %%#
DEFAULT_CODE = FXNCODE.POINT
DEFAULT_DOMAIN = np.linspace(-1, 1, 101)
DEFAULT_SEED = 0

#%% --- STATIC FUNCTION CONSTRUCTORS --- %%#
## Linear Functions ##
def _Point(amp=1, off=0, seed=None):
	return Function(FXNCODE.POINT,			amp=amp, off=off, seed=seed)
def _Line(amp=1, off=0, seed=None):
	return Function(FXNCODE.LINE,			amp=amp, off=off, seed=seed)
def _Poly(*args, amp=1, off=0, seed=None):
	return Function(FXNCODE.POLY,	*args,	amp=amp, off=off, seed=seed)

## Periodic Functions ##
def _Sine(*args, amp=1, off=0, seed=None):
	return Function(FXNCODE.SINE,	*args,	amp=amp, off=off, seed=seed)
def _Cosine(*args, amp=1, off=0, seed=None):
	return Function(FXNCODE.COSINE, *args,	amp=amp, off=off, seed=seed)

## Transcendental Functions ##
def _Exp(*args, amp=1, off=0, seed=None):
	return Function(FXNCODE.EXP,	*args,	amp=amp, off=off, seed=seed)

## Distributions ##
def _Gauss(*args, amp=1, off=0, seed=None):
	return Function(FXNCODE.GAUSS,	*args,	amp=amp, off=off, seed=seed)
def _Lorentz(*args, amp=1, off=0, seed=None):
	return Function(FXNCODE.LORENTZ,*args,	amp=amp, off=off, seed=seed)
def _Levy(*args, amp=1, off=0, seed=None):	
	return Function(FXNCODE.LEVY,	*args,	amp=amp, off=off, seed=seed)

## Processes ##
def _Wiener(*args, amp=1, off=0, seed=None):
	return Function(FXNCODE.WIENER,		*args, amp=amp, off=off, seed=seed)
def _Confined(*args, amp=1, off=0, seed=None):
	return Function(FXNCODE.CONFINED,	*args, amp=amp, off=off, seed=seed)

#%% --- STATIC FUNCTION HANDLES --- %%#
## Linear Functions ##
def _HPoint():
	return lambda t: 0*t
def _HLine():
	return lambda t: t
def _HPoly(*c):
	args = [c[n] if(isinstance(c[n],Function)) else _Point(off=c[n]) for n in range(len(c))]
	return lambda t: np.sum(np.array([args[n](t) for n in range(len(c))]) * 
		np.power.outer(t, np.arange(len(c), 1, -1)).T, axis=0)

## Periodic Functions ##
def _HSine(f, phi):
	# If any parameter is *not* a function, make it into a Point so we can call it #
	if(not isinstance(f, Function)):		f = _Point(off=f)			# Frequency #
	if(not isinstance(phi, Function)):		phi = _Point(off=phi)		# Phase 	#
	return lambda t: np.sin(2*np.pi* f(t) * t - phi(t))
def _HCosine(f, phi):
	# If any parameter is *not* a function, make it into a Point so we can call it #
	if(not isinstance(f, Function)):		f = _Point(off=f)			# Frequency #
	if(not isinstance(phi, Function)):		phi = _Point(off=phi)		# Phase 	#
	return lambda t: np.cos(2*np.pi* f(t) * t - phi(t))
	
## Transcedental Functions ##
def _HExp(mu, k):
	# If any parameter is *not* a function, make it into a Point so we can call it #
	if(not isinstance(mu, Function)):		mu = _Point(off=mu)			# Shift	#
	if(not isinstance(k, Function)):		k = _Point(off=k)			# Decay	#
	return lambda t: np.exp((t - mu(t))/k(t))

## Unnormalized Distributions ##
def _HGauss(mu, sigma):
	# If any parameter is *not* a function, make it into a Point so we can call it #
	if(not isinstance(mu, Function)):		mu = _Point(off=mu)			# Shift	#
	if(not isinstance(sigma, Function)):	sigma = _Point(off=sigma)	# Standard Deviation #
	return lambda t: np.exp(-1/2 * np.square((t - mu(t))/sigma(t)))
def _HLorentz(mu, gamma):
	# If any parameter is *not* a function, make it into a Point so we can call it #
	if(not isinstance(mu, Function)):		mu = _Point(off=mu)			# Shift	#
	if(not isinstance(gamma, Function)):	gamma = _Point(off=gamma)	# Half-Width @ Half-Max #
	return lambda t: 1/(1 + np.square((t - mu(t))/gamma(t)))
def _HLevy(mu, c):
	if(not isinstance(mu, Function)):		mu = _Point(off=mu)			# Shift	#
	if(not isinstance(c, Function)):		c = _Point(off=c)			# Scale #
	return lambda t: np.exp(-1/2 * (c(t)/(t - mu(t) + 1E-6))) / np.power(t - mu(t) + 1E-6, 3/2)

## Processes ##
def _HWiener(mu, sigma):
	if(not isinstance(mu, Function)):		mu = _Point(off=mu)			# Shift	#
	if(not isinstance(sigma, Function)):	sigma = _Point(off=sigma)	# Standard Deviation #
	return lambda t: np.cumsum(
		sigma(t) * np.random.randn(*np.shape(t)) *
		np.sqrt(np.concatenate([[0], np.diff(t)], axis=0))
		+ mu(t) * np.concatenate([[0], np.diff(t)], axis=0)) 
def _HConfined(rho, sigma):
	if(not isinstance(rho, Function)):		rho = _Point(off=rho)		# Radius	#
	if(not isinstance(sigma, Function)):	sigma = _Point(off=sigma)	# Standard Deviation #
	return lambda t: np.maximum(np.minimum(rho(t)/3 * sigma(t)*np.random.randn(*np.shape(t)),rho(t)),-rho(t))

#%% --- FUNCTION CLASS DEFINITION --- %%#
class Function:
	### Constructor ###
	def __init__(self, code=DEFAULT_CODE, *args, amp=1, off=0, seed=None):
		## Argument Transfer ##
		self.code = code
		self.amp = amp
		self.off = off

		# Every function gets its own seed if one is not specified #
		self.seed = seed if seed is not None else np.random.randint(2*32)

		## Build the arguments ##
		self.args = FXNCODE._GetArgs(code)	# Default Arguments 				#
		self.args[:len(args)] = args		# Overwrite with supplied arguments #

	### Accessors ###
	def _GetHandle(self):
		# Linear Functions #
		if(self.code == FXNCODE.POINT): 		return _HPoint
		elif(self.code == FXNCODE.LINE):		return _HLine
		elif(self.code == FXNCODE.POLY):		return _HPoly

		# Periodic Functions #
		elif(self.code == FXNCODE.SINE):		return _HSine
		elif(self.code == FXNCODE.COSINE):		return _HCosine

		# Transcedental Functions #
		elif(self.code == FXNCODE.EXP):			return _HExp

		# Distributions #
		elif(self.code == FXNCODE.GAUSS):		return _HGauss
		elif(self.code == FXNCODE.LORENTZ):		return _HLorentz
		elif(self.code == FXNCODE.LEVY):		return _HLevy

		# Processes #
		elif(self.code == FXNCODE.WIENER):		return _HWiener
		elif(self.code == FXNCODE.CONFINED):	return _HConfined

	### Methods ###
	def Eval(self, *args, amp=None, off=None, domain=DEFAULT_DOMAIN, seed=None):
		## Argument Defaults ##
		if(args): 			self.args[:len(args)] = args	# Overwrite with supplied arguments #
		if(amp is None): 	amp = self.amp
		if(off is None): 	off = self.off
		if(seed is None):	seed = self.seed

		## Break out of Recursion ##
		amp_ = amp.Eval(domain=domain, seed=amp.seed) if(isinstance(amp, Function)) else amp
		off_ = off.Eval(domain=domain, seed=off.seed) if(isinstance(off, Function)) else off

		## Evaluation ##
		if(seed is not None):	np.random.seed(seed)
		fxn = self._GetHandle()
		self.curve = fxn(*self.args)(domain)

		## Output ##
		return amp_ * self.curve + off_

	### Overrides ###
	def __call__(self, domain=DEFAULT_DOMAIN, seed=None):
		if(self.seed is not None and seed is None):	seed = self.seed
		return self.Eval(domain=domain, seed=seed)

	## Operators ##
	def __add__(self, o):
		# Addition - just combine offsets #
		if(not isinstance(o, Function)): o = _Point(off=o)
		off = Function(o.code, *o.args, amp=o.amp, off=self.off + o.off, seed=o.seed)
		return Function(self.code, *self.args, amp=self.amp, off=off, seed=self.seed)
	def __sub__(self, o):
		# Subtraction - invert amplitude and combine offsets #
		if(not isinstance(o, Function)): o = _Point(off=o)
		off = Function(o.code, *o.args, amp=-o.amp, off=self.off - o.off, seed=o.seed)
		return Function(self.code, *self.args, amp=self.amp, off=off, seed=self.seed)
	def __mul__(self, o):
		# Multiplication - combine amplitudes #
		if(not isinstance(o, Function)): o = _Point(amp=o)
		amp = Function(o.code, *o.args, amp=self.amp * o.amp, off=o.off, seed=o.seed)
		return Function(self.code, *self.args, amp=amp, off=self.off, seed=self.seed)
	def __div__(self, o):
		# Division - combine amplitudes but opposite #
		if(not isinstance(o, Function)): o = _Point(amp=o)
		amp = Function(o.code, *o.args, amp=self.amp / o.amp, off=o.off, seed=o.seed)
		return Function(self.code, *self.args, amp=amp, off=self.off, seed=self.seed)
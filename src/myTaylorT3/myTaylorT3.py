#Author: Gonzalo Morras Gutierrez
#E-mail: gonzalo.morras@estudiante.uam.es

import numpy as np
from lalsimulation import SimInspiralTaylorLength

M_sun = 1.98841e30           #mass of the sun in kilograms
t_sun = 4.92549094830932e-6  #value of G*Msun/(c**3) in seconds = 1.32712440018e20/(299792458**3)
d_sun = 1476.6250382504018   #value of G*Msun/(c**2) in meters  = 1.32712440018e20/(299792458**2)
Mpc_m = 3.085677581491367e22 #value of 1 Megaparsec (Mpc) in meters

#class to implement TaylorT3 as outlined in arxiv:0907.0700
class myTaylorT3:
	
	#initialization
	def __init__(self, m1=None, m2=None, distance=None, inclination=0, sampling_rate=512,  coal_time=0, f_ref=20, phi_ref=0):
		#m1, m2 and distance MUST be given
		if None in (m1,m2, distance):
			raise Exception('m1, m2 and distance must be given to initialize class.')

		#assign the values
		self.m1 = m1                        #primary mass in solar masses 
		self.m2 = m2                        #secondary mass in solar masses
		self.inclination=inclination        #inclination in radians
		self.distance = distance            #distance in Mpc
		self.coal_time = coal_time          #time of coalescence in s
		self.sampling_rate = sampling_rate  #sampling rate in Hz
		self.f_ref = f_ref                  #reference frequency in Hz
		self.phi_ref = phi_ref              #reference phase in radians
		
		#obtain different quantities that will be needed in the future
		self.nu = m1*m2/((m1+m2)**2)               #symmetric mass ratio
		self.Mtot_s = (m1+m2)*t_sun                #value of G*(m1+m2)/(c**3) in seconds
		self.Mtot_m = (m1+m2)*d_sun                #value of G*(m1+m2)/(c**2) in meters
		self.delta_t = 1/sampling_rate             #delta_t in s
		self.distance_M=distance*Mpc_m/self.Mtot_m #value of the distance in units of G*(m1+m2)/(c**2)

		#reference value of theta 
		self.theta_lso = self.theta(-SimInspiralTaylorLength(float(self.delta_t), float(m1*M_sun), float(m2*M_sun), float(f_ref), 0))
		#reference value of x
		self.x_lso = (np.pi*f_ref*self.Mtot_s)**(2/3)

	#method to compute theta at a given time
	def theta(self, time):
		return (self.nu*(self.coal_time - time)/(5*self.Mtot_s))**(-1/8)

	#method to create a time array for t1<t2<t_coal at the desired sampling rate
	def time_array(self, t1, t2):
		#t1 has to be smaller than t2
		if t1>t2: 
			raise Exception('The start time can not be larger than the end time')

		#t2 has to be smaller than the coalescence time
		if t2>self.coal_time:
			raise Exception('The end time can not be larger than the coalescence time.')
		return t1 + np.arange(int((t2-t1)*self.sampling_rate)+1)/self.sampling_rate

	# method to obtain instantaneous GW frequency for an array of times from arXiv:0907.0700
	def freq(self, time):
		
		#compute the value of theta
		theta = self.theta(time)

		#extract M and nu from self
		M = self.Mtot_s
		nu = self.nu

		#return instantaneous frequency as defined in 0907.0700
		return ((theta**3)/(8*np.pi*M))*(1+((743/2688)+(11/32)*nu)*theta**2-(3/10)*np.pi*theta**3+((1855099/14450688)+(56975/258048)*nu+(371/2048)*nu**2)*theta**4-((7729/21504)-(13/256)*nu)*np.pi*theta**5+(-(720817631400877/288412611379200)+(53/200)*np.pi**2+(107/280)*np.euler_gamma+((25302017977/4161798144)-(451/2048)*np.pi**2)*nu-(30913/1835008)*nu**2+(235925/1769472)*nu**3+(107/280)*np.log(2*theta))*theta**6+(-(188516689/433520640)-(97765/258048)*nu+(141769/1290240)*nu**2)*np.pi*theta**7)	
	
	# method to obtain the orbital phase for an array of times from arXiv:0907.0700
	def phi(self, time):

		#compute the value of theta
		theta = self.theta(time)

		#extract nu, theta_lso and phi_ref from self
		M = self.Mtot_s
		nu = self.nu
		theta_lso = self.theta_lso
		phi_ref = self.phi_ref

		#return the time and phase
		phi = phi_ref-(1/(nu*theta**5))*(1+((3715/8064)+(55/96)*nu)*theta**2-((3*np.pi)/4)*theta**3+((9275495/14450688)+(284875/258048)*nu+(1855/2048)*nu**2)*theta**4+((38645/21504)-(65/256)*nu)*np.log((theta/theta_lso))*np.pi*theta**5+((831032450749357/57682522275840)-(53/40)*np.pi**2+(-(126510089885/4161798144)+(2255/2048)*np.pi**2)*nu-(107/56)*np.euler_gamma+(154565/1835008)*nu**2-(1179625/1769472)*nu**3-(107/56)*np.log(2*theta))*theta**6+((188516689/173408256)+(488825/516096)*nu-(141769/516096)*nu**2)*np.pi*theta**7)
		return phi


	#method to obtain the amplitude of the plus and cross polarizations as a function of time
	def tdstrain(self, t1, t2, PyCBC_TimeSeries=False):
		
		#compute the time array 
		time = self.time_array(t1, t2)
		
		#compute the value of the dimensionless frequency parameter x
		x = (np.pi*self.freq(time)*self.Mtot_s)**(2/3)

		#compute the amplitude prefactor A = 2*G*M*nu*x/((c**2)*distance)
		A = 2*self.nu*x/self.distance_M
		
		#compute the modified GW phase from arXiv:0802.1249v3
		psi = 2*(self.phi(time) - 3*(x**1.5)*(1 - 0.5*self.nu*x)*np.log(x/self.x_lso))

		#compute the plus and cross polarizations according to arXiv:0802.1249v3
		ci = np.cos(self.inclination)
		ci2 = ci**2
		ci4 = ci**4
		hp = A*np.cos(psi)*(-(1+ci2) + x*((19/6)+(3*ci2/2)-(ci4/3)+self.nu*(-(19/6)+(11*ci2/6)+ci4)) + (x**1.5)*(-2*np.pi*(1+ci2)))
		hc = A*np.sin(psi)*(-2*ci + x*ci*((17/3)-(4*ci2/3)+self.nu*(-(13/3)+4*ci2))+ (x**1.5)*(-4*np.pi*ci))

		#if PyCBC_TimeSeries is True, return PyCBC Timeseries
		if PyCBC_TimeSeries:
			from pycbc.types import TimeSeries
			return TimeSeries(hp, delta_t=time[1]-time[0], epoch=time[0]), TimeSeries(hc, delta_t=time[1]-time[0], epoch=time[0])
		else:
			#else, return a dictionary with the time array, the cross polarization and the plus polarizarion
			return {'time':time, 'hp': hp, 'hc': hc}

		


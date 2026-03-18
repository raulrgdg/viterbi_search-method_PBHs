from matplotlib import pyplot as plt
from myTaylorT3 import myTaylorT3
from pycbc import waveform
import numpy as np

import time
start_runtime = time.time()


#parameters
m1=1.37
m2=0.89
distance=5.77
t0 = -300
tf = -1
sampling_rate = 4096

#compute my waveform and convert to pycbc timerseries
my_hp, my_hc  = myTaylorT3(m1=m1, m2=m2, distance=distance, sampling_rate=sampling_rate).tdstrain(t0, tf, PyCBC_TimeSeries=True)
my_hp, my_hc = my_hp.trim_zeros(), my_hc.trim_zeros()
#obtain amplitude and phase from polarizations
my_amp = waveform.utils.amplitude_from_polarizations(my_hp, my_hc)
my_phase = waveform.utils.phase_from_polarizations(my_hp, my_hc)

#compute pycbc waveform
hp, hc = waveform.get_td_waveform(approximant='TaylorT4', mass1=m1, mass2=m2, distance=distance, inclination=0, delta_t=1.0/sampling_rate, f_lower=15)
#select the wanted time slice
hp = hp.time_slice(t0, tf+1/sampling_rate)                 
hc = hc.time_slice(t0, tf+1/sampling_rate)
hp, hc = hp.trim_zeros(), hc.trim_zeros()
#obtain amplitude and phase from polarizations
amp = waveform.utils.amplitude_from_polarizations(hp, hc)
phase = waveform.utils.phase_from_polarizations(hp, hc)


#make the beginning of my phase the same as pycbcs
my_phase = my_phase - (my_phase[0] - phase[0])

#compute relative error between my phase and their phase
phase_rel_err = 0.5*np.abs(phase-my_phase)/(phase+my_phase)
amp_rel_err = 0.5*np.abs(amp-my_amp)/(amp+my_amp)

#make plots
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'lines.linewidth': 2})

#make phase plot
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16,16), sharex=True, gridspec_kw={'height_ratios':[3, 1]})
ax[0].plot(my_hp.sample_times, my_phase, '-', label='myTaylorT3')
ax[0].plot(hp.sample_times, phase, '--', label='LALSimulation TaylorT3')
ax[0].set_ylabel('GW Phase (radians)')
ax[0].legend(loc='upper left')
ax[0].set_title(r'$m_1=%.2f M_\odot, \; m_2=%.2f M_\odot, \; d_L=%.2f \mathrm{Mpc}$'%(m1, m2, distance))
ax[1].plot(my_hp.sample_times, phase_rel_err)
ax[1].set_ylabel('relative error')
ax[1].set_xlabel('Time [s]')
ax[1].set_xlim(t0,tf)
plt.savefig('phase_comparison.png')

#make amplitude plot
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16,16), sharex=True, gridspec_kw={'height_ratios':[3, 1]})
ax[0].plot(my_hp.sample_times, my_amp, '-', label='myTaylorT3')
ax[0].plot(hp.sample_times, amp, '--', label='LALSimulation TaylorT3')
ax[0].set_ylabel('GW Strain Amplitude')
ax[0].legend(loc='upper left')
ax[0].set_title(r'$m_1=%.2f M_\odot, \; m_2=%.2f M_\odot, \; d_L=%.2f \mathrm{Mpc}$'%(m1, m2, distance))
ax[1].plot(my_hp.sample_times, amp_rel_err)
ax[1].set_ylabel('relative error')
ax[1].set_xlabel('Time [s]')
ax[1].set_xlim(t0,tf)
plt.savefig('ampitude_comparison.png')

#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

plt.show()



import pyulog
import matplotlib
from matplotlib import pyplot as plt
import os
import numpy as np
from numpy.fft import fft as fft, fftshift as fftshift, fftfreq as fftfreq
import math

# for logs created by gazebo (simulation)
log_path = '/home/lucas/src/px4/Firmware/build/px4_sitl_default/tmp/rootfs/log'
log_date = '2019-08-14'
log_time = '13_14_23'
log_file = f'{log_path}/{log_date}/{log_time}.ulg'

# for Alessandro's logs
log_path = '/home/lucas/Documents/Log_Analysis/Vibrations/Logs'
log_index = '3'
log_date = '2018-11-15'
log_time = '15-57-38'
log_file = f'{log_path}/log_{log_index}_{log_date}-{log_time}.ulg'

# creates a ULog object with the relevant topics
ulog = pyulog.ULog(log_file,['vehicle_local_position','sensor_combined','actuator_outputs']) 
datalist = ulog.data_list # is a list of Data objects, which contain the final topic data for a single topic and instance
data_vlp = datalist[0].data # is a dictionnary
data_sc = datalist[1].data
data_rpm = datalist[3].data # the first one is Aux

time_vlp = data_vlp['timestamp']/1e6 # convert it from us to s
time_sc = data_sc['timestamp']/1e6
time_ao = data_rpm['timestamp']/1e6 
acc_x = data_vlp['ax']
acc_y = data_vlp['ay']
acc_z = data_vlp['az']
gacc_x=data_sc['accelerometer_m_s2[0]']
gacc_y=data_sc['accelerometer_m_s2[1]']
gacc_z=data_sc['accelerometer_m_s2[2]']
roll = data_sc['gyro_rad[0]']
pitch = data_sc['gyro_rad[1]']
yaw = data_sc['gyro_rad[2]']
rpm1 = data_rpm['output[0]']
rpm2 = data_rpm['output[1]']
rpm3 = data_rpm['output[2]']
rpm4 = data_rpm['output[3]']
rpm5 = data_rpm['output[4]']
rpm6 = data_rpm['output[5]']

# a reasonable threshold for strong vibration is anything with a peak-to-peak of more than 2-3 m/s/s 
# (http://docs.px4.io/master/en/log/flight_log_analysis.html)

threshold = 1.5 # m/s^2, should be 3 peak to peak
threshold_list = [threshold for timestamp in time_vlp]

# Figure 1 : vibrations in time
plt.figure()
plt.plot(time_vlp,threshold_list,'r:')
plt.plot(time_vlp,[-elem for elem in threshold_list],'r:')
plt.plot(time_vlp,acc_x,label='x',alpha=.7)
plt.plot(time_vlp,acc_y,label='y',alpha=.7)
plt.plot(time_vlp,acc_z,label='z',alpha=.7)
plt.title(f'vibrations for log created on {log_date}@{log_time}')
plt.xlabel('time (s)')
plt.ylabel('local acceleration (m/s$^2$)')
plt.grid()
plt.legend()
plt.axis([time_vlp[0], time_vlp[-1], -threshold-1, threshold+1])

# creating a new directory if savedir does not exist and saving the figure
savedir = f'/home/lucas/Documents/Log_Analysis/Vibrations/Figures/{log_date}'

if not os.path.isdir(savedir):
    os.makedirs(savedir)
    print(f'The fodler {savedir} has been created.')
plt.savefig(f'{savedir}/z_acc_{log_time}.png')

plt.show()


threshold_list_sc = [threshold for timestamp in time_sc]

# Figure 2 : Raw acceleration time plot

plt.figure()
plt.plot(time_sc,threshold_list_sc,'r:')
plt.plot(time_sc,[-elem for elem in threshold_list_sc],'r:')
plt.plot(time_sc,gacc_x,label='x',linewidth=0.5,alpha=.7)
plt.plot(time_sc,gacc_y,label='y',linewidth=0.5,alpha=.7)
plt.plot(time_sc,gacc_z,label='z',linewidth=0.5,alpha=.7)
plt.xlabel('time (s)')
plt.ylabel('raw acceleration (m/s$^2$)')
plt.grid()
plt.legend()
plt.axis([time_sc[0], time_sc[-1], -30, 20])
# plt.axis([0, time_acc[-1], -threshold-1, threshold+1])
plt.show()


# Figure 3 : rounds per minutes of the propellers over time
plt.figure()
plt.plot(time_ao,rpm1,label="motor 1",alpha=.7)
plt.plot(time_ao,rpm2,label="motor 2",alpha=.7)
plt.plot(time_ao,rpm3,label="motor 3",alpha=.7)
plt.plot(time_ao,rpm4,label="motor 4",alpha=.7)
plt.plot(time_ao,rpm5,label="motor 5",alpha=.7)
plt.plot(time_ao,rpm6,label="motor 6",alpha=.7)
plt.xlabel('time (s)')
plt.ylabel('rotation speed (rpm)')
plt.axis([time_ao[0], time_ao[-1], None, None])
plt.grid()
plt.legend()
plt.savefig(f'{savedir}/rpm_{log_time}.png')
plt.show()

pfreq1 = rpm1/60 # convert to Hz
pfreq2 = rpm2/60
pfreq3 = rpm3/60
pfreq4 = rpm4/60
pfreq5 = rpm5/60
pfreq6 = rpm6/60

# Figure 4 : time spent per frequency in Hertz
plt.figure()
plt.plot(pfreq1,time_ao,label="motor 1",alpha=.7)
plt.plot(pfreq2,time_ao,label="motor 2",alpha=.7)
plt.plot(pfreq3,time_ao,label="motor 3",alpha=.7)
plt.plot(pfreq4,time_ao,label="motor 4",alpha=.7)
plt.plot(pfreq5,time_ao,label="motor 5",alpha=.7)
plt.plot(pfreq6,time_ao,label="motor 6",alpha=.7)
plt.axis([None, None, time_ao[0], time_ao[-1]])
plt.xlabel('rotation speed (Hz)')
plt.ylabel('time (s)')
plt.grid()
plt.legend()
plt.savefig(f'{savedir}/Hz_{log_time}.png')
plt.show()


# computing the frequency range of the accelerations
Nvlp = len(acc_z) # number of data points
dt_vlp = np.mean(np.diff(time_vlp)) # average sampling time in
freq_acc = fftfreq(Nvlp,dt_vlp)*360/(2*math.pi) # Hz

# computing the amplitudes of the accelerations
acc_x_complex_spectrum = fft(acc_x, Nvlp)
acc_x_complex_spectrum = acc_x_complex_spectrum
acc_x_amplitudes = np.abs(acc_x_complex_spectrum)
acc_x_phase = np.angle(acc_x_complex_spectrum)

acc_y_complex_spectrum = fft(acc_y, Nvlp)
acc_y_complex_spectrum = acc_y_complex_spectrum
acc_y_amplitudes = np.abs(acc_y_complex_spectrum)
acc_y_phase = np.angle(acc_y_complex_spectrum)

acc_z_complex_spectrum = fft(acc_z, Nvlp)
acc_z_complex_spectrum = acc_z_complex_spectrum
acc_z_amplitudes = np.abs(acc_z_complex_spectrum)
acc_z_phase = np.angle(acc_z_complex_spectrum)

Axpos = acc_x_amplitudes[:Nvlp//2]
Axneg = np.flip(acc_x_amplitudes[Nvlp//2+1:])
Ax = Axpos + Axneg

Aypos = acc_y_amplitudes[:Nvlp//2]
Ayneg = np.flip(acc_y_amplitudes[Nvlp//2+1:])
Ay = Aypos + Ayneg


Azpos = acc_z_amplitudes[:Nvlp//2]
Azneg = np.flip(acc_z_amplitudes[Nvlp//2+1:])
Az = Azpos + Azneg

# computing the frequency range of the accelerations
Nsc = len(roll) # number of data points
dt_sc = np.mean(np.diff(time_sc)) # average sampling time in
freq_angles = fftfreq(Nsc,dt_sc)*360/(2*math.pi) # Hz


# computing the amplitudes of the accelerations
roll_complex_spectrum = fft(roll, Nsc)
roll_complex_spectrum = roll_complex_spectrum
roll_amplitudes = np.abs(roll_complex_spectrum)

pitch_complex_spectrum = fft(pitch, Nsc)
pitch_complex_spectrum = pitch_complex_spectrum
pitch_amplitudes = np.abs(pitch_complex_spectrum)

yaw_complex_spectrum = fft(yaw, Nsc)
yaw_complex_spectrum = yaw_complex_spectrum
yaw_amplitudes = np.abs(yaw_complex_spectrum)

Rpos = roll_amplitudes[:Nsc//2]
Rneg = np.flip(roll_amplitudes[Nsc//2:])
R = Rpos + Rneg

Ppos = pitch_amplitudes[:Nsc//2]
Pneg = np.flip(pitch_amplitudes[Nsc//2:])
P = Ppos + Pneg

Ypos = yaw_amplitudes[:Nsc//2]
Yneg = np.flip(yaw_amplitudes[Nsc//2:])
Y = Ypos + Yneg


# Figure 5 : frequency spectrum of the acceleration
plt.figure()
plt.plot(freq_acc[:Nvlp//2],Ax,label='x',alpha=.7)
plt.plot(freq_acc[:Nvlp//2],Ay,label='y',alpha=.7)
plt.plot(freq_acc[:Nvlp//2],Az,label='z',alpha=.7)
plt.axis([0, freq_acc[Nvlp//2-1], 0, None])
plt.xlabel('frequencies')
plt.ylabel('amplitude')
plt.title('Vertical acceleration spectrum')
plt.grid()
plt.legend()
plt.savefig(f'{savedir}/acc_spectrum_{log_time}.png')
plt.show()

# criteria for ideal angle spectrum 
peak_limit = 20 #Hz
hf_limit = 20 #amplitude

# Figure 6 : frequency spectrum of the angles
plt.figure()
plt.plot(freq_angles[:Nsc//2],R,label='roll',alpha=.7)
plt.plot(freq_angles[:Nsc//2],P,label='pitch',alpha=.7)
plt.plot(freq_angles[:Nsc//2],Y,label='yaw',alpha=.7)
plt.axis([0, freq_angles[Nsc//2-1], 0, None]) # end index is not inclusive !
plt.xlabel('frequencies')
plt.ylabel('amplitude')
plt.title('Angles spectrum')
plt.grid()
plt.legend()
plt.axhline(hf_limit,linestyle=':',color='r')
plt.axvline(peak_limit,linestyle=':',color='r')
plt.savefig(f'{savedir}/angle_spectrum_{log_time}.png')
plt.show()

## Score :

# score calculation for raw acceleration 
# we want to count the blank area between the zacc and the xacc or yacc as positive points, if they overlap as negative
# define the maximum for normalization
max_acc_score = 9.81*len(time_sc) # having constantly the z at gravity and x,y at 0
acc_score = sum(np.min([gacc_x[index],gacc_y[index]])-gacc_z[index] for index in range(len(time_sc)))/max_acc_score
print(acc_score) # should be above 0.5

# score calculation for angles spectrum
# define the maximum for normalization
max_peak_score = peak_limit*np.sum((P>=hf_limit) | (R>=hf_limit) | (Y>=hf_limit))
peak_score = np.sum([peak_limit - freq_angles[np.argmax([P[index],R[index],Y[index]])] for index in range(Nsc//2) if ((P[index]>=hf_limit) |(R[index]>=hf_limit) |(Y[index]>=hf_limit))])/max_peak_score
print(peak_score)

# define the maximum for normalization
max_hf_score = hf_limit*(Nsc//2 - np.argmax(freq_angles.__gt__(peak_limit)))
# for each frequency above peak_limit, take the distance between hf_limit and the highest curve and sum them
hf_score = np.sum([hf_limit - np.max([R[index],P[index],Y[index]]) for index in range(Nsc//2) if (freq_angles[index]>peak_limit) ])/max_hf_score
print(hf_score)

# here the weights of each score can be chosen
angles_spectrum_score = sum([peak_score, hf_score])/2
print(angles_spectrum_score)

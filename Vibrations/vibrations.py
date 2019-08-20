import pyulog
import matplotlib
from matplotlib import pyplot as plt
import os
import numpy as np
from numpy.fft import rfft as rfft, rfftfreq as rfftfreq
import math

# for logs created by gazebo (simulation)
log_path = '/home/lucas/src/px4/Firmware/build/px4_sitl_default/tmp/rootfs/log'
log_date = '2019-08-14'
log_time = '13_14_23'
log_file = f'{log_path}/{log_date}/{log_time}.ulg'

# for real logs
log_path = '/home/lucas/Documents/Log_Analysis/Vibrations/Logs'
log_index = '270'
log_date = '2019-8-19'
log_time = '14-20-40'
log_file = f'{log_path}/log_{log_index}_{log_date}-{log_time}.ulg'

# creates a ULog object with the relevant topics
ulog = pyulog.ULog(log_file,['sensor_combined','actuator_outputs']) 
datalist = ulog.data_list # is a list of Data objects, which contain the final topic data for a single topic and instance
data_sc = datalist[0].data # is a dictionnary
data_rpm = datalist[2].data # the first one is Aux

time_sc = data_sc['timestamp']/1e6 # convert it from us to s
time_ao = data_rpm['timestamp']/1e6 
acc_x=data_sc['accelerometer_m_s2[0]']
acc_y=data_sc['accelerometer_m_s2[1]']
acc_z=data_sc['accelerometer_m_s2[2]']
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

# Figure 1 : Raw acceleration time plot
plt.figure()
plt.plot(time_sc,acc_x,label='x',alpha=.7)
plt.plot(time_sc,acc_y,label='y',alpha=.7)
plt.plot(time_sc,acc_z,label='z',alpha=.7)
plt.axhline(threshold,linestyle=':',color='r')
plt.axhline(-threshold,linestyle=':',color='r')
plt.title(f'vibrations for log created on {log_date}@{log_time}')
plt.xlabel('time (s)')
plt.ylabel('raw acceleration (m/s$^2$)')
plt.grid()
plt.legend()
plt.axis([time_sc[0], time_sc[-1], -30, 20])

# creating a new directory if savedir does not exist and saving the figure
savedir = f'/home/lucas/Documents/Log_Analysis/Vibrations/Figures/{log_date}'

if not os.path.isdir(savedir):
    os.makedirs(savedir)
    print(f'The fodler {savedir} has been created.')
plt.savefig(f'{savedir}/z_acc_{log_time}.png')

plt.show()

# Figure 2 : rounds per minutes of the propellers over time
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

# Figure 3 : time spent per frequency in Hertz
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
N = len(acc_z) # number of data points
dt = np.mean(np.diff(time_sc)) # average sampling time in
freq = rfftfreq(N,dt) # Hz

# computing the amplitudes of the accelerations
acc_x_complex_spectrum = rfft(acc_x)
acc_x_complex_spectrum = acc_x_complex_spectrum
Ax = np.abs(acc_x_complex_spectrum)

acc_y_complex_spectrum = rfft(acc_y)
acc_y_complex_spectrum = acc_y_complex_spectrum
Ay = np.abs(acc_y_complex_spectrum)

acc_z_complex_spectrum = rfft(acc_z)
acc_z_complex_spectrum = acc_z_complex_spectrum
Az = np.abs(acc_z_complex_spectrum)

# computing the amplitudes of the angles
roll_complex_spectrum = rfft(roll)
roll_complex_spectrum = roll_complex_spectrum
R = np.abs(roll_complex_spectrum)

pitch_complex_spectrum = rfft(pitch)
pitch_complex_spectrum = pitch_complex_spectrum
P = np.abs(pitch_complex_spectrum)

yaw_complex_spectrum = rfft(yaw)
yaw_complex_spectrum = yaw_complex_spectrum
Y = np.abs(yaw_complex_spectrum)


# Figure 4 : frequency spectrum of the acceleration
plt.figure()
plt.plot(freq,Ax,label='x',alpha=.7)
plt.plot(freq,Ay,label='y',alpha=.7)
plt.plot(freq,Az,label='z',alpha=.7)
plt.axis([0, freq[-1], 0, None])
plt.xlabel('frequencies')
plt.ylabel('amplitude')
plt.title('Vertical acceleration spectrum')
plt.grid()
plt.legend()
plt.savefig(f'{savedir}/acc_spectrum_{log_time}.png')
plt.show()

# criteria for ideal angle spectrum 
peak_limit = 20 #Hz
hf_limit = 500 #amplitude

# Figure 5 : frequency spectrum of the angles
plt.figure()
plt.plot(freq,R,label='roll',alpha=.7)
plt.plot(freq,P,label='pitch',alpha=.7)
plt.plot(freq,Y,label='yaw',alpha=.7)
plt.axis([0, freq[-1], 0, None]) # end index is not inclusive !
plt.xlabel('frequencies')
plt.ylabel('amplitude')
plt.title('Angles spectrum')
plt.grid()
plt.legend()
plt.axhline(hf_limit,linestyle=':',color='r')
plt.axvline(peak_limit,linestyle=':',color='r')
plt.savefig(f'{savedir}/angle_spectrum_{log_time}.png')
plt.show()

## Means above 40 Hz :

Rhf = []
for index in range(len(R)): 
    if freq[index] >= 40:
        Rhf.append(R[index])
    
Rmean = np.mean(Rhf)

print(f'mean roll above 40 Hz : {Rmean}')

Phf = []
for index in range(len(P)): 
    if freq[index] >= 40:
        Phf.append(P[index])
    
Pmean = np.mean(Phf)

print(f'mean pitch above 40 Hz : {Pmean}')

Yhf = []
for index in range(len(Y)): 
    if freq[index] >= 40:
        Yhf.append(Y[index])
    
Ymean = np.mean(Yhf)

print(f'mean yaw above 40 Hz : {Ymean}')



## Score :

# score calculation for raw acceleration 
# we want to count the blank area between the zacc and the xacc or yacc as positive points, if they overlap as negative
# define the maximum for normalization
max_acc_score = 9.81*len(time_sc) # having constantly the z at gravity and x,y at 0
acc_score = sum(np.min([acc_x[index],acc_y[index]])-acc_z[index] for index in range(len(time_sc)))/max_acc_score
print(f'acc score : {acc_score}') # should be above 0.5

# score calculation for angles spectrum
max_peak_score = peak_limit*np.sum((P>=hf_limit) | (R>=hf_limit) | (Y>=hf_limit))
peak_score = np.sum([peak_limit - freq[np.argmax([P[index],R[index],Y[index]])] for index in range(N//2) if ((P[index]>=hf_limit) |(R[index]>=hf_limit) |(Y[index]>=hf_limit))])/max_peak_score
print(f'peak score :{peak_score}')

max_hf_score = hf_limit*(N - np.argmax(freq.__gt__(peak_limit)))
# for each frequency above peak_limit, take the distance between hf_limit and the highest curve and sum them
hf_score = np.sum([hf_limit - np.max([R[index],P[index],Y[index]]) for index in range(N//2) if (freq[index]>peak_limit) ])/max_hf_score
print(f'hf score : {hf_score}')
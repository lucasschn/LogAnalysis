import pyulog
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
ulog = pyulog.ULog(log_file,['vehicle_local_position','vehicle_global_position','actuator_outputs']) 
datalist = ulog.data_list # is a list of Data objects, which contain the final topic data for a single topic and instance
data_vlp = datalist[0].data # is a dictionnary
data_vgp = datalist[1].data
data_rpm = datalist[2].data

time_vlp = data_vlp['timestamp']/1e6 # convert it from us to s
time_vgp = data_vgp['timestamp']/1e6
time_rpm = data_rpm['timestamp']/1e6 
pos_x = data_vlp['x']
pos_y = data_vlp['y']
pos_z = data_vlp['z']
acc_x = data_vlp['ax']
acc_y = data_vlp['ay']
acc_z = data_vlp['az']
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
plt.plot(time_vlp,acc_x,label='x')
plt.plot(time_vlp,acc_y,label='y')
plt.plot(time_vlp,acc_z,label='z')
plt.title(f'vibrations for log created on {log_date}@{log_time}')
plt.xlabel('time (s)')
plt.ylabel('local acceleration (m/s$^2$)')
plt.grid()
plt.legend()
plt.axis([0, time_vlp[-1], -threshold-1, threshold+1])

# creating a new directory if savedir does not exist and saving the figure
savedir = f'/home/lucas/Documents/Log_Analysis/Vibrations/Figures/{log_date}'

if not os.path.isdir(savedir):
    os.makedirs(savedir)
    print(f'The fodler {savedir} has been created.')
plt.savefig(f'{savedir}/z_acc_{log_time}.png')

plt.show()

# Figure 2 : round per minutes of the propellers in time
plt.figure()
plt.plot(time_rpm,rpm1,label="motor 1")
plt.plot(time_rpm,rpm2,label="motor 2")
plt.plot(time_rpm,rpm3,label="motor 3")
plt.plot(time_rpm,rpm4,label="motor 4")
plt.plot(time_rpm,rpm5,label="motor 5")
plt.plot(time_rpm,rpm6,label="motor 6")
plt.xlabel('time (s)')
plt.ylabel('rotation speed (rpm)')
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
plt.plot(pfreq1,time_rpm,label="motor 1")
plt.plot(pfreq2,time_rpm,label="motor 2")
plt.plot(pfreq3,time_rpm,label="motor 3")
plt.plot(pfreq4,time_rpm,label="motor 4")
plt.plot(pfreq5,time_rpm,label="motor 5")
plt.plot(pfreq6,time_rpm,label="motor 6")
plt.xlabel('rotation speed (Hz)')
plt.ylabel('time (s)')
plt.grid()
plt.legend()
plt.savefig(f'{savedir}/Hz_{log_time}.png')
plt.show()


# computing the frequency range of the accelerations
N = len(acc_z) # number of data points
dt = np.mean(np.diff(time_vlp)) # average sampling time in
freq = fftfreq(N,dt)*360/(2*math.pi) # Hz

# computing the amplitudes of the accelerations
acc_x_complex_spectrum = fft(acc_x, N)
acc_x_complex_spectrum = acc_x_complex_spectrum
acc_x_amplitudes = np.abs(acc_x_complex_spectrum)
acc_x_phase = np.angle(acc_x_complex_spectrum)

acc_y_complex_spectrum = fft(acc_y, N)
acc_y_complex_spectrum = acc_y_complex_spectrum
acc_y_amplitudes = np.abs(acc_y_complex_spectrum)
acc_y_phase = np.angle(acc_y_complex_spectrum)

acc_z_complex_spectrum = fft(acc_z, N)
acc_z_complex_spectrum = acc_z_complex_spectrum
acc_z_amplitudes = np.abs(acc_z_complex_spectrum)
acc_z_phase = np.angle(acc_z_complex_spectrum)

Axpos = acc_x_amplitudes[:N//2]
Axneg = np.flip(acc_x_amplitudes[N//2+1:])
Ax = Axpos + Axneg

Aypos = acc_y_amplitudes[:N//2]
Ayneg = np.flip(acc_y_amplitudes[N//2+1:])
Ay = Aypos + Ayneg


Azpos = acc_z_amplitudes[:N//2]
Azneg = np.flip(acc_z_amplitudes[N//2+1:])
Az = Azpos + Azneg

# Figure 4 : frequency spectrum of the acceleration
plt.figure()
plt.plot(freq[:N//2],Ax,label='x')
plt.plot(freq[:N//2],Ay,label='y')
plt.plot(freq[:N//2],Az,label='z')
plt.xlabel('frequencies')
plt.ylabel('amplitude')
plt.title('Vertical acceleration spectrum')
plt.grid()
plt.legend()
plt.savefig(f'{savedir}/spectrum_{log_time}.png')
plt.show()
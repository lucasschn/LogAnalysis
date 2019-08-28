import pyulog
import matplotlib
from matplotlib import pyplot as plt
import os
import numpy as np
from numpy.fft import rfft as rfft, rfftfreq as rfftfreq
import math
import sys
sys.path.append('/home/lucas/Documents/Log_Analysis')

from analog import logextract as logextract, logscore as logscore, sixaxes_spectrum as sixaxes_spectrum

# for logs created by gazebo (simulation)
log_path = '/home/lucas/src/px4/Firmware/build/px4_sitl_default/tmp/rootfs/log'
log_date = '2019-08-14'
log_time = '13_14_23'
log_file = f'{log_path}/{log_date}/{log_time}.ulg'

# for logs downloaded with QGC
log_path = '/home/lucas/Documents/Log_Analysis/Logs'
log_index = '38'
log_date = '2019-8-27'
log_time = '14-18-14'
log_file = f'{log_path}/log_{log_index}_{log_date}-{log_time}.ulg'

# custom path
# log_file = f'{log_path}/13_49_01.ulg'

info = logextract(log_file)

time_sc = info['time_sc'] # already in s
time_ao = info['time_ao'] 
acc_x=info['acc_x']
acc_y=info['acc_y']
acc_z=info['acc_z']
roll = info['roll']
pitch = info['pitch']
yaw = info['yaw']
rpm = info['rpm']

# a reasonable threshold for strong vibration is anything with a peak-to-peak of more than 2-3 m/s/s 
# (http://docs.px4.io/master/en/log/flight_log_analysis.html)

threshold = 1.5 # m/s^2, should be 3 peak to peak

# Figure 1 : Raw acceleration time plot
plt.figure()
plt.plot(time_sc,acc_x,label='x', linewidth =.5,alpha=.7)
plt.plot(time_sc,acc_y,label='y', linewidth =.5,alpha=.7)
plt.plot(time_sc,acc_z,label='z', linewidth =.5,alpha=.7)
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
plt.plot(time_ao,rpm[0],label="motor 1",alpha=.7)
plt.plot(time_ao,rpm[1],label="motor 2",alpha=.7)
plt.plot(time_ao,rpm[2],label="motor 3",alpha=.7)
plt.plot(time_ao,rpm[3],label="motor 4",alpha=.7)
plt.plot(time_ao,rpm[4],label="motor 5",alpha=.7)
plt.plot(time_ao,rpm[5],label="motor 6",alpha=.7)
plt.xlabel('time (s)')
plt.ylabel('rotation speed (rpm)')
plt.axis([time_ao[0], time_ao[-1], None, None])
plt.grid()
plt.legend()
plt.savefig(f'{savedir}/rpm_{log_time}.png')
plt.show()

pfreq1 = rpm[0]/60 # convert to Hz
pfreq2 = rpm[1]/60
pfreq3 = rpm[2]/60
pfreq4 = rpm[3]/60
pfreq5 = rpm[4]/60
pfreq6 = rpm[5]/60

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
plt.legend(loc='SouthEast')
plt.savefig(f'{savedir}/Hz_{log_time}.png')
plt.show()


# computing the frequency range of the accelerations
N = len(acc_z) # number of data points
dt = np.mean(np.diff(time_sc)) # average sampling time in s
freq = rfftfreq(N,dt) # Hz

spectrum = sixaxes_spectrum(info)

Ax = spectrum['Ax']
Ay = spectrum['Ay']
Az = spectrum['Az']
R = spectrum['R']
P = spectrum['P']
Y = spectrum['Y']

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
plt.plot(freq,R/1e3,label='roll',alpha=.7)
plt.plot(freq,P/1e3,label='pitch',alpha=.7)
plt.plot(freq,Y/1e3,label='yaw',alpha=.7)
plt.axis([0, freq[-1], 0, None]) # end index is not inclusive !
plt.xlabel('frequencies')
plt.ylabel('amplitude * 1000')
plt.title('Angles spectrum')
plt.grid()
plt.legend()
plt.axhline(hf_limit/1e3,linestyle=':',color='r')
plt.axvline(peak_limit,linestyle=':',color='r')
plt.savefig(f'{savedir}/angle_spectrum_{log_time}.png')
plt.show()

## Means above 40 Hz :

def mean40hz(signal, freq):
        Shf = []
        for index in range(len(signal)):
                if freq[index] >= 40:
                        Shf.append(R[index])
        hfmean = np.mean(Shf)
        return hfmean

Rmean = mean40hz(R,freq)
Pmean = mean40hz(P,freq)
Ymean = mean40hz(Y,freq)

print(f'mean roll above 40 Hz : {Rmean}')
print(f'mean pitch above 40 Hz : {Pmean}')
print(f'mean yaw above 40 Hz : {Ymean}')

## Score :
scores = logscore(info)

import math
import pyulog
import numpy as np
import os
import csv
from numpy.fft import rfft as rfft, rfftfreq as rfftfreq 

def logextract(path):
    ulog = pyulog.ULog(log_file,['sensor_combined','actuator_outputs']) 
    datalist = ulog.data_list # is a list of Data objects, which contain the final topic data for a single topic and instance
    for topic in datalist: 
        if topic.name == 'sensor_combined': 
            data_sc = topic.data
        else: 
            if (np.all(topic.data['noutputs'] <= 1)): 
                continue
            else: 
                    data_rpc = topic.data
   

    time = data_sc['timestamp']/1e6 # convert it from us to s
    acc_x=data_sc['accelerometer_m_s2[0]']
    acc_y=data_sc['accelerometer_m_s2[1]']
    acc_z=data_sc['accelerometer_m_s2[2]']
    roll = data_sc['gyro_rad[0]']
    pitch = data_sc['gyro_rad[1]']
    yaw = data_sc['gyro_rad[2]']
    info = {'time':time, 'acc_x':acc_x, 'acc_y':acc_y,'acc_z':acc_z, 'roll': roll, 'pitch': pitch, 'yaw': yaw}
    return info


def logscore(time,acc_x,acc_y,acc_z,roll,pitch,yaw):
    # computing the frequency range of the accelerations
    N = len(acc_z) # number of data points
    dt = np.mean(np.diff(time)) # average sampling time in
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

    # score calculation for raw acceleration 
    # we want to count the blank area between the zacc and the xacc or yacc as positive points, if they overlap as negative
    max_acc_score = 9.81*len(time) # having constantly the z at gravity and x,y at 0
    acc_score = sum(np.min([acc_x[index],acc_y[index]])-acc_z[index] for index in range(len(time)))/max_acc_score
    print(f'acc score : {acc_score}') # should be above 0.5

    peak_limit = 20 #Hz
    hf_limit = 500 #amplitude

    # score calculation for angles spectrum
    max_peak_score = peak_limit*np.sum((P>=hf_limit) | (R>=hf_limit) | (Y>=hf_limit))
    peak_score = np.sum([peak_limit - freq[np.argmax([P[index],R[index],Y[index]])] for index in range(N//2) if ((P[index]>=hf_limit) |(R[index]>=hf_limit) |(Y[index]>=hf_limit))])/max_peak_score
    print(f'peak score :{peak_score}')

    max_hf_score = hf_limit*(N - np.argmax(freq.__gt__(peak_limit)))
    # for each frequency above peak_limit, take the distance between hf_limit and the highest curve and sum them
    hf_score = np.sum([hf_limit - np.max([R[index],P[index],Y[index]]) for index in range(N//2) if (freq[index]>peak_limit) ])/max_hf_score
    print(f'hf score : {hf_score}')

    scores = {'acc_score': acc_score,'peak_score': peak_score,'hf_score': hf_score}
    return scores

def addscore(log_file,scores):
    csv_file = open('log_scores.csv','a')
    writer = csv.writer(csv_file)
    if not csv_file.readline == 'Log File, Acc score, Peak score, HF score':
        writer.writerow(['Log File', 'Acc score', 'Peak score', 'HF score'])
    writer.writerow([log_file,scores['acc_score'],scores['peak_score'],scores['hf_score']])

# path constructor
log_path = '/home/lucas/Documents/Log_Analysis/Vibrations/Logs'
files = os.listdir(log_path)

for file in files:
    log_file = f'{log_path}/{file}'
    print(log_file)
    info = logextract(log_file)
    scores = logscore(info['time'],info['acc_x'],info['acc_y'],info['acc_z'],info['roll'],info['pitch'],info['yaw'])
    addscore(log_file,scores)


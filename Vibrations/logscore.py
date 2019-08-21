import math
import numpy as np
import os
import csv
from numpy.fft import rfft as rfft, rfftfreq as rfftfreq 
from analog import logextract as logextract, ampspectrum as ampspectrum

topic_list = []
topic_list.append('sensor_combined')
topic_list.append('actuator_outputs')
topic_list.append('vehicle_local_position_setpoint')
topic_list.append('vehicle_status')
topic_list.append('vehicle_land_detected')
topic_list.append('manual_control_setpoint')


def addscore(log_file,scores):
    """ Adds the score results scores of the file log_file to the file log_scores.csv """
    path2csv = 'log_scores.csv'
    try:
        csv_file = open(path2csv)
        print(f'{path2csv} has been succesfully opened.')
        if csv_file.readline() == 'Log File,Acc score,Peak score,HF score\n':
            csv_file = open(path2csv,'a')
            print('Header correct.')
            writer = csv.writer(csv_file)
        else:
            print('Header incorrect, starting a new file.')  
            csv_file = open(path2csv,'w') 
            writer = csv.writer(csv_file)
            writer.writerow(['Log File','Acc score','Peak score','HF score'])
    except IOError as err:
        if err.errno == errno.EACCES:
            print(f'{path2csv} cannot be read. Creating a new one.')
            csv_file = open(path2csv,'w')
            writer = csv.writer(csv_file)
            writer.writerow(['Log File','Acc score','Peak score','HF score'])
        if err.errno == errno.ENOENT:
            print(f'{path2csv} does not exist. Creating one.')
            csv_file = open(path2csv,'w')
            writer = csv.writer(csv_file)
            writer.writerow(['Log File','Acc score','Peak score','HF score'])
    finally:
        writer.writerow([log_file,scores['acc_score'],scores['peak_score'],scores['hf_score']])


def logscore(time,acc_x,acc_y,acc_z,roll,pitch,yaw):
    # computing the frequency range of the accelerations
    N = len(acc_z) # number of data points
    dt = np.mean(np.diff(time)) # average sampling time in
    freq = rfftfreq(N,dt) # Hz

    # computing the amplitudes of the accelerations
    Ax = ampspectrum(acc_x)
    Ay = ampspectrum(acc_y)
    Az = ampspectrum(acc_z)

    # computing the amplitudes of the angles
    R = ampspectrum(roll)
    P = ampspectrum(pitch)
    Y = ampspectrum(yaw)

    # score calculation for raw acceleration 
    # we want to count the blank area between the zacc and the xacc or yacc as positive points, if they overlap as negative
    max_acc_score = 9.81*len(time) # having constantly the z at gravity and x,y at 0
    acc_score = sum(np.min([acc_x[index],acc_y[index]])-acc_z[index] for index in range(len(time)))/max_acc_score
    print(f'acc score : {acc_score}') # should be above 0.5

    peak_limit = 20 #Hz
    hf_limit = 500 #amplitude

    # score calculation for angles spectrum
    max_peak_score = peak_limit*np.sum((P>=hf_limit) | (R>=hf_limit) | (Y>=hf_limit))
    if max_peak_score == 0:
        peak_score=1
    else:
        peak_score = np.sum([peak_limit - freq[np.argmax([P[index],R[index],Y[index]])] for index in range(N//2) if ((P[index]>=hf_limit) |(R[index]>=hf_limit) |(Y[index]>=hf_limit))])/max_peak_score
    print(f'peak score :{peak_score}')

    max_hf_score = hf_limit*(N - np.argmax(freq.__gt__(peak_limit)))
    # for each frequency above peak_limit, take the distance between hf_limit and the highest curve and sum them
    hf_score = np.sum([hf_limit - np.max([R[index],P[index],Y[index]]) for index in range(N//2) if (freq[index]>peak_limit) ])/max_hf_score
    print(f'hf score : {hf_score}')

    scores = {'acc_score': acc_score,'peak_score': peak_score,'hf_score': hf_score}
    return scores


# path constructor
log_path = '/home/lucas/Documents/Log_Analysis/Vibrations/Logs'
files = os.listdir(log_path)

for file in files:
    log_file = f'{log_path}/{file}'
    print(log_file)
    info = logextract(log_file,topic_list)
    scores = logscore(info['time_sc'],info['acc_x'],info['acc_y'],info['acc_z'],info['roll'],info['pitch'],info['yaw'])
    addscore(log_file,scores)
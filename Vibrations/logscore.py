import math
import numpy as np
from numpy.fft import rfftfreq as rfftfreq 
import csv
import os
import sys 
sys.path.append('/home/lucas/Documents/Log_Analysis')

from analog import logextract as logextract, logscore as logscore, ampspectrum as ampspectrum, LogError

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
    except IOError:
        print(f'{path2csv} cannot be read. Creating a new one.')
        csv_file = open(path2csv,'w')
        writer = csv.writer(csv_file)
        writer.writerow(['Log File','Acc score','Peak score','HF score'])
    finally:
        writer.writerow([log_file,scores['acc_score'],scores['peak_score'],scores['hf_score']])

# path constructor
log_path = '/home/lucas/Documents/Log_Analysis/Logs'
files = os.listdir(log_path)

for file in files:
    log_file = f'{log_path}/{file}'
    print(log_file)
    try :
        info = logextract(log_file,topic_list)
        scores = logscore(info)
        addscore(log_file,scores)
    except LogError:
        print(f'{log_file} is not relevant. Discarded.')
        continue
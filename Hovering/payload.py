import sys

sys.path.append('/home/lucas/Documents/Log_Analysis')

from analog import avghover as avghover, LogError
import numpy as np
import os
import csv 

def addhover(log_file,avg_hovering):
    """ Adds the average hovering rpm results of the file log_file to the file avg_rpm.csv """
    path2csv = 'avg_hover.csv'
    try:
        csv_file = open(path2csv)
        print(f'{path2csv} has been succesfully opened.')
        if csv_file.readline() == 'Log File,Avg rpm,Avg vert thrust\n':
            csv_file = open(path2csv,'a')
            print('Header correct.')
            writer = csv.writer(csv_file)
        else:
            print('Header incorrect, starting a new file.')  
            csv_file = open(path2csv,'w') 
            writer = csv.writer(csv_file)
            writer.writerow(['Log File','Avg rpm','Avg vert thrust'])
    except IOError:
        print(f'{path2csv} cannot be read or does not exist. Creating a new one.')
        csv_file = open(path2csv,'w')
        writer = csv.writer(csv_file)
        writer.writerow(['Log File','Avg rpm','Avg vert thrust'])
    finally:
        writer.writerow([log_file,avg_hovering['rpm'],avg_hovering['zthrust']])



# file to be tested
log_path = '/home/lucas/Documents/Log_Analysis/Logs'
log_index = '270'
log_date = '2019-8-19'
log_time = '14-33-43'
test_file = f'{log_path}/log_{log_index}_{log_date}-{log_time}.ulg'

files = os.listdir(log_path)

# database construction
for file in files:
    log_file = f'{log_path}/{file}'
    print(log_file)
    try :
        avg_hovering = avghover(log_file)
    except LogError:
        print(f'{log_file} is not relevant. Discarded.')
        continue
    addhover(log_file,avg_hovering)

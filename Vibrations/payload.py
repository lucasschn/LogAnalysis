from analog import avghover as avghover
import numpy as np
import os
import csv 

def addrpm(log_file,rpm):
    """ Adds the average hovering rpm results of the file log_file to the file avg_rpm.csv """
    path2csv = 'avg_rpm.csv'
    try:
        csv_file = open(path2csv)
        print(f'{path2csv} has been succesfully opened.')
        if csv_file.readline() == 'Log File,Avg rpm\n':
            csv_file = open(path2csv,'a')
            print('Header correct.')
            writer = csv.writer(csv_file)
        else:
            print('Header incorrect, starting a new file.')  
            csv_file = open(path2csv,'w') 
            writer = csv.writer(csv_file)
            writer.writerow(['Log File','Avg rpm'])
    except IOError:
        print(f'{path2csv} cannot be read or does not exist. Creating a new one.')
        csv_file = open(path2csv,'w')
        writer = csv.writer(csv_file)
        writer.writerow(['Log File','Avg rpm'])
    else: 
        csv_file = open(path2csv,'w')
        writer = csv.writer(csv_file)
        writer.writerow(['Log File','Avg rpm'])
    finally:
        writer.writerow([log_file,rpm])



# file to be tested
log_path = '/home/lucas/Documents/Log_Analysis/Vibrations/Logs'
log_index = '270'
log_date = '2019-8-19'
log_time = '14-20-40'
test_file = f'{log_path}/log_{log_index}_{log_date}-{log_time}.ulg'

files = os.listdir(log_path)

for file in files:
    log_file = f'{log_path}/{file}'
    print(log_file)
    avg_hovering_rpm = avghover(log_file)
    addrpm(log_file,avg_hovering_rpm)

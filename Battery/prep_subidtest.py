import sys
sys.path.append('/Users/lucas/Documents/Travail/Yuneec/LogAnalysis')
from analog import logextract
import csv

# path = '/home/lucas/Documents/Travail/Yuneec/Logs/Snow Orange (Battery 9) z0=1/log_195_2019-9-26-14-05-18.ulg'
path = '/home/lucas/Documents/Log_Analysis/Logs/Snow Orange (Battery 9) z0=1/log_195_2019-9-26-14-05-18.ulg'
info = logextract(path,'battery_status')
u = info['battery_current']
y = info['battery_voltage']

csv_file = open('Battery 9/log_195.csv','w')
writer = csv.writer(csv_file)

writer.writerow(['u','y'])
for k in range(len(u)):
    writer.writerow([u[k],y[k]])


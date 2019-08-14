import pyulog
from matplotlib import pyplot as plt
import os

log_path = '/home/lucas/src/px4/Firmware/build/px4_sitl_default/tmp/rootfs/log'

log_date = '2019-08-14'
log_time = '13_14_23'
log_file = f'{log_path}/{log_date}/{log_time}.ulg'

ulog = pyulog.ULog(log_file,'vehicle_local_position') # creates a ULog object 
datalist = ulog.data_list # is a list of Data objects, which contain the final topic data for a single topic and instance

value = datalist[0].field_data # is a list of FieldData objects
data = datalist[0].data # is a dictionnary
 
time = data['timestamp']
acc_z = data['az']

# a reasonable threshold for strong vibration is anything with a peak-to-peak of more than 2-3 m/s/s 
# (http://docs.px4.io/master/en/log/flight_log_analysis.html)

threshold = 3 # m/s^2
threshold_list = [threshold for timestamp in time]

plt.plot(time,threshold_list,'r:')
plt.plot(time,[-elem for elem in threshold_list],'r:')
plt.plot(time,acc_z)
plt.title(f'Vertical vibrations for log {log_date}/{log_time}')
plt.xlabel('time')
plt.ylabel('local z acceleration')
plt.grid()

savedir = f'/home/lucas/Documents/Log_Analysis/Vibrations/Figures/{log_date}'

if not os.path.isdir(savedir):
    os.makedirs(savedir)
    print(f'The fodler {savedir} has been created.')
plt.savefig(f'{savedir}/z_acc_{log_time}.png')

plt.show()

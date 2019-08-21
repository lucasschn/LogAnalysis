from analog import logextract as logextract, ishover as ishover
import numpy as np
import pandas as  pd


# for logs downloaded with QGC
log_path = '/home/lucas/Documents/Log_Analysis/Vibrations/Logs'
log_index = '270'
log_date = '2019-8-19'
log_time = '14-20-40'
log_file = f'{log_path}/log_{log_index}_{log_date}-{log_time}.ulg'

topic_list = []
topic_list.append('sensor_combined')
topic_list.append('actuator_outputs')
topic_list.append('vehicle_local_position_setpoint')
topic_list.append('vehicle_status')
topic_list.append('vehicle_land_detected')
topic_list.append('manual_control_setpoint')

info = logextract(log_file,topic_list)
rpm = info['rpm'] 
hovering_index = []
hovering_rpm = []

print(rpm)

for k in range(len(info['time_vs'])):
    hovering_index.append(ishover(info['navstate'][k],info['stick_in_x'][k],info['stick_in_y'][k],info['stick_in_z'][k]))
    hovering_rpm.append([rpm[i][k] for i in range(len(rpm)) if hovering_index[k]]) 
# arrange the sampling time problem
avg_hovering_rpm = np.mean(hovering_rpm)
    
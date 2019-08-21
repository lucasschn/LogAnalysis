import pyulog
import numpy as np
import pandas as pd

def ampspectrum(x):
    """ Returns the magnitudes of the fft for a real x"""
    complex_spectrum = rfft(x)
    X = np.abs(complex_spectrum)
    return X

def ishover(navstate,stick_in_x,stick_in_y,stick_in_z):
    """ Takes navstate and stick inputs at a certain timestep as arguments. Returns True is the drone is hovering at this moment, False elsewise"""
    ishover = False
    if navstate == 4: # Auto-loiter
        ishover = True 
    elif navstate == 2 and [stick_in_x, stick_in_y, stick_in_z] == [0,0,0]: # Position control
        ishover = True
    return ishover


def logextract(path,topic_list):
    """ Returns an info dictionnary containing the relevant information in the log "path" according to "topic_list" """
    ulog = pyulog.ULog(path,topic_list) 
    datalist = ulog.data_list # is a list of Data objects, which contain the final topic data for a single topic and instance
    info = {}
    for topic in datalist: 
        if topic.name == 'sensor_combined': 
            data_sc = topic.data
            time_sc = data_sc['timestamp']/1e6 # convert it from us to s
            acc_x=data_sc['accelerometer_m_s2[0]']
            acc_y=data_sc['accelerometer_m_s2[1]']
            acc_z=data_sc['accelerometer_m_s2[2]']
            roll = data_sc['gyro_rad[0]']
            pitch = data_sc['gyro_rad[1]']
            yaw = data_sc['gyro_rad[2]']
            info.update({'time_sc':time_sc, 'acc_x':acc_x, 'acc_y':acc_y,'acc_z':acc_z, 'roll': roll, 'pitch': pitch, 'yaw': yaw})
        elif topic.name == 'actuator_outputs':
                if (np.all(topic.data['noutputs'] <= 1)): 
                    continue
                else:                     
                    data_ao = topic.data
                    # columns = ['motor 1','motor 2','motor 3','motor 4','motor 5','motor 6']
                    rpm=[]
                    for k in range(np.unique(data_ao['noutputs']).item()):
                        rpm.insert(k,data_ao[f'output[{k}]'])
                    # rpm_df = pd.DataFrame(rpm,range(len(time_ao)),columns)
                    info.update({'rpm': rpm})
        elif topic.name == 'vehicle_local_position_setpoint':
            data_vlps = topic.data
            vert_thrust = data_vlps['thrust[2]']
        elif topic.name == 'vehicle_status':
            data_vs = topic.data
            time_vs = data_vs['timestamp']/1e6
            navstate = data_vs['nav_state']
            info.update({'time_vs': time_vs,'navstate': navstate})
        elif topic.name == 'vehicle_land_detected':
            data_vld = topic.data
        elif topic.name == 'manual_control_setpoint':
            data_mcs = topic.data
            time_mcs = data_mcs['timestamp']/1e6
            stick_in_x = data_mcs['x']
            stick_in_y = data_mcs['y']
            stick_in_z = data_mcs['z']
            info.update({'stick_in_x': stick_in_x, 'stick_in_y': stick_in_y, 'stick_in_z': stick_in_z})
    return info


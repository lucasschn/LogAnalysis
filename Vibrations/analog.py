import pyulog
import numpy as np
import pandas as pd
from pyulgresample.ulogdataframe import DfUlg, TopicMsgs

def ampspectrum(x):
    """ Returns the magnitudes of the fft for a real x"""
    complex_spectrum = rfft(x)
    X = np.abs(complex_spectrum)
    return X

def avghover(path):
    topic_list = []
    topic_list.append('vehicle_status')
    topic_list.append('actuator_outputs')
    topic_list.append('manual_control_setpoint')

    dfulg = DfUlg.create(path,topic_list)

    msglist = []
    for k in range(6):
        msglist.append(f'F_output_{k}')

    rpm1 = dfulg.df[f'T_actuator_outputs_0__{msglist[0]}']
    rpm2 = dfulg.df[f'T_actuator_outputs_0__{msglist[1]}']
    rpm3 = dfulg.df[f'T_actuator_outputs_0__{msglist[2]}']
    rpm4 = dfulg.df[f'T_actuator_outputs_0__{msglist[3]}']
    rpm5 = dfulg.df[f'T_actuator_outputs_0__{msglist[4]}']
    rpm6 = dfulg.df[f'T_actuator_outputs_0__{msglist[5]}']

    stick_in_x = dfulg.df['T_manual_control_setpoint_0__F_x']
    stick_in_y = dfulg.df['T_manual_control_setpoint_0__F_y']
    stick_in_z = dfulg.df['T_manual_control_setpoint_0__F_z']

    navstate = dfulg.df['T_vehicle_status_0__F_nav_state']


    hovering_bool=[]

    for k in range(len(navstate)):
        hovering_bool.append(ishover(navstate[k],stick_in_x[k],stick_in_y[k],stick_in_z[k]))

    average_hovering_rpm1 = np.mean(rpm1[hovering_bool])
    average_hovering_rpm2 = np.mean(rpm2[hovering_bool])
    average_hovering_rpm3 = np.mean(rpm3[hovering_bool])
    average_hovering_rpm4 = np.mean(rpm4[hovering_bool])
    average_hovering_rpm5 = np.mean(rpm5[hovering_bool])
    average_hovering_rpm6 = np.mean(rpm6[hovering_bool])

    average_hovering_rpm=np.mean([average_hovering_rpm1,average_hovering_rpm2,average_hovering_rpm3,average_hovering_rpm4,average_hovering_rpm5,average_hovering_rpm6])

    return average_hovering_rpm


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
                    time_ao = data_ao['timestamp']
                    columns = ['motor 1','motor 2','motor 3','motor 4','motor 5','motor 6']
                    rpm=[]
                    for k in range(np.unique(data_ao['noutputs']).item()):
                        rpm.insert(k,data_ao[f'output[{k}]'])
                    rpm_df = pd.DataFrame(np.array([rpm[0],rpm[1],rpm[2],rpm[3],rpm[4],rpm[5]]).transpose(),index=range(len(time_ao)),columns=columns)
                    info.update({'time_ao':time_ao,'rpm': rpm,'rpm_df': rpm_df})
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
            info.update({'time_mcs':time_mcs,'stick_in_x': stick_in_x, 'stick_in_y': stick_in_y, 'stick_in_z': stick_in_z})
    return info
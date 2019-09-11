import sys

sys.path.append('/home/lucas/Documents/Log_Analysis')
import analog
import numpy as np
import os
import csv 


class Payload_detecter: 

    def __init__(self):
        self.rpm = []
        self.thrust = []
    

    def create_database(self,folder):
        ''' Reinitilizes the hovering statistics and creates new ones based on all valid logs found in folder.'''
        self._init()
        files = os.listdir(folder)
        for file in files:
            try :
                avg_hovering = analog.avghover(log_file)
                update_database(log_file,avg_hovering)
                print(f'{log_file} added to database')
            except LogError:
                print(f'{log_file} is not relevant. Discarded.')
                continue


    def test_log(self,test_file):
        payload = False
        try : 
            test_results = analog.avghover(test_file)
            if -100*avg_hovering['zthrust'] > self.mean_trhust + self.std_thrust:
                payload = True
            return payload
        except LogError:
            print('There is not enough hovering in this file to give results.')


    def update_database(self,log_file,avg_hovering):
        ''' Adds hovering statistics of all valid logs found in folder to the instance attributes.'''
        if log_file not in self.logfiles:
            self.rpm.append(rpm)
            self.thrust.append(thrust)
            self.mean_rpm = np.mean(self.rpm)
            self.mean_thrust = np.mean(self.thrust)
            self.std_thrust = np.std(self.thrust)
        else: 
            print(f'{log_file} already found in the database.')


    def write_database_to_csv(self,path2csv='avg_hover.csv'):
        ''' Writes the hovering results to csv file. '''
        csv_file = open(path2csv,'w')
        writer = csv.writer(csv_file)
        writer.writerow(['Log File','Avg rpm','Avg vert thrust'])
        writer.writerow([self.log_file,self.rpm,self.thrust])

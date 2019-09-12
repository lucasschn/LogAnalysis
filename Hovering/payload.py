import sys

sys.path.append('/home/lucas/Documents/Log_Analysis')
import analog
import numpy as np
import os
import csv 
from analog import LogError

class Payload_detecter: 

    def __init__(self):
        self.logfiles = []
        self.rpm = []
        self.thrust = []
    

    def create_database(self,folder):
        ''' Reinitilizes the hovering statistics and creates new ones based on all valid logs found in folder.'''
        self.__init__()
        files = os.listdir(folder)
        for file in files:
            log_file = f'{folder}/{file}'
            try :
                avg_hovering = analog.avghover(log_file)
                self.update_database(log_file,avg_hovering)
                print(f'{log_file} added to database')
            except LogError:
                print(f'{log_file} is not relevant. Discarded.')
                continue

    def compute_stats(self):
        self.mean_rpm = np.mean(self.rpm)
        self.mean_thrust = np.mean(self.thrust)
        self.std_thrust = np.std(self.thrust)

    def export_database(self,path2csv='avg_hover.csv'):
        ''' Writes the hovering results to csv file. '''
        csv_file = open(path2csv,'w')
        writer = csv.writer(csv_file)
        writer.writerow(['Log File','Avg rpm','Avg vert thrust'])
        writer.writerow([self.logfiles,self.rpm,self.thrust])


    def import_database(self,path2csv='avg_hover.csv'):
        ''' Reads the hovering results from a csv file. Erases previously imported data. '''
        self.__init__()
        csv_file = open(path2csv,'r')
        reader = csv.DictReader(csv_file)
        for line in reader:
            self.rpm.append(float(line['Avg rpm']))
            self.thrust.append(-100*float(line['Avg vert thrust']))
        self.compute_stats()



    def test_log(self,test_file):
        waspayload = False
        try : 
            test_results = analog.avghover(test_file)
            if -100*test_results['zthrust'] > self.mean_thrust + self.std_thrust:
                waspayload = True
            return waspayload
        except LogError:
            print('There is not enough hovering in this file to give results.')


    def update_database(self,log_file,avg_hovering):
        ''' Adds hovering statistics of all valid logs found in folder to the instance attributes.'''
        if log_file not in self.logfiles:
            self.logfiles.append(log_file)
            self.rpm.append(avg_hovering['rpm'])
            self.thrust.append(avg_hovering['zthrust'])
            self.compute_stats()
        else: 
            print(f'{log_file} already found in the database.')



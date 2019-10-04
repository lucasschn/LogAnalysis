import numpy as np
import pandas as pd
import os
import sys
sys.path.append('/home/lucas/Documents/Log_Analysis/Battery')
#sys.path.append('/Users/Lucas/Documents/Travail/Yuneec/LogAnalysis')
import analog
import battery
from scipy import optimize

# Battery parameters
curve = battery.OCVcurve('Battery 9/Discharge 200mA/SOCvsOCV_discharge200mA.csv')
z0 = 1
iR10 = 0
Q = 6000*3.6
eta = 1

# Test with polynomial curve
p = np.polyfit(curve.SOC,curve.OCV,11)

# define logs to optimize on
folder = '/home/lucas/Documents/Log_Analysis/Logs/Snow Orange (Battery 9) z0=1'
#folder ='/Users/Lucas/Documents/Travail/Yuneec/Logs/Snow Orange (Battery 9) z0=1'
files = os.listdir(folder)

time = np.array([])
current = np.array([])
voltage = np.array([])
tstoplist = [0]
for file in sorted(files):
    print(file)
    info = analog.logextract(f'{folder}/{file}','battery_status')
    time = np.append(time,info['time_bs']- info['time_bs'][0] + tstoplist[-1])
    tstoplist.append(time[-1])
    current = np.append(current,info['battery_current'])
    voltage = np.append(voltage,info['battery_voltage']/4)
    
n = len(current)

# initial guess
ECparams = pd.read_csv('ECparams.csv')
x0 = np.array([ECparams['R0'], ECparams['R1'], ECparams['C1']])
initbat = battery.Thevenin(z0,Q,curve,ECparams['R0'],ECparams['R1'],ECparams['C1'])
vsim = initbat.simulate(time,current,curve)
x0 = np.append(x0,initbat.simz)
x0 = np.append(x0,initbat.simi1)
x0 = np.append(x0,initbat.simv)
prev_x = x0
print('Inititalizing optimization algortithm ...')

# function to be minimized
def rmserror(x,voltage,prev_x):
    y = x[-n:]
    rmserror = (np.mean((y-voltage)**2))**.5
    print(x-prev_x)
    prev_x = x
    return rmserror


def print_res(x,prev_x):
    print(x-prev_x)
    prev_x = x
    return False

# definition of the constraints
# x = [R0, R1, C1, z[0], ... , z[n], iR1[0], ..., iR1[n], y[0], ... ,y[n]]
cons = []

def initstate1_eq(x):
    return x[3]-z0

def initstate2_eq(x):
    return x[n+4]
    
con1 = {'type':'eq','fun':initstate1_eq}   
con2 = {'type':'eq','fun':initstate2_eq}    
cons.append(con1)
cons.append(con2)

for k in range(n):

    def state1_eq(x):
        #print('working ...')
        return x[k+4]-x[k+3]+eta*dt/Q*current[k]
    
    def state2_eq(x):
        #print('working ...')
        return x[n+k+5]-np.exp(-dt/(x[1]*x[2]))*x[k+n+4]-(1-np.exp(-dt/(x[1]*x[2])))*current[k]
    
    def output_eq(x):
        #print('working ...')
        #return x[2*n+k+5]-curve.OCVfromSOC(x[k+3]) + x[1]*x[n+k+4] + x[0]*current[k]
        return x[2*n+k+5]- np.polyval(p,x[k+3]) + x[1]*x[n+k+4] + x[0]*current[k]
    
    con1 = {'type':'eq','fun':state1_eq}   
    con2 = {'type':'eq','fun':state2_eq}   
    con3 = {'type':'eq','fun':output_eq}   
    cons.append(con1)
    cons.append(con2)
    cons.append(con3)

# definition of the bounds (the parameters and the variables cannot be negative)
bnd = tuple((0,None) for _ in range(len(x0)))

dt = np.mean(np.diff(time))

# maximum of iterations
opt = {'maxiter':1,'disp':True}
results = optimize.minimize(rmserror,x0,args=(voltage, prev_x),method='SLSQP',constraints=cons,options=opt, callback=print_res, bounds=bnd)

print(results)
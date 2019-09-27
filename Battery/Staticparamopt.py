import numpy as np
import pandas as pd
import os
import sys
sys.path.append('/home/lucas/Documents/Log_Analysis/Battery')
import analog
import battery
from scipy import optimize
from scipy.optimize import NonlinearConstraint as nlcon

# Battery parameters
curve = battery.OCVcurve('Battery 9/Discharge 200mA/SOCvsOCV_discharge200mA.csv')
z0 = 1
iR10 = 0
Q = 6000*3.6
eta = 1

# define logs to optimize on
folder = '/home/lucas/Documents/Log_Analysis/Logs/Jack Sparrow (Luigi) z0=1'
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

# function to be minimized
def rmserror(x):
    y = x[-n:]
    rmserror = (np.mean((y-voltage)**2))**.5
    return rmserror

# initial guess
ECparams = pd.read_csv('ECparams.csv')
x0 = np.array([ECparams['R0'], ECparams['R1'], ECparams['C1']])
initbat = battery.Thevenin(z0,Q,curve,ECparams['R0'],ECparams['R1'],ECparams['C1'])
vsim = initbat.simulate(time,current,curve)
x0 = np.append(x0,initbat.simz)
x0 = np.append(x0,initbat.simi1)
x0 = np.append(x0,initbat.simv)


# definition of the constraints
# x = [R0, R1, C1, z[0], ... , z[n], iR1[0], ..., iR1[n], y[0], ... ,y[n]]
cons = []
for k in range(n-1):

    def state1_eq(x):
        print('working ...')
        return x[k+4]-x[k+3]+eta*dt/Q*current[k]
    
    def state2_eq(x):
        print('working ...')
        return x[k+(n+1)+4]-np.exp(-dt/(x[1]*x[2]))*x[k+(n+1)+3]-(1-np.exp(-dt/(x[1]*x[2])))*current[k]
    
    def output_eq(x):
        print('working ...')
        return curve.OCVfromSOC(x[k+3])-x[1]*x[k+(n+1)+3]-x[0]*current[k]-x[-n+k]
    
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
opt = {'maxiter':30}
results = optimize.minimize(rmserror,x0,method='SLSQP',constraints=cons,options=opt, bounds=bnd)

print(results)
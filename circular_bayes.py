#! /usr/bin/python2

import numpy as np
from scipy.stats import vonmises
from thllib import flylib
from matplotlib import pyplot as plt
import sys
from scipy.ndimage import convolve
from scipy import stats
from scipy.signal import correlate

flynum = int(sys.argv[1])

fly = flylib.NetFly(flynum)

fly.open_signals('hdf5')

angles = np.linspace(0,360,360*2)[:-1]

vel_kappa = 10.0 #vonmises kappa applied to velocity prediction in baysian filter, higher values will favor prediction.
state_kappa = 0.01

#vel_kernel = vonmises.pdf(np.deg2rad(angles)-np.pi,vel_kappa)
#vel_kernel /= np.sum(vel_kernel)
#state_kernel = vonmises.pdf(np.deg2rad(angles)-np.pi,state_kappa)
#state_kernel = None #/= np.sum(state_kernel)

def fly_pdf(x,mu,k1,k2):
    return (stats.vonmises.pdf(x,k1,loc = mu) + stats.vonmises.pdf(x,k2,loc = mu+np.pi))/2.0

angles = np.linspace(0,360,360*2)[:-1]
#state_kernel = fly_pdf(np.deg2rad(angles),np.pi,100,1.5)
vel_kernel = vonmises.pdf(np.deg2rad(angles),vel_kappa,loc = np.pi)#fly_pdf(np.deg2rad(angles),np.pi,100.0,0.01)

#imkernel = np.vstack([state_kernel])
observed_values = np.array(fly.correlations)#convolve(fly.correlations,imkernel,mode = 'wrap')

state_estimates = list()
velocities = list()
predictions = list()

vel_kernel = vonmises.pdf(np.deg2rad(angles),vel_kappa,loc = np.pi)

velocities = []
for i in range(len(observed_values)-1):
    velocities.append(np.argmax(correlate(observed_values[i],observed_values[i+1])))

prediction = np.roll(observed_values[0],velocities[0])
prediction = convolve(prediction,vel_kernel,mode = 'wrap')
prediction /= np.sum(prediction)
posterior = prediction*observed_values[1]
posterior /= np.sum(posterior)  
predictions.append(prediction)
state_estimates.append(posterior)
i = 0
for observation,velocity in zip(observed_values[1:],velocities):
    prediction = np.roll(state_estimates[-1],velocity)
    prediction = convolve(prediction,vel_kernel,mode = 'wrap')
    prediction /= np.sum(prediction)
    posterior = prediction*observation
    posterior /= np.sum(posterior)
    predictions.append(prediction)
    state_estimates.append(posterior)
    i+=1
    if not((i%200)>0):
        print i
        
        
fly.save_hdf5(np.array(velocities),'pos_shifts',overwrite = True)
fly.save_hdf5(np.array(state_estimates),'state_estimates',overwrite = True)
fly.save_hdf5(np.array(predictions),'predictions',overwrite = True)
import numpy as np
from scipy.stats import vonmises
from thllib import flylib
from matplotlib import pyplot as plt
import sys
from scipy.ndimage import convolve
from scipy import stats

flynum = int(sys.argv[1])

fly = flylib.NetFly(flynum)

fly.open_signals('hdf5')

angles = np.linspace(0,360,360*2)[:-1]

vel_kappa = 5.0 #vonmises kappa applied to velocity prediction in baysian filter, higher values will favor slow transitions.
state_kappa = 0.01

#vel_kernel = vonmises.pdf(np.deg2rad(angles)-np.pi,vel_kappa)
#vel_kernel /= np.sum(vel_kernel)
#state_kernel = vonmises.pdf(np.deg2rad(angles)-np.pi,state_kappa)
#state_kernel = None #/= np.sum(state_kernel)

def fly_pdf(x,mu,k1,k2):
    return (stats.vonmises.pdf(x,k1,loc = mu) + stats.vonmises.pdf(x,k2,loc = mu+np.pi))/2.0

angles = np.linspace(0,360,360*2)[:-1]
state_kernel = fly_pdf(np.deg2rad(angles),np.pi,100,1.5)
vel_kernel = fly_pdf(np.deg2rad(angles),np.pi,100.0,0.01)

imkernel = np.vstack([state_kernel])
observed_values = convolve(fly.correlations,imkernel,mode = 'wrap')

state_kernel = None#state_kernel = fly_pdf(np.deg2rad(angles),np.pi,100,1.5)
vel_kernel = fly_pdf(np.deg2rad(angles),np.pi,100.0,0.01)

def predict(state_pdf,velocity,kernel):
    p = convolve(np.roll(state_pdf, velocity), kernel, mode='wrap')
    return p/np.sum(p)

def estimate_state(prediction,observation,kernel = None):
    """estimate the state given a observation likelyhood and prediction likelyhood
    and a kernel describing the state confidence"""
    if kernel is None:
        return observation*prediction
    else:
        return convolve(observation,kernel)*prediction

def wraped_descrete_vel(like_1,like_2):
    """estimate the descrete velocity of a transition from
    likelyhood1 like_1 to likelyhood2 like2"""
    from scipy.signal import correlate
    return np.argmax(correlate(like_1,like_2))-len(angles)

#observed_values = fly.correlations
#correlations_smooth = gaussian_filter(fly.correlations,(0.3,20),mode = 'wrap')
#observed_values = correlations_smooth

state_estimates = list()
velocities = list()
predictions = list()

idx_0 = 0
l0 = observed_values[idx_0]
l1 = observed_values[idx_0+1]
v = wraped_descrete_vel(l0,l1)
predictions.append(predict(l1/np.sum(l1),
                           v,vel_kernel))

state_estimates.append(estimate_state(predictions[-1],
                                      observed_values[idx_0+2],
                                      kernel = state_kernel))

velocities.append(wraped_descrete_vel(l1/np.sum(l1),
                                      state_estimates[-1]))

for i in range(len(observed_values)-2):
    predictions.append(predict(state_estimates[-1],
                               velocities[-1],
                               vel_kernel))
    state_estimates.append(estimate_state(predictions[-1],
                                          observed_values[idx_0+2+i],
                                          kernel = state_kernel))
    velocities.append(wraped_descrete_vel(state_estimates[-2],
                                          state_estimates[-1]))
    if not((i%200)>0):
        print i
        
        
fly.save_hdf5(np.array(velocities),'pos_shifts',overwrite = True)
fly.save_hdf5(np.array(state_estimates),'state_estimates',overwrite = True)
fly.save_hdf5(np.array(predictions),'predictions',overwrite = True)
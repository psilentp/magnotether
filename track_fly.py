from scipy import ndimage
from thllib import flylib
import numpy as np
import h5py
import cv2
import sys
#from matplotlib import pyplot as plt

flynum = int(sys.argv[1])#

###### parameters ######
light_on_thresh = 10 #threshold to use to find the begining of experiment. Average pixel value.
flash_offset = 50 #deal with the fact that the camera is over-exposed when the light first comes on
angles = np.linspace(0,360,360*2)[:-1] #resoluton of the angular tracking, larger values yeald slower tracking

fly = flylib.NetFly(flynum)
fly.open_signals('hdf5')

def rotate_image(image, angle,center = None):
    row,col = image.shape
    if center is None:
        center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row),borderMode = cv2.BORDER_REPLICATE)
    return new_image

captured_frames = np.squeeze(np.argwhere(np.array(fly.avepxl)>light_on_thresh))

first_frame = fly.images[captured_frames[0]+flash_offset]

center = (first_frame.shape[0]/2,first_frame.shape[1]/2)

template = list()
for i in angles:
    template.append(cv2.pyrDown(cv2.pyrDown(rotate_image(first_frame,i,center = center))))

import time
class timer():
    def __init__(self):
        self.last_time = time.time()
    
    def dt(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        return dt

frame_size = len(first_frame.ravel())

tmr = timer()
#imt = img_trim
orientations = list()
correlations = list()

load_image = list()
dot_product = list()
rest_of_loop = list()

for idx in captured_frames:
    im = cv2.pyrDown(cv2.pyrDown(fly.images[idx])).ravel()
    load_image.append(tmr.dt())
    corvals = [np.dot(t.ravel(),im) for t in template]
    dot_product.append(tmr.dt())
    tidx = np.argmax(corvals)
    correlations.append(corvals)
    orientations.append(angles[tidx])
    if not((idx%200)>0):
        #from IPython import display
        print idx,angles[tidx]
    rest_of_loop.append(tmr.dt())

fly.save_hdf5(np.array(orientations),'orientations',overwrite = True)
fly.save_hdf5(np.array(correlations),'correlations',overwrite = True)
fly.save_hdf5(np.array(load_image),'profile_load_image',overwrite = True)
fly.save_hdf5(np.array(dot_product),'profile_dot_product',overwrite = True)
fly.save_hdf5(np.array(rest_of_loop),'profile_rest_of_loop',overwrite = True)
fly.save_hdf5(angles,'angles',overwrite = True)
fly.save_txt(str(first_frame),'first_frame')
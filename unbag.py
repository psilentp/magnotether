#! /usr/bin/python2

import rosbag
from thllib import flylib
import numpy as np
import os
import h5py
import cv2
import sys
from matplotlib import pyplot as plt
from scipy import ndimage

#%matplotlib inline
#from matplotlib import pyplot as plt

flynum = int(sys.argv[1])#1351

samp_range = 5000 # search over this range to find center point of image
search_bound = 50 # window to search for center of fly in pxls
crop = 100 #crop fly around this window
img_thresh = 240 #threshold to find the center of rotation of the fly.
cent_fun = np.mean #function to use when looking for the image center

fly = flylib.NetFly(flynum)
bag_file_name = fly.flypath + '/' + [bf for bf in os.listdir(fly.flypath) if '.bag' in bf][0]
inbag = rosbag.Bag(bag_file_name)
topics = inbag.get_type_and_topic_info()[1].keys()

if os.path.exists(fly.flypath + '/' + 'images.hdf5'):
    os.remove(fly.flypath + '/' + 'images.hdf5')
if os.path.exists(fly.flypath + '/' + 'times.hdf5'):
    os.remove(fly.flypath + '/' + 'times.hdf5')
if os.path.exists(fly.flypath + '/' + 'avepxl.hdf5'):
    os.remove(fly.flypath + '/' + 'avepxl.hdf5')
    
images = h5py.File(fly.flypath + '/' + 'images.hdf5')
times = h5py.File(fly.flypath + '/' + 'times.hdf5')
avepxl = h5py.File(fly.flypath + '/' + 'avepxl.hdf5')

img_msgs = [(topic,msg,t) for topic,msg,t in inbag.read_messages(topics = '/camera/image_raw/compressed')]

#samp_range = len(msgs)
#mid_trial = len(img_msgs)/2 
s_start = 0#mid_trial - samp_range/2
s_end = len(img_msgs)#mid_trial + samp_range/2


def decode_img_msg(msg):
    import cv2
    img_np_arr = np.fromstring(msg[1].data, np.uint8)
    img = cv2.imdecode(img_np_arr,cv2.IMREAD_GRAYSCALE)
    img = cv2.pyrDown(img)
    return img#[imt[0]:imt[1],imt[2]:imt[3]]

imgs = [decode_img_msg(img_msgs[i]) for i in sorted(np.random.choice(np.arange(s_start,s_end),
                                                         size = samp_range,
                                                         replace = False))]


center_img = (255-cent_fun(imgs,axis = 0))
th_img = (center_img) > 230
r,c = center_img.shape
cc,rc = ndimage.center_of_mass(th_img[(r/2)-search_bound:(r/2)+search_bound,(c/2)-search_bound:(c/2)+search_bound])
center = (rc+(c/2)-search_bound,cc+(r/2)-search_bound)

print center

rl,rr,ct,cb = tuple([int(p) for p in [center[1]-crop,
                                center[1]+crop,
                                center[0]-crop,
                                center[0]+crop]])
print rl,rr,ct,cb
first_img = decode_img_msg(img_msgs[0])[rl:rr,ct:cb]
#mid_img = decode_img_msg(img_msgs[mid_trial])
#plt.imshow(mid_img[rl:rr,ct:cb])
#plt.show()

images.create_dataset('images',shape = (len(img_msgs),) + first_img.shape)
times.create_dataset('times',shape = (len(img_msgs),),dtype = 'f')
avepxl.create_dataset('avepxl',shape = (len(img_msgs),),dtype = 'f')
t0 = img_msgs[0][1].header.stamp.to_time()

for i,msg in enumerate(img_msgs):
    if not((i%100)>0):
        print i
    im_array = decode_img_msg(msg)[rl:rr,ct:cb]
    images['images'][i] = im_array
    times['times'][i] = msg[1].header.stamp.to_time() - t0
    avepxl['avepxl'][i] = np.mean(im_array)
    
cond_msgs = [(topic,msg,t) for topic,msg,t in inbag.read_messages(topics = '/magnotether/condition')]
cstr = '\n'.join(['\t'.join([str(cmsg[1].data),str(cmsg[2].to_time()-t0)]) for cmsg in cond_msgs])

for h5file in [images,times,avepxl]:
    h5file.flush()
    h5file.close()
    
fly.save_txt(cstr,'exp_condition.txt')
    
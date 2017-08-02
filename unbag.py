import rosbag
from thllib import flylib
import numpy as np
import os
import h5py
import cv2
import sys
from matplotlib import pyplot as plt
from scipy import ndimage

flynum = int(sys.argv[1])#1328
samp_range = 5000 # search over this range to find center point of image
search_bound = 75 # window to search for center of fly in pxls
crop = 200 #crop fly around this window

fly = flylib.NetFly(flynum)
bag_file_name = fly.flypath + '/' + [bf for bf in os.listdir(fly.flypath) if '.bag' in bf][0]
inbag = rosbag.Bag(bag_file_name)
topics = inbag.get_type_and_topic_info()[1].keys()

images = h5py.File(fly.flypath + '/' + 'images.hdf5')
times = h5py.File(fly.flypath + '/' + 'times.hdf5')
avepxl = h5py.File(fly.flypath + '/' + 'avepxl.hdf5')

img_msgs = [(topic,msg,t) for topic,msg,t in inbag.read_messages(topics = topics)]

mid_trial = len(img_msgs)/2 
s_start = mid_trial - samp_range/2
s_end = mid_trial + samp_range/2

def decode_img_msg(msg):
    import cv2
    img_np_arr = np.fromstring(msg[1].data, np.uint8)
    img = cv2.imdecode(img_np_arr,cv2.IMREAD_GRAYSCALE)
    return img#[imt[0]:imt[1],imt[2]:imt[3]]

imgs = [decode_img_msg(img_msgs[i]) for i in sorted(np.random.choice(np.arange(s_start,s_end),
                                                         size = 200,
                                                         replace = False))]
max_img = (255-np.max(imgs,axis = 0))
th_img = max_img >50
r,c = max_img.shape
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
times.create_dataset('times',shape = (len(img_msgs),))
avepxl.create_dataset('avepxl',shape = (len(img_msgs),))

for i,msg in enumerate(img_msgs):
    if not((i%100)>0):
        print i
    im_array = decode_img_msg(msg)[rl:rr,ct:cb]
    images['images'][i] = im_array
    times['times'][i] = msg[1].header.stamp.to_time()
    avepxl['avepxl'][i] = np.mean(im_array)

for h5file in [images,times,avepxl]:
    h5file.flush()
    h5file.close()
from matplotlib import animation
import numpy as np
from matplotlib import pyplot as plt
from thllib import flylib as flb
import figurefirst as fifi
import sys

reload(fifi)
flynum = int(sys.argv[1])#1328
fly = flb.NetFly(flynum)
fly.open_signals(['hdf5','txt'])
position = fly.position
velocity = fly.velocity

layout = fifi.FigureLayout('fig_plot.svg')
layout.make_mplfigures()

idx = 5000
window = 100

#positions = angles[np.argmax(filt_cor,axis=1)]
first_frame = np.squeeze(np.argwhere(np.array(fly.avepxl)>10)[0]) + 1

fig = plt.gcf()

img = layout.axes['image'].imshow(fly.images[idx+first_frame],cmap = plt.cm.gray,vmin = 0,vmax = 255)
layout.axes['image'].axis('off')
layout.axes['image'].set_zorder(-10)
#ax2 = fig.add_subplot(1,1,1,projection = 'polar',frameon = False)
#ax2.patch.set_alpha(0.0)
ln = layout.axes['polar'].plot([position[idx],position[idx]],[0,0.5],'-o')
layout.axes['polar'].patch.set_alpha(0.0)
layout.axes['polar'].set_ybound(0,1)
layout.axes['polar']['axis'].set_frame_on(False)
fn = layout.axes['framenum'].text(0,0,str(0),fontsize = 20)

ptrace = layout.axes['position'].plot(position[idx-window:idx+window])
vtrace = layout.axes['velocity'].plot(velocity[idx-window:idx+window])
layout.axes['position'].axvline(window,color = 'k')
layout.axes['velocity'].axvline(window,color = 'k')

layout.axes['position'].set_ybound(0,2*np.pi)
layout.axes['velocity'].set_ybound(np.percentile(velocity,0.5),np.percentile(velocity,99.5))
layout.axes['position'].set_yticks(np.linspace(0,2*np.pi,3))
layout.axes['position'].tick_params(axis='both', which='major', labelsize=6)
layout.axes['velocity'].set_yticks(np.linspace(np.percentile(velocity,0.5),np.percentile(velocity,99.5),3))
layout.axes['velocity'].tick_params(axis='both', which='major', labelsize=6)
layout.axes['velocity'].ticklabel_format()
fifi.mpl_functions.set_spines(layout)

mpath = fly.flypath + '/' + 'tracking_test.mp4'
def animate(idx):
    l = np.maximum(idx-window,0)
    r = np.minimum(idx+window,len(position))
    img.set_data(fly.images[idx+first_frame-1])
    ln[0].set_data([position[idx],position[idx]],[0,0.5])
    ptrace[0].set_data(np.arange(0,r-l),position[l:r])
    vtrace[0].set_data(np.arange(0,r-l),velocity[l:r])
    fn.set_text(str(idx))
    return img,ln[0],
    if not((idx%200)>0):
            print idx
            
anim = animation.FuncAnimation(fig, animate,frames=np.arange(100,15000), interval=20,blit = True)
anim.save(mpath, fps=30, extra_args=['-vcodec', 'h264', '-pix_fmt','yuv420p'])
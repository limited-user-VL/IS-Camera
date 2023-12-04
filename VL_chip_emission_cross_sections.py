import numpy as np
from matplotlib import pyplot as plt
# import time
import datetime
import os
import imageio
# from IPython import display
from IPython import get_ipython
from pathlib import Path
import sys
import site
from getmac import get_mac_address as gma


# add VL api to pythonpath
root = Path(r'G:\Shared drives\VitreaLab Share')
api_folder = root / 'Lab Software' / 'api_vitrealab'
if api_folder not in sys.path:
    site.addsitedir(str(api_folder))
#from camera_control.tis_camera import TISCamera
from camera_control.tis_camera_cmos import TISCamera
import Kinesis.KDC101_Translation_Class as KDC101_Class 

# reads mac address to identify PC
mac = gma()

#%% set serial numbers
# characterisation setup right
if mac == '18:c0:4d:26:9c:53':
    print("Using right PC")
    # define SN
    SN_LTS = 45158784  # stage LTS300 - long axis (X)
    SN_KDC = 27004871  # stage KCube DC Motor
    SN_PM = 'USB0::0x1313::0x8072::1907998::INSTR'  # bus PM
    SN_CAMERA = '16910441'
    SN_CAMERA = "41220696" #DMM 37UX265-ML


#Connect stage
z_stage = KDC101_Class.KDC101_translation(SN_KDC)


# Connect camera
cam1 = None
del cam1
cam1 = TISCamera(SN_CAMERA) #monochrome, SN 28020401

cam1.set_format('Y16 (1280x1024)')
cam1.set_format('Y16 (640x480)')
cam1.set_format('Y16 (2048x1536)')
print(cam1.get_format())


#%%connect to z-stage
z_stage = KDC101_Class.KDC101_translation(SN_KDC)
out_z_coord = z_stage.read_position()
print(out_z_coord)


#%%
delta_z = -0.1
z_stage.rel_move(delta_z)
out_z_coord = z_stage.read_position()
print(out_z_coord)

#%%
def abs_move_stage(abs_pos):
    
    if 0<abs_pos<24.5:
        out_z_coord_start = z_stage.read_position()
        z_stage.abs_move(float(abs_pos))
        out_z_coord_stop = z_stage.read_position()
        print(f"Start position: {out_z_coord_start:.2f}mm")
        print(f"Stop position: {out_z_coord_stop:.2f}mm")
    else:
        print("Give a position in the range [0-25mm]")
    
    
def rel_move_stage(rel_step = +0.1):
    rel_step = float(rel_step)
    
    if 0.001<rel_step<1.0:
        z_stage = KDC101_Class.KDC101_translation(SN_KDC)
        out_z_coord_start = z_stage.read_position()
        z_stage.rel_move(float(rel_step))
        out_z_coord_stop = z_stage.read_position()
        print(f"Start position: {out_z_coord_start:.2f}mm")
        print(f"Stop position: {out_z_coord_stop:.2f}mm")
    else:
        print("Give a valid relative jog size: [0.001-1.0]mm")


def snap_image(exp_time = 1/5000, camera_gain = 0, FPS = 5, plot=True):
    
    inlineOrWindow = 0 # inline=0 window=1
    
    #exp_time = 1/5000
    #camera_gain = 0
    #FPS = 5
    
    cam1.set_framerate(FPS)
    cam1.set_exposure(exp_time) #seconds
    cam1.set_gain(camera_gain)
    
    print('FPS: %.4f' %(cam1.get_framerate()))
    print('Exposure: %.10f (s)' %(cam1.get_exposure()))
    print('Gain: %.4f (dB)' %(cam1.get_gain()))
    
    
    img = cam1.snap_image() #save image
    
    if plot:
        if not inlineOrWindow:
            get_ipython().run_line_magic('matplotlib', 'inline')
        else:
            get_ipython().run_line_magic('matplotlib', 'qt5')
        
        
        if len(np.shape(img))==2: # monochrome
           
            plt.clf()
            # plt.imshow(img, cmap='coolwarm')
            plt.imshow(img, cmap='plasma')
            plt.title('SN:' + cam1.sn + ', Format:' + cam1.get_format())
            plt.colorbar()
            plt.show()
            
            print(' ')
            print('Mean:' + str(np.mean(img)))
            print('Max: ' + str(np.max(img)))
            
                
        elif len(np.shape(img))==3: # RGB
              
            img_norm = img[:,:,[0,1,2,3]]/(2**16) #normalize so that we can plot an RGB image
            # img_norm = np.uint8(img) #normalize
            img_norm_flip = np.flipud(img_norm)
               
            img_show = img_norm_flip
            img_show = img_norm
        
            
            plt.clf()
            # plt.imshow(img_show, cmap='coolwarm')
            plt.imshow(img_show, cmap='plasma')
            plt.title('SN:' + cam1.sn + ', Format:' + cam1.get_format())
            plt.show()
            
            print(' ')
            print('Mean (16 bit): R=%.2f, G=%.2f, B=%.2f' %(np.mean(img[:,:,0]),np.mean(img[:,:,1]),np.mean(img[:,:,2])))
            print('Max (16 bit=65536): R=%.2f, G=%.2f, B=%.2f' %(np.max(img[:,:,0]),np.max(img[:,:,1]),np.max(img[:,:,2])))

 #%%      
if not True: # save image
     
    now = datetime.datetime.now() #date and time
    today = now.strftime("%Y-%m-%d") #date
    now_time = now.strftime("%H.%M.%S") #time
    
    
    # location = (location + '/' + camera.get_format()+'/')
    
    # location = (r'G:\Shared drives\VitreaLab Share\Lab Data\Light Engine Team\X-Reality Projects (XR)\Virtual Reality (VR)\Lab data\2022-11-09\python')
    
    location = (r'G:\Shared drives\VitreaLab Share\Lab Data\Light Engine Team\X-Reality Projects (XR)\Augmented Reality (AR)\Lab data\AR')
    
    location = (location + '/' + today +'/')
    
    location = (location + '10mm lens/Speckle/' )
    
       
    nam = ('B_Dy_1_FPS=%.2f_Gain=%.2fdB_exp=%.4fs' 
           %(camera.get_framerate(),camera.get_framerate(),camera.get_exposure()))
    
    # nam = ('white_MEMS=40Vpp_401Hz_FPS=%.4f_Gain=%.4fdB_exp=%.4fs' 
    #        %(camera.get_framerate(),camera.get_framerate(),camera.get_exposure()))

    if not os.path.exists(location):
        os.makedirs(location)
    
    imageio.imwrite(location + nam +'.tiff', img) # save image
#%%

if not True: # TEST
#%%

    location = 'C:/Users/limit/Desktop/'
    nam = 'test'
    
    if not os.path.exists(location):
        os.makedirs(location)
    
    img = camera.snap_image()
    
    
    imageio.imwrite(location + nam +'.tiff', img) # save image
        
    img_load = imageio.imread(location + nam +'.tiff')
    
    print(' ')
    print('Mean (16 bit): R=%.2f, G=%.2f, B=%.2f' %(np.mean(img[:,:,0]),np.mean(img[:,:,1]),np.mean(img[:,:,2])))
    print('Max (16 bit): R=%.2f, G=%.2f, B=%.2f' %(np.max(img[:,:,0]),np.max(img[:,:,1]),np.max(img[:,:,2])))
    
    print(' ')
    print('Mean: R=%.2f, G=%.2f, B=%.2f' %(np.mean(img_load[:,:,0]),np.mean(img_load[:,:,1]),np.mean(img_load[:,:,2])))
    print('Max: R=%.2f, G=%.2f, B=%.2f' %(np.max(img_load[:,:,0]),np.max(img_load[:,:,1]),np.max(img_load[:,:,2])))
    
    print(' ')
    print(np.sum(img-img_load))
#%%





if not True: # video
#%%
    for ii in range(0,100): 

        img = camera.snap_image()
        img_norm = img[:,:,[0,1,2,3]]/(2**16) #normalize

        plt.imshow(img_norm)
        plt.show()
        print(ii)
#%%
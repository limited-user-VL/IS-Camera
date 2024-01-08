# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:08:23 2023

@author: Rui

Goal: determine the angle of emission of each beam in a photonic chip.
Initially designed for AR light engine characterisation.

The setup consists of a photonic chip imaged through a lens (f=60mm), in a 
configuration with magnification close to 1. The image is captured
by an imaging source camera (DMK33UX183).

A translation stage (Thorlabs Z825B) sweeps the position of the chip
along the optical axis and for each position and image is captured;

From the relative displacement of the beams (x,y) between different
snapshots one retrieves the direction vectors.
"""

import site
from pathlib import Path
import os
import sys
import time
import numpy as np
import pickle 
import datetime


api_folder = Path(r"G:/") / "Geteilte Ablagen" / "VitreaLab Share" / "Lab Software"/ "api_vitrealab"
if api_folder not in sys.path:
    site.addsitedir(str(api_folder))
api_folder = Path(r"G:/") / "Shared drives" / "VitreaLab Share" / "Lab Software" / "api_vitrealab"
if api_folder not in sys.path:
    site.addsitedir(str(api_folder))
    
import Kinesis.KDC101_Translation_Class as KDC101_Class  # output linear stages y -z



#%%
#Initialise camera and stage
from camera_control.tis_camera import TISCamera

exposure = 0.01

camera = TISCamera('19020486')
#camera.set_format('Y16 (1280x720)')
camera.set_format('Y16 (4096x2160)')
camera.set_framerate(50)
camera.set_gain(0.0)
camera.set_exposure(exposure)
img = camera.snap_image() #save image

SN_z_outp_stage = 27258444
z_stage = KDC101_Class.KDC101_translation(SN_z_outp_stage)


#delta_z = -0.1
#z_pos = z_stage.read_position()
#z_stage.abs_move(z_pos+delta_z)
#z_stage.rel_move(delta_z)
   

#%% Auxililary funcitons - Optimise exposure

def optimise_exposure(camera):
    exposure = camera.get_exposure()
    time.sleep(0.1)
    
    while True:
        max_signal = camera.snap_image().max()
        
        if max_signal>3100:
            exposure=exposure/2.0
            print(exposure)
            camera.set_exposure(exposure) #define camera exposure

        elif max_signal<2500:
            exposure=exposure*1.2
            print(exposure)
            camera.set_exposure(exposure) #define camera exposure
        else:
            break

    print(f"Finished optimisation: {exposure=: .5f}s and {max_signal=: .0f}")
    return exposure, max_signal



def acquire(min_pos = 8, max_pos = 11,  n_shots = 10):
    
    pos0 = z_stage.read_position()
    pos_arr = np.linspace(min_pos, max_pos, n_shots)
    
    
    img_dict = {}
    for i, z in enumerate(pos_arr):
        z_stage.abs_move(z)
        print(f"\nMoved z-stage to {z: .2f}mm")
        time.sleep(0.1)
        
        optimise_exposure(camera)
        img = camera.snap_image()
        print("Image acquired.")
        img_dict[str(z)]=img
        
    return img_dict

    
#%% Execute code
current_date = datetime.date.today()
save_path = Path("G:/") / "Shared drives" / "VitreaLab Share" / "Lab Data" / "Light Engine Team" / "X-Reality Projects (XR)" / "Augmented Reality (AR)" / "Lab Data" / "AR Lab"  / str(current_date)
if not os.path.exists(save_path):
    os.makedirs(save_path)

img_dict = acquire(min_pos = 4.5, max_pos = 6.5, n_shots = 20)

time_stamp = datetime.datetime.now().strftime("%H:%M:%S").replace(":","_")
file_path = save_path / f'in_out_focus_dict_{time_stamp}.pkl'
with open(file_path, 'wb+') as f:
    pickle.dump(img_dict, f)
print(f"Image was saved in {file_path}")
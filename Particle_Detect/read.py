import os
import numpy as np
import cv2 as cv
#import mayavi.mlab as mlab
from PIL import Image
from Data import Data
'''
_Parameter Config_
    include:
        - Path of the image_folder
        - The output path of the label
'''
Left_Path=''
Right_Path=''
Output_Path=''
def Read_Image(Image_Path):
    """_summary_
    Support reading bmp file.
    Args:
        Image_Path (_type_): _description_
    """    
    img=Image.open(Image_Path)
    img=np.array(img, dtype=np.uint8)
    return img


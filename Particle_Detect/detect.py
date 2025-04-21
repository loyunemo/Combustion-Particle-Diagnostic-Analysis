import numpy as np
from Data import Data
from PIL import Image
from read import Read_Image
from Seg import Kmeans_Seg,MOG2
from Locate import locate,write_single_img_json
import os
import json
import cv2
Left_Image_Path='C:/Users/Claudius/WorkingSpace/Article1/0722experiment2/51/BMP192.168.8.51-20240707-072633'
Right_Image_Path='C:/Users/Claudius/WorkingSpace/Article1/0722experiment2/52/BMP192.168.8.52-20240707-072635'
K1 = np.array([[1.151812867953681e+04, 0, 2.930281306554746e+02],
               [0, 1.144065639977414e+04,9.299614621342721e+02], 
               [0, 0, 1]], dtype=np.float64)
dist1 = np.array([-1.03599173650834,30.4343730533085,-0.0175971456828163,0.0775278591952446], dtype=np.float64)

K2 = np.array([[1.416281370757014e+04, 0, 2.186893663098017e+03],
               [0,1.397124917519382e+04, -2.934480724268153e+02], 
               [0, 0, 1]], dtype=np.float64)
dist2 = np.array([1.49649840089790,-14.9214987001899, -0.0891119097877299,0.0319277395780453], dtype=np.float64)

Kmeans=Kmeans_Seg()
Back_SubSegLeft=MOG2(Left_Image_Path,dist1,K1)
Back_SubSegLeft.pretrain(1000,1999)
print("[INFO]: Pretraining Left Image Done")
Back_SubSegRight=MOG2(Right_Image_Path,dist2,K2)
Back_SubSegRight.pretrain(1000,1999)
print("[INFO]: Pretraining Right Image Done")
Left_sight=[]
Right_sight=[]

    
for val in range(2000,2100):
    left=Back_SubSegLeft.MOG2_Seg(val,False)
    right=Back_SubSegRight.MOG2_Seg(val,False)
    Image_Info_Left,Radius_Info_Left,Img_Path_Info_Left=locate(left,Left_Image_Path,Kmeans,Back_SubSegLeft)
    Image_Info_Right,Radius_Info_Right,Img_Path_Info_Right=locate(right,Right_Image_Path,Kmeans,Back_SubSegRight)
    Left_sight.append([Radius_Info_Left,Img_Path_Info_Left])
    Right_sight.append([Radius_Info_Right,Img_Path_Info_Right])
Left_Sight_Data_JSON='Particle_info_left.json'
Right_Sight_Data_JSON='Particle_info_right.json'
with open(Left_Sight_Data_JSON, "r+") as file:  # 以读写模式打开
     file.truncate(0)  # 清空文件内容
with open(Right_Sight_Data_JSON, "r+") as file:
    file.truncate(0)
for i in range(len(Left_sight)):
    write_single_img_json(Left_Sight_Data_JSON,Left_sight[i][0],Left_sight[i][1])
    write_single_img_json(Right_Sight_Data_JSON,Right_sight[i][0],Right_sight[i][1])
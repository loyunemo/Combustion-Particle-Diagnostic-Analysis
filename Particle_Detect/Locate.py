import numpy as np
from Data import Data
from PIL import Image
from read import Read_Image
from Seg import Kmeans_Seg,MOG2
import os
import json
import cv2
def locate(info,path,Kmeans_Instance,Seg):
    Image_Info=[]
    Radius_Info=[]
    for iter in info:
        sth=iter.Data_Write()
        Particle_path=os.path.join(path,Seg.Imagelist[sth[-1]])
        unix_path = Particle_path.replace("\\", "/")  # 替换 \ 为 /
        Single_imginfo,Single_radiusinfo=Kmeans_Instance.Seg(unix_path,sth[0],sth[1],sth[2],sth[3],Seg.Distort_Cofficient,Seg.Intrinsic_Matrix,False)
        Image_Info.append(Single_imginfo)
        Radius_Info.append(Single_radiusinfo)
        Img_Path_Info=[Particle_path,sth[-1]]
    return Image_Info,Radius_Info,Img_Path_Info
def write_single_img_json(filename,Radius_Info,Img_Path_Info):
    with open(filename, 'a', encoding='utf-8') as f:
        for info in Radius_Info:
            for t in info:    
                if t[2]<1e-4:
                    continue
                disk ={'x':t[0],'y':t[1],'r':t[2],'x_in':t[3],'y_in':t[4],'img_path':Img_Path_Info[0],'img_num':Img_Path_Info[1]}
                json.dump(disk, f, ensure_ascii=False)
                f.write('\n')  # 添加换行符





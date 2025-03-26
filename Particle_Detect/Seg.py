import os
import numpy as np
import cv2 as cv
import math
#import mayavi.mlab as mlab
from PIL import Image
from Data import Data
class MOG2(object):
    def __init__(self):
        pass
class Kmeans_Seg(object):
    """_summary_
    Using Kmeans to segment the image.Return the label mask and the center of the cluster,at the same time 

    Args:
        object (_type_): _description_
    """
    def __init__(self):
        self.K=3
        self.iteration=10
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    def Seg(self,img,x,y,w,h,distort_Cofficient,Intrinsic_Matrix,show=False):
        """_summary_

        Args:
            img (_type_): the path to the image
            x (_type_): the x coordinate selected part of the image
            y (_type_): the y coordinate selected part of the image
            w (_type_): the width selected part of the image
            h (_type_): _description_
            show (_type_): _description_
        Returns:
            _type_: _description_
        """        

        Img_Select=np.array(Image.open(img),dtype=np.uint8)
        Img_Select=cv.undistort(Img_Select, Intrinsic_Matrix, distort_Cofficient)
        Img_Select=Img_Select[y:y+h,x:x+w]  
        print(f"[IMAGE INFO]:{Img_Select.shape}")
        #cv.imshow('Selected Image',Img_Select)  # 显示选定的图像区域
        #cv.waitKey(0)  # 等待按键事件
        print(f"[PART INFO]:X:{x},Y:{y},W:{w},H:{h}")      
        Z = Img_Select.reshape((-1, 1)).astype(np.float32)  # Reshape and convert to np.float32
        _, labels, centers = cv.kmeans(Z, self.K, None, self.criteria, self.iteration, cv.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        sorted_indices = np.argsort(np.mean(centers, axis=1))  # 根据亮度排序
        color_map = {sorted_indices[0]: [0, 0, 0],   # 最暗的部分用蓝色（BGR）
                 sorted_indices[1]: [0, 0, 0], # 中间亮度部分用黄色（BGR）
                 sorted_indices[2]: [0, 0, 255]}   # 最亮部分用红色（BGR）
        colored_image = np.array([color_map[label] for label in labels.flatten()], dtype=np.uint8)
        colored_image = colored_image.reshape((h, w, 3))
        analyzed_image = cv.cvtColor(colored_image, cv.COLOR_BGR2GRAY)
        cntsa, _ = cv.findContours(analyzed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if (len(cntsa)==1):
            (x1,y1),radius=cv.minEnclosingCircle(cntsa[0])
            info=[[x1,y1,radius,x,y]]
        else:
            info=[]
            for cnta in cntsa:
                (x1,y1),radius=cv.minEnclosingCircle(cnta)
                info.append([x1,y1,radius,x,y])
       
        if show==True:
            resizeimg=cv.resize(colored_image, (w*20, h*20))
            cv.imshow('Segmented Image',resizeimg)
            cv.waitKey(0)
        return colored_image,info
        
class MOG2(object):
    """_summary_
    Encapsulation of MOG2 algorithm
    Args:
        object (_type_): _description_
    """    
    def __init__(self,Image_Path,distort_Cofficient,Intrinsic_Matrix):
        self.fgbg = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=False)#前景提取器    
        self.Image_Path=Image_Path
        self.Intrinsic_Matrix=Intrinsic_Matrix
        self.Distort_Cofficient=distort_Cofficient
        self.Imagelist=sorted([f for f in os.listdir(self.Image_Path) if f.endswith('bmp')])
    def pretrain(self,train_begin,train_end):
        """_summary_
        Pretrain the MOG2 model,to make the model in accordance with the real background
 
        Args:
            train_begin (_type_): _description_
            train_end (_type_): _description_
        """        
        self.train_begin=train_begin
        self.Image_Path=self.Image_Path
        self.train_end=train_end
        self.train_list=self.Imagelist[train_begin:train_end]
        for iter in self.train_list:
            image_path = os.path.join(self.Image_Path, iter)
            image = Image.open(image_path)
            image_8bit = np.array(image, dtype=np.uint8)
            img_target=cv.undistort(image_8bit, self.Intrinsic_Matrix, self.Distort_Cofficient)
            self.fgbg.apply(img_target)
    def MOG2_Seg(self,Img_Serial_Num,show=False):
        img8_b=np.array(Image.open(os.path.join(self.Image_Path,self.Imagelist[Img_Serial_Num])))
        img=cv.undistort(img8_b, self.Intrinsic_Matrix, self.Distort_Cofficient)
        if len(img.shape)==3:
            img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
            canvas=img.copy()
        elif len(img.shape)==2:
            canvas=cv.cvtColor(img.copy(),cv.COLOR_GRAY2RGB)
        
        fgmask = self.fgbg.apply(img)
        #fgmask=255-fgmask#正常了
        fgmask=cv.GaussianBlur(fgmask, (3, 3), 0) 
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)
        #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)
        #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel) 
        cnts,_=cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        DropletData=[]
        for cnt in cnts:
            x, y, w, h = cv.boundingRect(cnt)
            if show==True:
                cv.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 0), 1)

            DropletData.append(Data(x,y,w,h,img,Img_Serial_Num))
        if show==True:
            cv.imshow('Segmented Image',canvas)
            cv.waitKey(0)
        return DropletData
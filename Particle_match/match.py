import numpy as np
import os
import json
import cv2
import screeninfo
import random
import math
from Data import Data
from Particle import Particle_Info
canvas=np.zeros((2592, 1920, 3), dtype=np.uint8)
def show_resized_image_auto(img):
    # 读取图片    
    if img is None:
        print("无法加载图片")
        return
    
    # 获取屏幕尺寸
    screen = screeninfo.get_monitors()[0]
    screen_width, screen_height = screen.width, screen.height

    # 计算缩放比例
    h,w= img.shape[:2]
    scale = min(screen_width / w, screen_height / h) * 0.9  # 预留10%边距
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 调整图片大小
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 显示图片
    cv2.imshow("Resized Image", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class Point(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def __str__(self):
        return "({0},{1})".format(self.x, self.y)

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Point(x, y)
    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Point(x, y)
    def __mul__(self, other):
        x = self.x * other.x
        y = self.y * other.y
        return Point(x, y)
    def __len__(self):
        return math.sqrt(self.x**2+self.y**2)
    def __truediv__(self, other):
        if other.x == 0 or other.y == 0:
            raise ZeroDivisionError("Division by zero is not allowed.")   
        x = self.x / other.x
        y = self.y / other.y
    def dot(self, other):
        return self.x * other.x + self.y * other.y
class Particle_Set(object):
    def __init__(self,name,json_set):
        self.name=name
        self.Time_Stamp=json_set[0]['img_num']
        self.Image_Path=json_set[0]['img_path']
        self.Particle_Num=len(json_set)
        self.Particle_List=[]
        for iter,entry in enumerate(json_set):
            coordinate=Point(entry['x']+entry['x_in'],entry['y']+entry['y_in'])
            radius=entry['r']
            iternum=iter
            Particle_info_Select=Particle_Info(coordinate.x,coordinate.y,radius,self.Time_Stamp,iternum)
            self.Particle_List.append([coordinate,radius,iternum,Particle_info_Select])
    def plot(self,canvas,color=(0,255,0)):
        for iter in self.Particle_List:
            coordinate=iter[0]
            radius=iter[1]
            iternum=iter[2]
            cv2.circle(canvas,(int(coordinate.x),int(coordinate.y)),int(radius),color,2)
            #cv2.putText(canvas,str(iternum),(int(coordinate.x),int(coordinate.y)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
class Image_Set(object):
    def __init__(self,particle_set_lst):
        self.particle_set_lst=particle_set_lst
        
                
        
def Match_Evaluate(before,after,canvas):
    for Particle in before.Particle_List:
        cv2.circle(canvas,(int(Particle.coordinate.x),int(Particle.coordinate.y)),int(Particle[1]),(255,0,0),2)
    for Particle in after.Particle_List:
        cv2.circle(canvas,(int(Particle.coordinate.x),int(Particle.coordinate.y)),int(Particle[1]),(0,255,0),2)
    for Particle in before.Particle_List:
        coordinate=Particle.coordinate
        radius=Particle[1]
        iternum=Particle[2]
        distance_lst=[]
        for num,Particle_after in enumerate(after.Particle_List):
            coordinate_after=Particle_after[0]
            radius_after=Particle_after[1]
            iternum_after=Particle_after[2]
            distance=np.sqrt((coordinate.x-coordinate_after.x)**2+(coordinate.y-coordinate_after.y)**2)
            distance_lst.append([distance,iternum_after])
        distance_lst.sort(key=lambda x: x[0])
        for i in range(5):
            cv2.line(canvas,(int(coordinate.x),int(coordinate.y)),(int(after.Particle_List[distance_lst[i][1]][0].x),int(after.Particle_List[distance_lst[i][1]][0].y)),(0,0,255),2)
            cv2.putText(canvas,str(distance_lst[i][0]),(int(coordinate.x),int(coordinate.y)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
    cv2.imwrite('Match.jpg',canvas)
    cv2.imshow('Match',canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def Match_Evaluate(before,after,canvas,numb=None,drawflagbefore=True,drawflagafter=True,showflag=True,color=(0,0,255),colorbefore=(255,0,0),colorafter=(0,255,0)):
    if drawflagbefore==True:
        for Particle in before.Particle_List:
            cv2.circle(canvas,(int(Particle.coordinate.x),int(Particle.coordinate.y)),int(Particle.radius),colorbefore,2)
    if drawflagafter==True:
        for Particle in after.Particle_List:
            cv2.circle(canvas,(int(Particle.coordinate.x),int(Particle.coordinate.y)),int(Particle.radius),colorafter,2)
    if numb==None:
        return
    Particle=before.Particle_List[numb]
    coordinate=Particle.coordinate
    radius=Particle.radius
    iternum=Particle.Lstnum
    distance_lst=[]
    for num,Particle_after in enumerate(after.Particle_List):
        coordinate_after=Particle_after.coordinate
        radius_after=Particle_after.radius
        iternum_after=Particle_after.Lstnum
        distance=np.sqrt((coordinate.x-coordinate_after.x)**2+(coordinate.y-coordinate_after.y)**2)
        distance_lst.append([distance,iternum_after])
    distance_lst.sort(key=lambda x: x[0])
    for i in range(5):
        #cv2.line(canvas,(int(coordinate.x),int(coordinate.y)),(int(after.Particle_List[distance_lst[i][1]][0].x),int(after.Particle_List[distance_lst[i][1]][0].y)),color,1)
        #cv2.putText(canvas,str(distance_lst[i][0]),(int((coordinate.x+after.Particle_List[distance_lst[i][1]][0].x)/2),int((coordinate.y+after.Particle_List[distance_lst[i][1]][0].y)/2)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        print(f"[INFO]:The Minimum Distance Match is:({numb},{distance_lst[i][1]}),Distance:{distance_lst[i][0]},RadiusDelta:{abs(after.Particle_List[distance_lst[i][1]].radius-Particle.radius)}") 
    
    cv2.imwrite('Match.jpg',canvas)
    if showflag==True: 
        screen = screeninfo.get_monitors()[0]
        screen_width, screen_height = screen.width, screen.height
        h,w= canvas.shape[:2]
        scale = min(screen_width / w, screen_height / h) * 0.9  # 预留10%边距
        new_w, new_h = int(w * scale), int(h * scale)
        canvas = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)
        #cv2.resize(canvas,(canvas.shape[1]//2,canvas.shape[0]//2),interpolation=cv2.INTER_AREA)
        cv2.imshow('Match',canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def Similarity_Evaluate(frame_start,frame_medium,frame_end,canvas):
    ParticleLd=[]
    for Particle in frame_start.Particle_List:
        coordinate=Particle.coordinate
        radius=Particle[1]
        iternum=Particle[2]
        distance_lst=[]
        for num,Particle_after in enumerate(frame_medium.Particle_List):
            coordinate_after=Particle_after[0]
            radius_after=Particle_after[1]
            iternum_after=Particle_after[2]
            distance=np.sqrt((coordinate.x-coordinate_after.x)**2+(coordinate.y-coordinate_after.y)**2)
            distance_lst.append([distance,iternum_after,Point(coordinate_after.x-coordinate.x,coordinate_after.y-coordinate.y)])
        distance_lst.sort(key=lambda x: x[0])
        distance_lst=distance_lst[:3]
        for i in range(3):
            coordinate_medium=frame_medium.Particle_List[distance_lst[i][1]][0]
            radius_medium=frame_medium.Particle_List[distance_lst[i][1]][1]
            iternum_medium=frame_medium.Particle_List[distance_lst[i][1]][2]
            dislst=[]
            for Particle_second in frame_end.Particle_List:
                coordinate_second=Particle_second[0]
                radius_second=Particle_second[1]
                iternum_second=Particle_second[2]
                distance2=np.sqrt((coordinate_medium.x-coordinate_second.x)**2+(coordinate_medium.y-coordinate_second.y)**2)
                p2=Point(coordinate_second.x-coordinate_medium.x,coordinate_second.y-coordinate_medium.y)
                if(distance_lst[i][2].__len__()==0 and p2.__len__()==0):
                    similarity=1
                elif(distance_lst[i][2].__len__()==0 or p2.__len__()==0):
                    similarity=0
                else:
                    similarity=-1.5*distance_lst[i][2].dot(p2)/(distance_lst[i][2].__len__()*p2.__len__())+0*abs(distance_lst[i][0]-distance2)/max(distance_lst[i][0],distance2)
                dislst.append([distance2,iternum_second,p2,similarity])
            dislst.sort(key=lambda x: x[0])
            dist=dislst
            distance_lst[i]=[distance_lst[i].copy(),dist[0]]
        distance_lst.sort(key=lambda x: x[1][3])
        select=distance_lst[0]
        ParticleLd.append([Particle.coordinate,Particle[1],Particle[2],select])
        cv2.line(canvas,(int(coordinate.x),int(coordinate.y)),(int(frame_medium.Particle_List[select[0][1]][0].x),int(frame_medium.Particle_List[select[0][1]][0].y)),(0,0,255),1)
        cv2.line(canvas,(int(frame_medium.Particle_List[select[0][1]][0].x),int(frame_medium.Particle_List[select[0][1]][0].y)),(int(frame_end.Particle_List[select[1][1]][0].x),int(frame_end.Particle_List[select[1][1]][0].y)),(0,128,128),1)
def Similarity_Distance_Evaluate(frame_start,frame_medium,frame_end,canvas):
    ParticleLd=[]
    for Particle in frame_start.Particle_List:
        coordinate=Particle.coordinate
        radius=Particle[1]
        iternum=Particle[2]
        distance_lst=[]
        for num,Particle_after in enumerate(frame_medium.Particle_List):
            coordinate_after=Particle_after[0]
            radius_after=Particle_after[1]
            iternum_after=Particle_after[2]
            distance=np.sqrt((coordinate.x-coordinate_after.x)**2+(coordinate.y-coordinate_after.y)**2)
            distance_lst.append([distance,iternum_after,Point(coordinate_after.x-coordinate.x,coordinate_after.y-coordinate.y)])
        distance_lst.sort(key=lambda x: x[0])
        distance_lst=distance_lst[0]

        coordinate_medium=frame_medium.Particle_List[distance_lst[1]][0]
        radius_medium=frame_medium.Particle_List[distance_lst[1]][1]
        iternum_medium=frame_medium.Particle_List[distance_lst[1]][2]
        dislst=[]
        for Particle_second in frame_end.Particle_List:
            coordinate_second=Particle_second[0]
            radius_second=Particle_second[1]
            iternum_second=Particle_second[2]
            distance2=np.sqrt((coordinate_medium.x-coordinate_second.x)**2+(coordinate_medium.y-coordinate_second.y)**2)
            p2=Point(coordinate_second.x-coordinate_medium.x,coordinate_second.y-coordinate_medium.y)
            if(distance_lst[2].__len__()==0 and p2.__len__()==0):
                similarity=1
            elif(distance_lst[2].__len__()==0 or p2.__len__()==0):
                similarity=0
            else:
                similarity=1.5*distance_lst[2].dot(p2)/(distance_lst[2].__len__()*p2.__len__())-0.5*abs(distance_lst[0]-distance2)/max(distance_lst[0],distance2)
            dislst.append([distance2,iternum_second,p2,similarity])
        dislst.sort(key=lambda x: x[0])
        dist=dislst
        distance_lst=[distance_lst.copy(),dist[0]]
        select=distance_lst
        ParticleLd.append([Particle.coordinate,Particle[1],Particle[2],select])
        cv2.line(canvas,(int(coordinate.x),int(coordinate.y)),(int(frame_medium.Particle_List[select[0][1]].coordinate.x),int(frame_medium.Particle_List[select[0][1]].coordinate.y)),(255,128,255),1)
        cv2.line(canvas,(int(frame_medium.Particle_List[select[0][1]].coordinate.x),int(frame_medium.Particle_List[select[0][1]].coordinate.y)),(int(frame_end.Particle_List[select[1][1]].coordinate.x),int(frame_end.Particle_List[select[1][1]].coordinateS.y)),(0,128,128),1)
def Similarity_Predict_Evaluate(frame_start,frame_medium,frame_end,canvas,drawflag1=True,drawflag2=True):
    ParticleLd=[]
    ID_Select=[]
    for Particle in frame_start.Particle_List:
        coordinate=Particle.coordinate
        radius=Particle.radius
        iternum=Particle.Lstnum
        distance_lst=[]
        for num,Particle_after in enumerate(frame_medium.Particle_List):
            coordinate_after=Particle_after.coordinate
            radius_after=Particle_after.radius
            iternum_after=Particle_after.Lstnum
            distance=np.sqrt((coordinate.x-coordinate_after.x)**2+(coordinate.y-coordinate_after.y)**2)
            distance_lst.append([distance,iternum_after,Point(coordinate_after.x-coordinate.x,coordinate_after.y-coordinate.y),iternum])
        distance_lst.sort(key=lambda x: x[0])
        distance_lst=distance_lst[:5]
        popid=[]
        for i in range(5):
            coordinate_medium=frame_medium.Particle_List[distance_lst[i][1]].coordinate
            radius_medium=frame_medium.Particle_List[distance_lst[i][1]].radius
            iternum_medium=frame_medium.Particle_List[distance_lst[i][1]].Lstnum
            dislst=[]
            for Particle_second in frame_end.Particle_List:
                coordinate_second=Particle_second.coordinate
                radius_second=Particle_second.radius
                iternum_second=Particle_second.Lstnum
                distance2=np.sqrt((coordinate_medium.x-coordinate_second.x)**2+(coordinate_medium.y-coordinate_second.y)**2)
                p2=Point(coordinate_second.x-coordinate_medium.x,coordinate_second.y-coordinate_medium.y)
                delta=(distance_lst[i][2]-p2).__len__()
                if(distance_lst[i][2].__len__()==0 ):
                    if(delta<10):
                        dislst.append([distance2,iternum_medium,p2,delta,iternum_second])
                elif(delta<distance_lst[i][2].__len__()/3.0 and p2.dot(distance_lst[i][2])>0 and max(p2.__len__(),delta)<150):
                    if(delta==0):
                        dislst.append([distance2,iternum_medium,p2,delta,iternum_second])
                    else:
                        if(p2.dot(distance_lst[i][2])/distance_lst[i][2].__len__()/delta>np.cos(45/180*math.pi)):
                            dislst.append([distance2,iternum_medium,p2,delta,iternum_second])
            if(len(dislst)==0):
                popid.append(i)
            else:
                dislst.sort(key=lambda x: x[3])
                distance_lst[i]=[distance_lst[i].copy(),dislst[0]]
        popid.reverse()
        for i in popid: 
            distance_lst.pop(i)
        if(len(distance_lst)==0):
            continue
        else:
            distance_lst.sort(key=lambda x: x[1][3])
            select=distance_lst[0]
            ParticleLd.append([Particle.coordinate,Particle.radius,Particle.Lstnum,select])
            ID_Select.append([Particle.Lstnum,select[1][1],select[1][4]])
            start=Particle_Info(Particle.coordinate.x,Particle.coordinate.y,Particle.radius,frame_start.Time_Stamp,Particle.Lstnum)
            #medium=Particle_Info(frame_medium.Particle_List[select[0][1]][0].x,frame_medium.Particle_List[select[0][1]][0].y,frame_medium.Particle_List[select[0][1]][1],frame_medium.Time_Stamp,select[0][1])
            #end=Particle_Info(frame_end.Particle_List[select[1][4]][0].x,frame_end.Particle_List[select[1][4]][0].y,frame_end.Particle_List[select[1][4]][1],frame_end.Time_Stamp,select[1][4])
            if drawflag1==True:
                cv2.line(canvas,(int(coordinate.x),int(coordinate.y)),(int(frame_medium.Particle_List[select[0][1]][0].x),int(frame_medium.Particle_List[select[0][1]][0].y)),(0,0,255),1)
            if drawflag2==True:
                cv2.line(canvas,(int(frame_medium.Particle_List[select[0][1]][0].x),int(frame_medium.Particle_List[select[0][1]][0].y)),(int(frame_end.Particle_List[select[1][4]][0].x),int(frame_end.Particle_List[select[1][4]][0].y)),(0,128,128),1)
    return ID_Select


def Read_json(json_path,img_serial_num):
    
    sort_img_lst=sorted([f for f in os.listdir(json_path) if f.endswith('json')])
    beginname=sort_img_lst[0]
    begin_serial=int(beginname.split('.')[0].split('_')[1])
    img_name=sort_img_lst[img_serial_num-begin_serial]
    jsonname=os.path.join(json_path,img_name)
    unix_path = jsonname.replace("\\", "/")  # 替换 \ 为 /
    Info = []
    with open(unix_path, 'r', encoding='utf-8') as f:
        Info= json.load(f)
    return Info

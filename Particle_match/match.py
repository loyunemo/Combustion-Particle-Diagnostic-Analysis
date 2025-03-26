import numpy as np
import os
import json
import cv2
import screeninfo
import random
import math
from Data import Data
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
        return Point(x, y)
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
            self.Particle_List.append([coordinate,radius,iternum])
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
        cv2.circle(canvas,(int(Particle[0].x),int(Particle[0].y)),int(Particle[1]),(255,0,0),2)
    for Particle in after.Particle_List:
        cv2.circle(canvas,(int(Particle[0].x),int(Particle[0].y)),int(Particle[1]),(0,255,0),2)
    for Particle in before.Particle_List:
        coordinate=Particle[0]
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
def Match_Evaluate(before,after,canvas,numb,drawflagbefore=True,drawflagafter=True,showflag=True,color=(0,0,255),colorbefore=(255,0,0),colorafter=(0,255,0)):
    if drawflagbefore==True:
        for Particle in before.Particle_List:
            cv2.circle(canvas,(int(Particle[0].x),int(Particle[0].y)),int(Particle[1]),colorbefore,2)
    if drawflagafter==True:
        for Particle in after.Particle_List:
            cv2.circle(canvas,(int(Particle[0].x),int(Particle[0].y)),int(Particle[1]),colorafter,2)
    Particle=before.Particle_List[numb]
    coordinate=Particle[0]
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
        #cv2.line(canvas,(int(coordinate.x),int(coordinate.y)),(int(after.Particle_List[distance_lst[i][1]][0].x),int(after.Particle_List[distance_lst[i][1]][0].y)),color,1)
        #cv2.putText(canvas,str(distance_lst[i][0]),(int((coordinate.x+after.Particle_List[distance_lst[i][1]][0].x)/2),int((coordinate.y+after.Particle_List[distance_lst[i][1]][0].y)/2)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        print(f"[INFO]:The Minimum Distance Match is:({numb},{distance_lst[i][1]}),Distance:{distance_lst[i][0]},RadiusDelta:{abs(after.Particle_List[distance_lst[i][1]][1]-Particle[1])}") 
    
    cv2.imwrite('Match.jpg',canvas)
    if showflag==True: 
        cv2.imshow('Match',canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def Similarity_Evaluate(frame_start,frame_medium,frame_end,canvas):
    ParticleLd=[]
    for Particle in frame_start.Particle_List:
        coordinate=Particle[0]
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
        ParticleLd.append([Particle[0],Particle[1],Particle[2],select])
        cv2.line(canvas,(int(coordinate.x),int(coordinate.y)),(int(frame_medium.Particle_List[select[0][1]][0].x),int(frame_medium.Particle_List[select[0][1]][0].y)),(0,0,255),1)
        cv2.line(canvas,(int(frame_medium.Particle_List[select[0][1]][0].x),int(frame_medium.Particle_List[select[0][1]][0].y)),(int(frame_end.Particle_List[select[1][1]][0].x),int(frame_end.Particle_List[select[1][1]][0].y)),(0,128,128),1)
def Similarity_Distance_Evaluate(frame_start,frame_medium,frame_end,canvas):
    ParticleLd=[]
    for Particle in frame_start.Particle_List:
        coordinate=Particle[0]
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
        ParticleLd.append([Particle[0],Particle[1],Particle[2],select])
        cv2.line(canvas,(int(coordinate.x),int(coordinate.y)),(int(frame_medium.Particle_List[select[0][1]][0].x),int(frame_medium.Particle_List[select[0][1]][0].y)),(255,128,255),1)
        cv2.line(canvas,(int(frame_medium.Particle_List[select[0][1]][0].x),int(frame_medium.Particle_List[select[0][1]][0].y)),(int(frame_end.Particle_List[select[1][1]][0].x),int(frame_end.Particle_List[select[1][1]][0].y)),(0,128,128),1)
def Similarity_Predict_Evaluate(frame_start,frame_medium,frame_end,canvas,drawflag1=True,drawflag2=True):
    ParticleLd=[]
    for Particle in frame_start.Particle_List:
        coordinate=Particle[0]
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
        distance_lst=distance_lst[:5]
        popid=[]
        for i in range(5):
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
                delta=(distance_lst[i][2]-p2).__len__()
                if(distance_lst[i][2].__len__()==0 ):
                    if(delta<10):
                        dislst.append([distance2,iternum_second,p2,delta])
                elif(delta<distance_lst[i][2].__len__()/3.0 and p2.dot(distance_lst[i][2])>0 and max(p2.__len__(),delta)<150):
                    if(delta==0):
                        dislst.append([distance2,iternum_second,p2,delta])
                    else:
                        if(p2.dot(distance_lst[i][2])/distance_lst[i][2].__len__()/delta>np.cos(15/180*math.pi)):
                            dislst.append([distance2,iternum_second,p2,delta])
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
            ParticleLd.append([Particle[0],Particle[1],Particle[2],select])
            if drawflag1==True:
                cv2.line(canvas,(int(coordinate.x),int(coordinate.y)),(int(frame_medium.Particle_List[select[0][1]][0].x),int(frame_medium.Particle_List[select[0][1]][0].y)),(0,0,255),1)
            if drawflag2==True:
                cv2.line(canvas,(int(frame_medium.Particle_List[select[0][1]][0].x),int(frame_medium.Particle_List[select[0][1]][0].y)),(int(frame_end.Particle_List[select[1][1]][0].x),int(frame_end.Particle_List[select[1][1]][0].y)),(0,128,128),1)



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
json_path_left='C:/Users/Claudius/WorkingSpace/Article1/Particle_Info_Left_Data'
json_path_right='C:/Users/Claudius/WorkingSpace/Article1/Particle_Info_Right_Data'
info_left=[]
info_right=[]
for set in range(2000,2100):
    info_left.append(Particle_Set(f'INFO_{set}_LEFT',Read_json(json_path_left,set)))
    info_right.append(Particle_Set(f'INFO_{set}_RIGHT',Read_json(json_path_right,set)))
canvas_left=np.zeros((1920,2592,3), dtype=np.uint8)
canvas_match_left=np.zeros((1920,2592,3), dtype=np.uint8)
#canvas_right=np.zeros((2592, 1920, 3), dtype=np.uint8)
colorset1 = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]
#colorset=[(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,128,128),(128,0,128),(128,128,0),(0,128,128)]

#cv2.imwrite('Debug.jpg',canvas_left)
#show_resized_image_auto(canvas_left)
Similarity_Predict_Evaluate(info_left[0],info_left[1],info_left[2],canvas_match_left,drawflag1=True,drawflag2=True)
#Similarity_Predict_Evaluate(info_left[1],info_left[2],info_left[3],canvas_match_left,drawflag1=False,drawflag2=True)
Match_Evaluate(info_left[0],info_left[1],canvas_match_left,90,drawflagbefore=True,drawflagafter=True,showflag=False,colorbefore=colorset1[10],colorafter=colorset1[11])
Match_Evaluate(info_left[1],info_left[2],canvas_match_left,84,drawflagbefore=False,drawflagafter=True,showflag=False,color=colorset1[0],colorafter=colorset1[12])
Match_Evaluate(info_left[2],info_left[3],canvas_match_left,85,drawflagbefore=False,drawflagafter=True,showflag=False,color=colorset1[8],colorafter=colorset1[7])
Match_Evaluate(info_left[1],info_left[2],canvas_match_left,81,drawflagbefore=False,drawflagafter=False,showflag=False,color=colorset1[0])
Match_Evaluate(info_left[1],info_left[2],canvas_match_left,88,drawflagbefore=False,drawflagafter=False,showflag=False,color=colorset1[0])
Match_Evaluate(info_left[1],info_left[2],canvas_match_left,71,drawflagbefore=False,drawflagafter=False,showflag=False,color=colorset1[0])
Match_Evaluate(info_left[1],info_left[2],canvas_match_left,76,drawflagbefore=False,drawflagafter=False,showflag=True,color=colorset1[0])
cv2.imwrite('Match.jpg',canvas_match_left)
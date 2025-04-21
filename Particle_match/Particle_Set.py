from Particle import Particle_Info
import cv2 as cv2
import numpy as np
class Particle_Set(object):
    def __init__(self,name,json_set):
        self.name=name
        self.Time_Stamp=json_set[0]['img_num']
        self.Image_Path=json_set[0]['img_path']
        self.Particle_Num=len(json_set)
        self.Particle_List=[]
        for iter,entry in enumerate(json_set):
            radius=entry['r']
            Particle_Single_Info=Particle_Info(entry['x']+entry['x_in'],entry['y']+entry['y_in'],radius,self.Time_Stamp,iter)
            self.Particle_List.append(Particle_Single_Info)
    def plot(self,canvas,color=(0,255,0)):
        for iter in self.Particle_List:
            coordinate=iter[0]
            radius=iter[1]
            iternum=iter[2]
            cv2.circle(canvas,(int(coordinate.x),int(coordinate.y)),int(radius),color,2)
            #cv2.putText(canvas,str(iternum),(int(coordinate.x),int(coordinate.y)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
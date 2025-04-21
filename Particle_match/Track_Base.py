import numpy as np
import os
from Particle import Particle_Info
from Point import Point
from Kalman_Filter import Kalman_Filter
from Particle_Set import Particle_Set
import json
import cv2
import random
def Dist_Evaluate(Particle1:Particle_Info,place:Point):
    #Evaluate the distance between the particle and the place
    coordinate=Particle1.coordinate
    distance=(coordinate-place).__len__()-Particle1.radius
    return distance
class Track(object):
    def __init__(self,Trackid:int,Particle_Begin:Particle_Info,Particle_Medium:Particle_Info,Particle_End:Particle_Info):
        self.track_id=Trackid
        self.tracklst=[]
        self.tracklst.append(Particle_Begin)
        self.tracklst.append(Particle_Medium)
        self.tracklst.append(Particle_End)
        self.Kalman_Filter=Kalman_Filter(dt=1)
        self.Unhit=0
        self.Color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
        self.Max_Distance=10
        Particle_Begin.Link_Next(Particle_Medium)
            
        #Particle_Begin.Link_Pre(None)
        Particle_Medium.Link_Next(Particle_End)
        Particle_Medium.Link_Pre(Particle_Begin)
        Particle_End.Link_Next(None)
        Particle_End.Link_Pre(Particle_Medium)
        
        self.Kalman_Filter.update(np.array([Particle_Begin.coordinate.x,Particle_Begin.coordinate.y,0,0]))
        self.Kalman_Filter.predict()
        self.Kalman_Filter.update(np.array([Particle_Medium.coordinate.x,Particle_Medium.coordinate.y,Particle_Medium.coordinate.x-Particle_Begin.coordinate.x,Particle_Medium.coordinate.y-Particle_Begin.coordinate.y]))
        self.Kalman_Filter.predict()
        self.Kalman_Filter.update(np.array([Particle_End.coordinate.x,Particle_End.coordinate.y,Particle_End.coordinate.x-Particle_Medium.coordinate.x,Particle_End.coordinate.y-Particle_Medium.coordinate.y]))
                                  
    def __str__(self):
        return "Track ID: {0}, Track Length: {1}".format(self.track_id, len(self.tracklst))
    def Extend(self,Particle:Particle_Info):
        self.tracklst.append(Particle)
        Particle.Link_Pre(self.tracklst[-2])
        Particle.Link_Next(None)
        self.tracklst[-2].Link_Next(Particle)
    def IDHash(self):
        IDLst=[i.Lstnum for i in self.tracklst]
        return IDLst
    def Predict(self):
        target_Place=self.Kalman_Filter.predict()
        target_Place=Point(target_Place[0],target_Place[1])
        return target_Place
    def MergeTrack(self,other):

        self.tracklst.extend(other.tracklst)
        self.tracklst[-1].Link_Next(None)
        self.tracklst[0].Link_Pre(None)
        self.Kalman_Filter.update(np.array([self.tracklst[0].coordinate.x,self.tracklst[0].coordinate.y,0,0]))
        self.Kalman_Filter.predict()
        for i in range(1,len(self.tracklst)):
            self.Kalman_Filter.update(np.array([self.tracklst[i].coordinate.x,self.tracklst[i].coordinate.y,self.tracklst[i].coordinate.x-self.tracklst[i-1].coordinate.x,self.tracklst[i].coordinate.y-self.tracklst[i-1].coordinate.y]))
            self.Kalman_Filter.predict()
    def Update(self,NewFrame:Particle_Set):
        target_Place=self.Kalman_Filter.predict()
        target_Place=Point(target_Place[0],target_Place[1])
        for Particle in NewFrame.Particle_List:
            Select_Lst=[]
            if (Particle.coordinate-target_Place).__len__()<self.Max_Distance:
                deltaradius=1-abs(Particle.radius-self.tracklst[-1].radius)/max(Particle.radius,self.tracklst[-1].radius)
                if deltaradius>0.5:
                    Select_Lst.append(Particle)
            Select_Lst.sort(key=lambda x: Dist_Evaluate(x))
    def Plot_Track(self,canvas):
        for Particle in self.tracklst:
            cv2.circle(canvas,(int(Particle.coordinate.x),int(Particle.coordinate.y)),int(Particle.radius),self.Color,1)
            #Particle.plot(canvas,self.Color)
        for i in range(len(self.tracklst)-1):
            cv2.line(canvas,(int(self.tracklst[i].coordinate.x),int(self.tracklst[i].coordinate.y)),(int(self.tracklst[i+1].coordinate.x),int(self.tracklst[i+1].coordinate.y)),self.Color,1)
    def Track_To_Json(self):
        json_set=[]

        for Particle in self.tracklst:
            Info={"img_Num":Particle.Time_Stamp,"Particle_Num":Particle.Lstnum,"x":Particle.coordinate.x,"y":Particle.coordinate.y,"r":Particle.radius}
            json_set.append(Info)
        return json_set

        
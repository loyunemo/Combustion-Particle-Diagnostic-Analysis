import numpy as np
import cv2
from Particle_Set import Particle_Set
from Particle import Particle_Info
from Point import Point
from collections import defaultdict
from Track_Base import Track
from itertools import combinations
from scipy.optimize import linear_sum_assignment
from tools import calcRcamEpipolars,vertex_ditance,Reconstruct_3d_point,ReProjection_To_Left,ReProjection_To_Right,triangulate_point,Reconstruct_3d_point_Single
import math
import os
import csv
def Evaluate_Similarity(id0:int,id1:int,id2:int,Frame0:Particle_Set,Frame1:Particle_Set,Frame2:Particle_Set):
    #Evaluate the similarity between three particles
    Particle0=Frame0.Particle_List[id0]
    Particle1=Frame1.Particle_List[id1]
    Particle2=Frame2.Particle_List[id2]
    coordinate0=Particle0.coordinate
    coordinate1=Particle1.coordinate
    coordinate2=Particle2.coordinate
    radius0=Particle0.radius
    radius1=Particle1.radius
    radius2=Particle2.radius
    vec1=Point(coordinate1.x-coordinate0.x,coordinate1.y-coordinate0.y)
    vec2=Point(coordinate2.x-coordinate1.x,coordinate2.y-coordinate1.y)
    '''   if vec1.__len__()==0 and vec2.__len__()==0:
        return -1
    elif vec1.__len__()==0 or vec2.__len__()==0:
        return 10000
    else:
    '''
    if radius0==0 and radius1==0 and radius2==0:
        f=(vec2-vec1).__len__()
    else:
        f=(vec2-vec1).__len__()*(1+abs(radius2-radius1)+abs(radius1-radius0)/(radius2+radius1+radius0)*3.0)
    return f
class Match_Planner(object):
    def __init__(self,img_path):
        self.TrackLst=[]
        self.IsoParticleSet=[]
        self.OutTrack=[]
        self.FrameSet=[]
        self.img_path=img_path
    def Find_Smallest_Track_2d(self,Frame0:Particle_Set,Frame1:Particle_Set,Frame2:Particle_Set,Enable_Track=True):
        #给出三帧，找到合理的匹配，并且去掉冲突匹配，这里只要给出所有匹配就好
        #Link the track
        ParticleLd=[]
        ID_Select=[]
        for Particle in Frame0.Particle_List:
            if Enable_Track and Particle.PreNode is not None:
                continue
            coordinate=Particle.coordinate
            radius=Particle.radius
            iternum=Particle.Lstnum
            distance_lst=[]
            for num,Particle_after in enumerate(Frame1.Particle_List):
                if Enable_Track and Particle_after.PreNode is not None:
                    continue

                coordinate_after=Particle_after.coordinate
                radius_after=Particle_after.radius
                iternum_after=Particle_after.Lstnum
                distance=np.sqrt((coordinate.x-coordinate_after.x)**2+(coordinate.y-coordinate_after.y)**2)
                if(distance<150):
                    distance_lst.append([distance,iternum_after,Point(coordinate_after.x-coordinate.x,coordinate_after.y-coordinate.y),iternum])
            #distance_lst.sort(key=lambda x: x[0])
            popid=[]
            dislst=[]
            for i in range(len(distance_lst)):
                coordinate_medium=Frame1.Particle_List[distance_lst[i][1]].coordinate
                radius_medium=Frame1.Particle_List[distance_lst[i][1]].radius
                iternum_medium=Frame1.Particle_List[distance_lst[i][1]].Lstnum
                
                for Particle_second in Frame2.Particle_List:
                    if Enable_Track and Particle_second.PreNode is not None:
                        continue
                    coordinate_second=Particle_second.coordinate
                    radius_second=Particle_second.radius
                    iternum_second=Particle_second.Lstnum
                    distance2=np.sqrt((coordinate_medium.x-coordinate_second.x)**2+(coordinate_medium.y-coordinate_second.y)**2)
                    p2=Point(coordinate_second.x-coordinate_medium.x,coordinate_second.y-coordinate_medium.y)
                    delta=(distance_lst[i][2]-p2)
                    if(delta.__len__()<30 ):
                        dislst.append([delta.__len__(),iternum_medium,p2,delta,iternum_second])
            if(len(dislst)==0):
                continue
            else:
                dislst.sort(key=lambda x: x[0])
                select=dislst[0]
                ID_Select.append([Particle.Lstnum,select[1],select[4]])
        return ID_Select
    def Find_Smallest_Track(self,Frame0:Particle_Set,Frame1:Particle_Set,Frame2:Particle_Set,Enable_Track=True):
        #给出三帧，找到合理的匹配，并且去掉冲突匹配，这里只要给出所有匹配就好
        #Link the track
        ParticleLd=[]
        ID_Select=[]
        for Particle in Frame0.Particle_List:
            if Enable_Track and Particle.PreNode is not None:
                continue
            coordinate=Particle.coordinate
            radius=Particle.radius
            iternum=Particle.Lstnum
            distance_lst=[]
            for num,Particle_after in enumerate(Frame1.Particle_List):
                if Enable_Track and Particle_after.PreNode is not None:
                    continue

                coordinate_after=Particle_after.coordinate
                radius_after=Particle_after.radius
                iternum_after=Particle_after.Lstnum
                distance=np.sqrt((coordinate.x-coordinate_after.x)**2+(coordinate.y-coordinate_after.y)**2)
                distance_lst.append([distance,iternum_after,Point(coordinate_after.x-coordinate.x,coordinate_after.y-coordinate.y),iternum])
            distance_lst.sort(key=lambda x: x[0])
            distance_lst=distance_lst[:5]
            popid=[]
            for i in range(len(distance_lst)):
                coordinate_medium=Frame1.Particle_List[distance_lst[i][1]].coordinate
                radius_medium=Frame1.Particle_List[distance_lst[i][1]].radius
                iternum_medium=Frame1.Particle_List[distance_lst[i][1]].Lstnum
                dislst=[]
                for Particle_second in Frame2.Particle_List:
                    if Enable_Track and Particle_second.PreNode is not None:
                        continue
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
                            if(p2.dot(distance_lst[i][2])/distance_lst[i][2].__len__()/delta>np.cos(0.25*math.pi)):
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
                ParticleLd.append([Particle,select])
                ID_Select.append([Particle.Lstnum,select[1][1],select[1][4]])
        return ID_Select
    def Track_Sort(self,Frame0:Particle_Set,Frame1:Particle_Set,Frame2:Particle_Set,enable_track=True):
        #去除冲突匹配，然后Link
        trackid1=self.Find_Smallest_Track_2d(Frame0,Frame1,Frame2,enable_track)
        trackid2=self.Find_Smallest_Track_2d(Frame2,Frame1,Frame0,enable_track)
        trackid2_reversed = [list(reversed(i)) for i in trackid2]
        trackid2_reversed.sort(key=lambda x: x[0])
        track_selected = []
        for i in trackid2_reversed:
            if i in trackid1:
                track_selected.append(i)
        #track2copy=track_selected.copy()
        #track2copy.sort(key=lambda x: x[1])
        track_Non_Conflict=[]
        for i in track_selected:
            flag=False
            nums=0
            for num,t in enumerate(track_Non_Conflict):
                for iter in t:
                    if (iter[1]==i[1] or iter[2]==i[2]):
                        flag=True
                        track_Non_Conflict[num].append(i)
                        break
            if flag==False:
                track_Non_Conflict.append([i])
        for t in track_Non_Conflict:
            if len(t)>=2:
                t.sort(key=lambda x: Evaluate_Similarity(x[0],x[1],x[2],Frame0,Frame1,Frame2))
        track_Non_Conflict=[i[0] for i in track_Non_Conflict]

        startid=len(self.TrackLst)+len(self.OutTrack)
        for num,i in enumerate(track_Non_Conflict):
            trackselect=Track(startid+num,Frame0.Particle_List[i[0]],Frame1.Particle_List[i[1]],Frame2.Particle_List[i[2]])
            if len(self.TrackLst)==0:
                self.TrackLst.append([trackselect])
            else:
                if self.TrackLst[-1][0].tracklst[0].img_serial_num==trackselect.tracklst[0].img_serial_num:
                    self.TrackLst[-1].append(trackselect)
                else:
                    self.TrackLst.append([trackselect])
        lst=[i for i in range(Frame0.Particle_Num)]
        removelst=[i[0] for i in track_Non_Conflict]
        for i in removelst:
            lst.remove(i)
        self.IsoParticleSet.append([Frame0.Particle_List[i] for i in lst])
        print("[INFO]:",track_Non_Conflict)
        return track_Non_Conflict   
    def Track_Sort_Merge(self,Frame0:Particle_Set,Frame1:Particle_Set,Frame2:Particle_Set,enable_track=False):
        #去除冲突匹配，然后Link
        trackid1=self.Find_Smallest_Track_2d(Frame0,Frame1,Frame2,enable_track)
        trackid2=self.Find_Smallest_Track_2d(Frame2,Frame1,Frame0,enable_track)
        trackid2_reversed = [list(reversed(i)) for i in trackid2]
        trackid2_reversed.sort(key=lambda x: x[0])
        track_selected = []
        for i in trackid2_reversed:
            if i in trackid1:
                track_selected.append(i)
        #track2copy=track_selected.copy()
        #track2copy.sort(key=lambda x: x[1])
        track_Non_Conflict=[]
        for i in track_selected:
            flag=False
            nums=0
            for num,t in enumerate(track_Non_Conflict):
                for iter in t:
                    if (iter[0]==i[0] or iter[1]==i[1] or iter[2]==i[2]):
                        flag=True
                        track_Non_Conflict[num].append(i)
                        break
            if flag==False:
                track_Non_Conflict.append([i])
        for t in track_Non_Conflict:
            if len(t)>=2:
                t.sort(key=lambda x: Evaluate_Similarity(x[0],x[1],x[2],Frame0,Frame1,Frame2))
        track_Non_Conflict=[i[0] for i in track_Non_Conflict]

        startid=len(self.TrackLst)+len(self.OutTrack)
        for num,i in enumerate(track_Non_Conflict):
            if Frame1.Particle_List[i[1]].PreNode is not None:
                Frame1.Particle_List[i[1]].Link_Next(Frame2.Particle_List[i[2]])
            else:
                trackselect=Track(startid+num,Frame0.Particle_List[i[0]],Frame1.Particle_List[i[1]],Frame2.Particle_List[i[2]])
                self.TrackLst.append(trackselect)
                
        lst=[i for i in range(Frame0.Particle_Num)]
        removelst=[i[0] for i in track_Non_Conflict]
        for i in removelst:
            lst.remove(i)
        self.IsoParticleSet.append([Frame0.Particle_List[i] for i in lst])
        for tr in self.TrackLst:
            if tr.tracklst[-1].NextNode is not None:
                tr.Extend(tr.tracklst[-1].NextNode)
        self.FrameSet.append(Frame2)
        print("[INFO]:",track_Non_Conflict)
        return track_Non_Conflict                               
    def Merge_All(self):
        #合并所有的Track
        for i in range(len(self.TrackLst)-1):
            for j in range(i+1,len(self.TrackLst)):
                if self.TrackLst[i].IDHash()[-1]==self.TrackLst[j].IDHash()[0]:
                    self.TrackLst[i].tracklst.extend(self.TrackLst[j].tracklst)
                    self.TrackLst.pop(j)
                    break
        for i in self.OutTrack:
            for j in range(len(self.TrackLst)):
                if i.IDHash()==self.TrackLst[j].IDHash():
                    self.OutTrack.remove(i)
                    break        
    def Base_Track(self,Frame0:Particle_Set,Frame1:Particle_Set,Frame2:Particle_Set): 
        out=self.Track_Sort(Frame0,Frame1,Frame2)       
        self.FrameSet.append(Frame0)
        self.FrameSet.append(Frame1)
        self.FrameSet.append(Frame2)
        return out
    def Track_After(self,Frame:Particle_Set):
        out=self.Track_Sort(self.FrameSet[-2],self.FrameSet[-1],Frame)       
        self.FrameSet.append(Frame)

        return out
    def Track_After_2frame(self,Frame0:Particle_Set,Frame1:Particle_Set):
        out=self.Track_Sort(self.FrameSet[-1],Frame0,Frame1)       
        self.FrameSet.append(Frame0)
        self.FrameSet.append(Frame1)
        return out
    def UpdateNewFrame(self,NewFrame:Particle_Set):
        #更新新的帧，添加新的Track
        self.FrameSet.append(NewFrame)
        self.Track_Sort(self.FrameSet[-2],self.FrameSet[-1],NewFrame,enable_track=False)
        for i in range(len(self.TrackLst)):
            for j in range(len(self.TrackLst)):
                if self.TrackLst[i].tracklst[-1].PreNode==self.TrackLst[j].tracklst[0] :
                    self.TrackLst[i].Extend(self.TrackLst.tracklst[-1])
                    self.TrackLst.pop(j)
                    break
    def Info(self):
        #输出Track的信息
        sumtrackparticle=0
        allparticle=0
        for i in self.FrameSet:
            allparticle+=len(i.Particle_List)
            for j in i.Particle_List:
                if j.PreNode is None or j.NextNode is None:
                    sumtrackparticle+=1
        print("[INFO]: Total Particle Number:",allparticle,";Total Track Number:",sumtrackparticle)
        print("[INFO]: Match percent:",sumtrackparticle/allparticle)
    def Track_Extend(self,NewFrame:Particle_Set):
        UpdateParticleSet=[]       
        for tr in self.TrackLst:
            a=tr.Predict()    #匈牙利匹配找到最优解
            UpdateParticleSet.append(a)
        Cost_Matrix=[]
        for Particle in NewFrame.Particle_List:
            costlst=[]
            for i in UpdateParticleSet:
                costlst.append((Particle.coordinate-i).__len__())
            Cost_Matrix.append(costlst)
        Cost_Matrix=np.array(Cost_Matrix)
        row_ind, col_ind = linear_sum_assignment(Cost_Matrix)
        popid=[]
        for i in range(len(row_ind)):
            Particle=NewFrame.Particle_List[row_ind[i]]
            Distance=(Particle.coordinate-UpdateParticleSet[col_ind[i]]).__len__()
            if Distance<10:
                self.TrackLst[col_ind[i]].Extend(Particle)
            else:


                self.TrackLst[col_ind[i]].Unhit+=1
                predict_Particle=Particle_Info(UpdateParticleSet[col_ind[i]].x,UpdateParticleSet[col_ind[i]].y,self.TrackLst[col_ind[i]].tracklst[-1].radius,NewFrame.Time_Stamp,0)
                self.TrackLst[col_ind[i]].Extend(predict_Particle)
                if self.TrackLst[col_ind[i]].Unhit>=5:
                    self.TrackLst[col_ind[i]].Unhit=0
                    self.OutTrack.append(self.TrackLst[col_ind[i]])
                    popid.append(col_ind[i])
                    #self.TrackLst.remove(self.TrackLst[row_ind[i]])
        popid.sort(key=lambda x:x,reverse=True)
        for ix in popid:
            self.TrackLst.pop(ix)    #self.TrackLst[row_ind[i]].Extend(Particle)
    def Update(self,NewFrame:Particle_Set):
        self.Track_Extend(NewFrame)
        self.Track_Sort(self.FrameSet[-2],self.FrameSet[-1],NewFrame)


    def write_tracks_to_csv(self, filename):
        with open(filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            for track in self.TrackLst:
                img_nums = ['img_Num']
                particle_nums = ['Particle_Num']
                xs = ['x']
                ys = ['y']
                rs = ['r']

                for particle in track.tracklst:
                    img_nums.append(particle.img_serial_num)
                    particle_nums.append(particle.Lstnum)
                    xs.append(particle.coordinate.x)
                    ys.append(particle.coordinate.y)
                    rs.append(particle.radius)

                # 可以加一行 track_id 来标注每个track分界
                writer.writerow([f'Track ID: {track.track_id}'])
                writer.writerow(img_nums)
                writer.writerow(particle_nums)
                writer.writerow(xs)
                writer.writerow(ys)
                writer.writerow(rs)
                writer.writerow([])  # 空行用于分隔不同track
def Track_3d_Match(Sight1:Match_Planner,Sight2:Match_Planner,FrameLstLeft,FrameLstRight):
    Point3dLeft=[]
    Point3dRight=[]
    K1s = np.array([[1.151812867953681e+04, 0, 2.930281306554746e+02],
               [0, 1.144065639977414e+04,9.299614621342721e+02], 
               [0, 0, 1]], dtype=np.float64)
    K2s = np.array([[1.416281370757014e+04, 0, 2.186893663098017e+03],
               [0,1.397124917519382e+04, -2.934480724268153e+02], 
               [0, 0, 1]], dtype=np.float64)
    dist1 = np.array([-1.03599173650834,30.4343730533085,-0.0175971456828163,0.0775278591952446], dtype=np.float64)
    dist2 = np.array([1.49649840089790,-14.9214987001899, -0.0891119097877299,0.0319277395780453], dtype=np.float64)
    K1, roi = cv2.getOptimalNewCameraMatrix(    K1s, dist1, (2592, 1920), alpha=0, newImgSize=(2592, 1920))
    K2, roi = cv2.getOptimalNewCameraMatrix(    K2s, dist2, (2592, 1920), alpha=0, newImgSize=(2592, 1920))
    R = np.array([[0.747832245426745,	-0.00137646971907046,	0.663886314086309],
                [-0.0499508241855523	,0.997046660145766	,0.0583341465640672], 
                [-0.662005927362570	,-0.0767858243641788,	0.745554886855111]], dtype=np.float64)
    T = np.array([-332.797331625628,5.51661638734196,197.141393142023]).T
    Left=Sight1.Base_Track(FrameLstLeft[0],FrameLstLeft[1],FrameLstLeft[2])
    Right=Sight2.Base_Track(FrameLstRight[0],FrameLstRight[1],FrameLstRight[2])
    path_left=Sight1.img_path
    path_right=Sight2.img_path
    #path_right='C:/Users/Claudius/WorkingSpace/Article1/0722experiment2/51/BMP192.168.8.51-20240707-072633'
    pl=sorted([f for f in os.listdir(path_left) if f.endswith('bmp')])
    pr=sorted([f for f in os.listdir(path_right) if f.endswith('bmp')])
    print(FrameLstLeft[0].Time_Stamp)
    print(len(pl))
    print("[Info]:Track_Number:Left:{},Right:{}".format(len(Sight1.TrackLst),len(Sight2.TrackLst)))
    Match3dErrMatrix=[]
    Min_Match=[] 
    #canvasL=np.zeros((1920,2592,3), dtype=np.uint8)
    canvasL=cv2.imread(os.path.join(path_left,pl[FrameLstLeft[0].Time_Stamp]))
    #canvasL=cv2.undistort(canvasL,K1s,dist1)
    canvasR=cv2.imread(os.path.join(path_right,pr[FrameLstRight[0].Time_Stamp]))
    #canvasR=cv2.undistort(canvasR,K2s,dist2)
    import random
    for i in range(3):
        color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for Particle in FrameLstLeft[i].Particle_List:
            cv2.circle(canvasL,(int(Particle.coordinate.x),int(Particle.coordinate.y)),int(Particle.radius),color ,2)
        for Particle in FrameLstRight[i].Particle_List:
            cv2.circle(canvasR,(int(Particle.coordinate.x),int(Particle.coordinate.y)),int(Particle.radius),color ,2)
    '''
    for num,TrackObjectLeft in enumerate(Sight1.TrackLst):
        
        cv2.line(canvasL,(int(TrackObjectLeft.tracklst[0].coordinate.x),int(TrackObjectLeft.tracklst[0].coordinate.y)),(int(TrackObjectLeft.tracklst[1].coordinate.x),int(TrackObjectLeft.tracklst[1].coordinate.y)),(0,0,255),1)
        cv2.line(canvasL,(int(TrackObjectLeft.tracklst[1].coordinate.x),int(TrackObjectLeft.tracklst[1].coordinate.y)),(int(TrackObjectLeft.tracklst[2].coordinate.x),int(TrackObjectLeft.tracklst[2].coordinate.y)),(255,245,0),1)
        cv2.putText(canvasL,str(num),(int(TrackObjectLeft.tracklst[0].coordinate.x)+3,int(TrackObjectLeft.tracklst[0].coordinate.y)+3),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
    for num,TrackObjectRight in enumerate(Sight2.TrackLst):
        cv2.line(canvasR,(int(TrackObjectRight.tracklst[0].coordinate.x),int(TrackObjectRight.tracklst[0].coordinate.y)),(int(TrackObjectRight.tracklst[1].coordinate.x),int(TrackObjectRight.tracklst[1].coordinate.y)),(0,0,255),1)
        cv2.line(canvasR,(int(TrackObjectRight.tracklst[1].coordinate.x),int(TrackObjectRight.tracklst[1].coordinate.y)),(int(TrackObjectRight.tracklst[2].coordinate.x),int(TrackObjectRight.tracklst[2].coordinate.y)),(255,245,0),1)
        cv2.putText(canvasR,str(num),(int(TrackObjectRight.tracklst[0].coordinate.x)+3,int(TrackObjectRight.tracklst[0].coordinate.y)+3),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
    '''
    cv2.imwrite('TrackLeft.jpg',canvasL)
    cv2.imwrite('TrackRight.jpg',canvasR)
    for TrackObjectLeftLst,TrackObjectRightLst in zip(Sight1.TrackLst,Sight2.TrackLst):
        '''
        canvassL=canvasL.copy()
        
        cv2.line(canvassL,(int(TrackObjectLeft.tracklst[0].coordinate.x),int(TrackObjectLeft.tracklst[0].coordinate.y)),(int(TrackObjectLeft.tracklst[1].coordinate.x),int(TrackObjectLeft.tracklst[1].coordinate.y)),(0,0,255),1)
        cv2.line(canvassL,(int(TrackObjectLeft.tracklst[1].coordinate.x),int(TrackObjectLeft.tracklst[1].coordinate.y)),(int(TrackObjectLeft.tracklst[2].coordinate.x),int(TrackObjectLeft.tracklst[2].coordinate.y)),(255,245,0),1)
        '''
        Match3dErrMatrixLst=[]
        Match3dErrMatrix=[]
        for TrackObjectLeft in TrackObjectLeftLst:
            Lst=[]
            for TrackObjectRight in TrackObjectRightLst:
                '''
                canvassR=canvasR.copy()
                cv2.line(canvassR,(int(TrackObjectRight.tracklst[0].coordinate.x),int(TrackObjectRight.tracklst[0].coordinate.y)),(int(TrackObjectRight.tracklst[1].coordinate.x),int(TrackObjectRight.tracklst[1].coordinate.y)),(0,0,255),1)
                cv2.line(canvassR,(int(TrackObjectRight.tracklst[1].coordinate.x),int(TrackObjectRight.tracklst[1].coordinate.y)),(int(TrackObjectRight.tracklst[2].coordinate.x),int(TrackObjectRight.tracklst[2].coordinate.y)),(255,245,0),1)
                '''
                Line1,Line2,Line3=calcRcamEpipolars(TrackObjectLeft,K1,K2,R,T)
                delta1,delta2,delta3=vertex_ditance(TrackObjectRight,Line1,Line2,Line3)
                deltaerr=delta1+delta2+delta3
                ReconstructPoint1,ReconstructPoint2,ReconstructPoint3=Reconstruct_3d_point(TrackObjectLeft,TrackObjectRight,K1,K2,R,T)
                Point_plane1Left,Point_plane2Left,Point_plane3Left=ReProjection_To_Left(ReconstructPoint1,ReconstructPoint2,ReconstructPoint3,K1,R,T)
                Point_plane1Right,Point_plane2Right,Point_plane3Right=ReProjection_To_Right(ReconstructPoint1,ReconstructPoint2,ReconstructPoint3,K2,R,T)
                errL=(TrackObjectLeft.tracklst[0].coordinate-Point_plane1Left).__len__()+(TrackObjectLeft.tracklst[1].coordinate-Point_plane2Left).__len__()+(TrackObjectLeft.tracklst[2].coordinate-Point_plane3Left).__len__()
                errR=(TrackObjectRight.tracklst[0].coordinate-Point_plane1Right).__len__()+(TrackObjectRight.tracklst[1].coordinate-Point_plane2Right).__len__()+(TrackObjectRight.tracklst[2].coordinate-Point_plane3Right).__len__()
                #print("[ERR]:",deltaerr)
                Lst.append(deltaerr)

            Match3dErrMatrix.append(Lst)
        row_ind,col_ind=linear_sum_assignment(Match3dErrMatrix)
        Lst_Best_Match=[]
        Lst3d_Match=[]
        for row,col in zip(row_ind,col_ind):
            Lst_Best_Match.append([row,col,Match3dErrMatrix[row][col]])
            for i in range(3):
                    TrackObjectLeftLst[row].tracklst[i].Link_AnotherSide(TrackObjectRightLst[col].tracklst[i])
                    TrackObjectRightLst[col].tracklst[i].Link_AnotherSide(TrackObjectLeftLst[row].tracklst[i])
                    if(TrackObjectLeftLst[row].tracklst[i] in Point3dLeft):
                        continue
                    else:
                        Point3dLeft.append(TrackObjectLeftLst[row].tracklst[i])
            
        Match3dErrMatrixLst.append(Match3dErrMatrix)
    P3dLst=[]
    for num,Point3d in enumerate(Point3dLeft):
        if Point3d.PreNode  is None:
            ptr=Point3d
            lst1=[]
            while ptr is not None:
                result=Reconstruct_3d_point_Single(ptr,ptr.AnotherSideNode,K1,K2,R,T)
                ptr=ptr.NextNode
                lst1.append(result)
            P3dLst.append(lst1)
        else:
            continue
    with open('Track_3d_Data.csv', mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for iter in P3dLst:
            row1=[itera[0] for itera in iter]
            row2=[itera[1] for itera in iter]
            row3=[itera[2] for itera in iter]
            writer.writerow(row1)
            writer.writerow(row2)
            writer.writerow(row3)
            writer.writerow([])  # 空行用于分隔不同track    

    return Match3dErrMatrix
        

        
        

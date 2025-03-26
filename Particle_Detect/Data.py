class Data(object):
    def __init__(self,x,y,w,h,img,Serial_Num):
        self.x=x
        self.y=y
        self.w=w
        self.h=h
        self.img=img[x:x+w,y:y+h]
        self.Serial_Num=Serial_Num
    def Data_Write(self):
        return [self.x,self.y,self.w,self.h,self.Serial_Num]
    
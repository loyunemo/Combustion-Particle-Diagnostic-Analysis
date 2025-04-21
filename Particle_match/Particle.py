from Point import Point
class Particle_Info(object):
    def __init__(self, x,y,r,img_serial_num,Lstnum):
        self.coordinate=Point(x,y)
        self.radius=r
        self.img_serial_num=img_serial_num#该帧的时间序号
        self.Lstnum=Lstnum#单帧里面的序号
        self.NextNode=None
        self.PreNode=None
        self.AnotherSideNode=None
    def Link_Next(self,NextNode):
        if NextNode is None:
            self.NextNode=NextNode  
            return
        assert isinstance(NextNode, Particle_Info), "NextNode must be an instance of Particle_Info"
        self.NextNode=NextNode
    def Link_Pre(self,PreNode):
        if PreNode is None:
            self.PreNode=PreNode
            return
        assert isinstance(PreNode, Particle_Info), "PreNode must be an instance of Particle_Info"
        self.PreNode=PreNode
    def Link_AnotherSide(self,AnotherSideNode):
        if AnotherSideNode is None:
            self.AnotherSideNode=AnotherSideNode
            return
        assert isinstance(AnotherSideNode, Particle_Info), "AnotherSideNode must be an instance of Particle_Info"
        self.AnotherSideNode=AnotherSideNode

    
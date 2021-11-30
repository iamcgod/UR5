import cv2
import numpy as np
import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
import matplotlib.pyplot as plt
import math
import threading
import time
import math3d as m3d


# import DIP_find_defect_step_by_step as DIP
point_list = []
# center = 
class UR_Control():
    def __init__(self):
        self.v = 0.3
        self.a = 0.5
        self.Pi = math.pi
        self.jointspeed = 1
        self.pos_bool = False
        self.rob = urx.Robot("192.168.0.3")
        self.Gripper = Robotiq_Two_Finger_Gripper(self.rob)
        self.r = 0.05
        self.l = 0
        
    def speed(self,vel):
        self.v = vel*0.01
        return 

    def settcp(self,var):
        self.rob.set_tcp(var)
        return

    def setpayload(self,load):
        self.rob.set_payload(load)
        return 
        
    # def Connect_ur(self):
    #     self.Gripper = Robotiq_Two_Finger_Gripper(self.rob)
    #     return

    def Disconnect_ur(self):
        self.rob.stopj(self.a)
        self.rob.stopl(1)
        self.rob.close()
        print("Robot is close")
        return

    def Getdata(self):
        pos = self.rob.getl()
        
        return pos

    def Getdataj(self):
        posej = self.rob.getj()
        return posej

    def set2Home(self):
        # self.rob.movel(center, acc =0.1, vel = 0.1, wait = False)
        self.rob.movej([0, -(self.Pi/2), 0, -(self.Pi/2), 0, 0], acc=0.3, vel=0.05, wait=False)
        time.sleep(1)
        return

    def grip_open(self):
        self.Gripper.open_gripper()
        return

    def grip_close(self):
        self.Gripper.close_gripper()
        return

    def MoveJoint(self, number): #六軸
        if self.pos_bool == False:
            pose = self.rob.getj()
            self.posej = pose
            print("Get Position")
        else:
            print("Already Get")
        if number <6:
            self.pos_bool = True
            a = (self.Pi/180)*5
            self.posej[number] += self.a*self.jointspeed
            self.rob.movej(self.posej, vel=self.v, wait = False)
        #    self.jointspeed +=1
        else:
            number -= 6
            self.pos_bool = True
            a = (self.Pi/180)*5
            self.posej[number] -= self.a*self.jointspeed
            self.rob.movej(self.posej, vel=self.v, wait = False)
        #    self.jointspeed +=1
        time.sleep(1)
        self.jointspeed = 1
        print('wait for 1 seconds')
        return

    def stoprob(self): #停止機器人
        self.rob.stopj(self.a)
        self.rob.stopl(1)
        return

    def moveline(self, number): #線性移動
        self.posel = self.rob.getl()
        if number<3:
            self.posel[number] += 0.005
            self.rob.movel(self.posel, acc =self.a, vel = self.v, wait = False)
        else:
            number -=3
            self.posel[number] -= 0.005
            self.rob.movel(self.posel, acc =self.a, vel = self.v, wait = False)
        time.sleep(1)
        print(self.posel)
        return

    def movep(self, number): #圓周移動
        posep = self.rob.getl()
        self.posep = posep
        if number <3: 
            pose = self.rob.getl(wait = True)
            self.posep[number] += self.jointspeed
            self.rob.movep(self.posep, acc=self.a, vel=self.v, radius = self.r, wait = False)
            while True:
                p = self.rob.getl(wait = True)
                if p[number] > self.posep[number]-0.05:
                    break
            #self.rob.movep(self.posep, acc = self.a, vel = self.v, radius = 0 , wait = True)

        else:
            number -= 2
            self.posep[number] -= self.jointspeed
            self.rob.movep(self.posep, acc = self.a, vel=self.v, radius = self.r, wait = False)
            while True:
                p = self.rob.getl(wait = True)
                if p[number] < self.posep[number]+0.05:
                    break
            #self.rob.movep(self.posep, acc = self.a, vel = self.v, radius = 0 , wait = True)
        time.sleep(2)
        print('wait for 2 seconds', self.posep)
        return

    def toolmovel(self, number): #工具模式的movel
        if number ==0:
            self.l +=0.005
            self.rob.translate_tool((self.l, 0, 0), acc=self.a, vel=self.v, wait = False)
        elif number == 1:
            self.l +=0.005
            self.rob.translate_tool((0, self.l, 0), acc=self.a, vel=self.v, wait = False)
        elif number ==2:
            self.l +=0.005
            self.rob.translate_tool((0, 0, self.l), acc=self.a, vel=self.v, wait = False)
        elif number == 3:
            self.l -=0.005
            self.rob.translate_tool((self.l, 0, 0), acc=self.a, vel=self.v, wait = False)
        elif number == 4:
            self.l -=0.005
            self.rob.translate_tool((0, self.l, 0), acc=self.a, vel=self.v, wait = False)
        elif number == 5:
            self.l -=0.005
            self.rob.translate_tool((0, 0, self.l), acc=self.a, vel=self.v, wait = False)
        time.sleep(1)
        print('movetcpl success')
        return
              
    def free(self): #自由操作
        self.rob.set_freedrive(1,30)
        return

    def tcpmove(self, number): #TCP方向
        a = (self.Pi/180)*5
        trans = self.rob.get_pose()
        if number == 0:
            trans.orient.rotate_xt(a)
            self.rob.set_pose(trans, acc=0.2, vel=0.05, wait = False) 
        elif number ==1:
            trans.orient.rotate_yt(a)
            self.rob.set_pose(trans, acc=0.2, vel=0.05 ,wait = False)
        elif number ==2:
            trans.orient.rotate_zt(a)
            self.rob.set_pose(trans, acc=0.2, vel=0.05 ,wait = False)
        elif number == 3 :
            trans.orient.rotate_xt(-a)
            self.rob.set_pose(trans, acc=0.2, vel=0.05, wait = False) 
        elif number == 4 :
            trans.orient.rotate_yt(-a)
            self.rob.set_pose(trans, acc=0.2, vel=0.05, wait = False)
        elif number == 5 :
            trans.orient.rotate_zt(-a)
            self.rob.set_pose(trans, acc=0.2, vel=0.05, wait = False)  
        time.sleep(1)
        return

    def testmovec(self): #movec測試
        print("Test movec")
        pose = self.rob.get_pose()
        via = pose.copy()
        via.pos[0] += l
        to = via.copy()
        to.pos[1] += l
        self.rob.movec(via, to, acc=a, vel=v)
        return

    def move_to_point(self,point):
        self.rob.movel(point, acc =0.1, vel = 0.1, wait = False)
        pose = self.rob.getl()
        time.sleep(1)
        return pose

    def move_to_xy(self, x, y):
        pos = self.rob.getl()
        pos[0]+=x*0.01
        pos[1]+=y*0.01
        
        self.rob.movel(pos,acc=0.2,vel = 0.3, wait = False)
        time.sleep(0.5)
        return pos

    def move_to_z(self):
        pose=self.rob.getl()
        pose[2]-=16*0.01
        self.rob.movel(pose,acc=0.1,vel=0.1,wait = False)
        print("move Z")
        return pose


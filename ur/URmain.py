import sys
import cv2
from PyQt5.QtCore import *
from PyQt5.QtCore import QStringListModel, Qt, QTimer,QObject, pyqtSignal
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
import urx
from PyQt5.QtWidgets import QApplication, QWidget, QDialog, QLabel, QTextEdit, QPushButton, QCheckBox, QListWidgetItem, QMessageBox, QListWidget
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
import time
import threading

import math
import socket
# from numpy.lib.function_base import angle
import torch
from datetime import datetime
from View.UR_ui_5 import *  # 連接畫面

# import EasyPySpin
# from ImageEvents import get_main
# from AcquireAndDisplay import display_main
from utils import DIP_find_defect_step_by_step as DIP
import URuse

##[-0.016655463520290313, -0.5714131483848128, 0.2733962263096104] 目前焦距的距離 0.2733962263096104

HOST = "192.168.0.3"   # The remote host
PORT = 30002              # The same port as used by the server

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

cmd = "set_digital_out(2,True)" + "\n"
s.send (cmd.encode("utf-8"))
time.sleep(2)

error = 0.049464366411133956

robot_ur5 = URuse.UR_Control()
# # cap_ur5 = URvis.UR_cap()

print("Robot object is available as robot or rob")

class Foo(QObject):
    check_state = pyqtSignal()  # 宣告Signal
    # closeApp = pyqtSignal()

    def __init__(self):
        super(Foo, self).__init__()


class URpractice(QMainWindow):
    def __init__(self, parent=None):
        super(URpractice, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('UR Practice')
        self.f = Foo()

        self.pos_list = []
        self.move_list = []
        ########### call UR function ###########
        self.movelvalue() #線性數據
        self.movejvalue() #角度數據
        self.movel() #線性移動
        self.MoveJoint() #六軸移動
        self.movetcp() #TCP方向
        self.updataData() #更新數據
        self.URspeed() #速度
        self.ui.pushButton_11.clicked.connect(lambda: self.cap1())
        self.ui.pushButton_getl.clicked.connect(lambda: self.getldata())
        ##回零
        self.ui.pushButton_home.clicked.connect(lambda: URuse.UR_Control.set2Home(robot_ur5))
        ##停止機器人
        self.ui.pushButton.clicked.connect(lambda: URuse.UR_Control.stoprob(robot_ur5))
        ###自由移動
        self.ui.pushButton_free.pressed.connect(lambda: URuse.UR_Control.free(robot_ur5))
        self.ui.pushButton_free.released.connect(lambda: URuse.UR_Control.stoprob(robot_ur5))

        self.ui.pushButton_openccd.clicked.connect(lambda: self.cap_open())#display_main())#self.cap_open())#
        # # 夾爪
        self.ui.pushButton_gripclose.setIcon(QtGui.QIcon("./icon/close.jpg"))
        self.ui.pushButton_gripclose.setIconSize(QtCore.QSize(105, 75))
        self.ui.pushButton_gripopen.setIcon(QtGui.QIcon("./icon/open.jpg"))
        self.ui.pushButton_gripopen.setIconSize(QtCore.QSize(105, 75))
        self.ui.pushButton_gripopen.clicked.connect(lambda: URuse.UR_Control.grip_open(robot_ur5))
        self.ui.pushButton_gripclose.clicked.connect(lambda: URuse.UR_Control.grip_close(robot_ur5))

        # combolist set
        combo = ['機座', '工具']
        self.ui.comboBox_list.addItems(combo)
        self.ui.comboBox_list.currentTextChanged.connect(self.movelvalue)
        self.ui.comboBox_list.currentTextChanged.connect(self.movel)
        self.ui.pushButton_3.clicked.connect(lambda: self.gettcp())

        # 右鍵選單
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.right_menu)

        # tablewidget
        self.table = self.ui.tableWidget
        self.table.setEditTriggers(QTableWidget.DoubleClicked)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setColumnCount(7)
        self.table.setColumnWidth(0, 20)
        horizontalHeader = ["", "x", "y", "z", "rx", "ry", "rz"]
        self.table.setHorizontalHeaderLabels(horizontalHeader)

        ############ DIP ############
        self.pattern_path = ' '
        self.save_root = ' '
        self.angle_size = self.ui.spinBox_angle.value()
        self.scale_size = self.ui.doubleSpinBox_scale.value()
        self.golden_sample = ' '
        self.best_diff = ' '
        self.opening_img = ' '
        self.predict_img = ' '
        # 以下呼叫各種要用到的def
        self.ui.toolButton_pattern.clicked.connect(lambda: self.openSlot_pattern(self.ui.label_pattern))
        self.ui.toolButton_save.clicked.connect(lambda: self.openSlot_chooseDir())
        self.ui.spinBox_angle.valueChanged.connect(self.Valuechange_angle)
        self.ui.doubleSpinBox_scale.valueChanged.connect(self.Valuechange_scale)
        self.ui.pushButton_detect.clicked.connect(lambda: self.detect())
        self.ui.pushButton_save.clicked.connect(lambda: self.save())

        self.timer = QTimer(self)  # 呼叫 QTimer
        self.timer.timeout.connect(self.run)  # 當時間到時會執行 run
        # self.timer.start(5000) #啟動 Timer .. 每隔1000ms 會觸發 run
        ############ DIP ############
        self.show()

    def right_menu(self, pos):
        menu = QMenu()
        del_option = menu.addAction('Delete')
        del_option.triggered.connect(lambda: self.delete_point())
        menu.exec_(self.mapToGlobal(pos))
        return

    def getldata(self):
        pos = URuse.UR_Control.Getdata(robot_ur5)
        self.pos_list.append(pos)
        print("save pos:", len(self.pos_list))
        print(pos)
        self.append_list(pos)
        return

    def append_list(self, pos):
        # tablewidget
        row_count = self.table.rowCount() #數現在有幾行
        self.table.insertRow(row_count) #插在最後一行
        self.check = QTableWidgetItem() 
        self.check.setCheckState(Qt.Unchecked) #一開始設定不打勾
        self.table.setItem(row_count, 0, self.check) 
        for j in range(6): #依照順序將點的座標填入
            self.table.setItem(row_count, j+1, QTableWidgetItem(
                str(pos[j])[:10]))
        return

    def delete_point(self):
        rows = sorted(set(index.row()
                      for index in self.table.selectedIndexes()))
        for row in rows:
            print('Row %d is selected' % row)
            self.table.removeRow(row)
        return

    def table_item_clicked(self): #最後確認有打勾的點 加到move_list裡面 
        self.move_list = []       
        for i in range(self.table.rowCount()):
            if self.table.item(i, 0).checkState() == QtCore.Qt.Checked: #檢查第0欄的打勾狀態
                self.move_list.append(self.pos_list[i])
                print("point %s is selected!!" % (i+1))
        return self.move_list

    def gettcp(self):  # 取得TCP設定值 payload
        if self.ui.plainTextEdit_x.toPlainText() == "":
            t1 = 0
        else:
            t1 = 0.001*int(self.ui.plainTextEdit_x.toPlainText())
        if self.ui.plainTextEdit_y.toPlainText() == "":
            t2 = 0
        else:
            t2 = 0.001*int(self.ui.plainTextEdit_y.toPlainText())
        if self.ui.plainTextEdit_z.toPlainText() == "":
            t3 = 0
        else:
            t3 = 0.001*int(self.ui.plainTextEdit_z.toPlainText())
        if self.ui.plainTextEdit_rx.toPlainText() == "":
            t4 = 0
        else:
            t4 = int(self.ui.plainTextEdit_rx.toPlainText())
        if self.ui.plainTextEdit_ry.toPlainText() == "":
            t5 = 0
        else:
            t5 = int(self.ui.plainTextEdit_ry.toPlainText())
        if self.ui.plainTextEdit_rz.toPlainText() == "":
            t6 = 0
        else:
            t6 = int(self.ui.plainTextEdit_rz.toPlainText())
        var = t1, t2, t3, t4, t5, t6
        if self.ui.plainTextEdit_payload.toPlainText() == "":
            payload = 0
        else:
            payload = int(self.ui.plainTextEdit_payload.toPlainText())
        print("tcp:", var, "\npayload:", payload)
        URuse.UR_Control.settcp(robot_ur5, var)
        URuse.UR_Control.setpayload(robot_ur5, payload)
        self.ui.label_37.setText("setting success")
        return

    def movelvalue(self):  # 得到XY數據, pos
        pos = URuse.UR_Control.Getdata(robot_ur5)
        text = self.ui.comboBox_list.currentText()

        if text == '機座':
            self.ui.label_x_2.setText(str(round(1000*pos[0], 2)))
            self.ui.label_y_2.setText(str(round(pos[1]*1000, 2)))
            self.ui.label_z_2.setText(str(round(pos[2]*1000, 2)))
            self.ui.label_rx_2.setText(str(round(pos[3], 3)))
            self.ui.label_ry_2.setText(str(round(pos[4], 3)))
            self.ui.label_rz_2.setText(str(round(pos[5], 3)))
        elif text == '工具':
            self.ui.label_x_2.setText(str(round(pos[0])))
            self.ui.label_y_2.setText(str(round(pos[1])))
            self.ui.label_z_2.setText(str(round(pos[2])))
            self.ui.label_rx_2.setText(str(round(pos[3])))
            self.ui.label_ry_2.setText(str(round(pos[4])))
            self.ui.label_rz_2.setText(str(round(pos[5])))
        return

    def movejvalue(self):  # 取得joint 1rad = 57.29578 degree
        posej = URuse.UR_Control.Getdataj(robot_ur5)
        self.ui.label_4.setText(str(round(posej[0]*180/3.14, 2)))
        self.ui.label_6.setText(str(round(posej[1]*180/3.14, 2)))
        self.ui.label_7.setText(str(round(posej[2]*180/3.14, 2)))
        self.ui.label_8.setText(str(round(posej[3]*180/3.14, 2)))
        self.ui.label_9.setText(str(round(posej[4]*180/3.14, 2)))
        self.ui.label_10.setText(str(round(posej[5]*180/3.14, 2)))
        return

    def updataData(self):  # 更新數據
        self.timer = QTimer(self)  # 呼叫 QTimer
        self.timer.timeout.connect(self.movelvalue)  # 當時間到時會執行
        self.timer.timeout.connect(self.movejvalue)
        self.timer.start(500)  # 啟動 Timer 每隔1000ms 1秒 會觸發
        return

    def movel(self):  # 線性移動
        self.text = self.ui.comboBox_list.currentText()
        if self.text == '工具':
            print(self.text)
            self.ui.pushButton_tcpdown.pressed.connect(lambda: self.toolx())
            self.ui.pushButton_tcpup.pressed.connect(lambda: self.toolxx())
            self.ui.pushButton_tcpleft.pressed.connect(lambda: self.toolyy())
            self.ui.pushButton_tcpright.pressed.connect(lambda: self.tooly())
            self.ui.pushButton_up_2.pressed.connect(lambda: self.toolz())
            self.ui.pushButton_down.pressed.connect(lambda: self.toolzz())
        else:
            self.ui.pushButton_tcpup.pressed.connect(lambda: self.movelinexx())
            self.ui.pushButton_tcpdown.pressed.connect(
                lambda: self.movelinex())
            self.ui.pushButton_tcpleft.pressed.connect(
                lambda: self.movelineyy())
            self.ui.pushButton_tcpright.pressed.connect(
                lambda: self.moveliney())
            self.ui.pushButton_up_2.pressed.connect(lambda: self.movelinez())
            self.ui.pushButton_down.pressed.connect(lambda: self.movelinezz())
        # 停止
        self.ui.pushButton_tcpup.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_tcpdown.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_tcpleft.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_tcpright.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_up_2.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_down.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        return

    def movelinex(self):
        var = 0
        URuse.UR_Control.moveline(robot_ur5, var)
        return

    def moveliney(self):
        var = 1
        URuse.UR_Control.moveline(robot_ur5, var)
        return

    def movelinez(self):
        var = 2
        URuse.UR_Control.moveline(robot_ur5, var)
        return

    def movelinexx(self):
        var = 3
        URuse.UR_Control.moveline(robot_ur5, var)
        return

    def movelineyy(self):
        var = 4
        URuse.UR_Control.moveline(robot_ur5, var)
        return

    def movelinezz(self):
        var = 5
        URuse.UR_Control.moveline(robot_ur5, var)
        return

    def toolx(self):
        va = 0
        URuse.UR_Control.toolmovel(robot_ur5, va)
        return

    def tooly(self):
        va = 1
        URuse.UR_Control.toolmovel(robot_ur5, va)
        return

    def toolz(self):
        va = 2
        URuse.UR_Control.toolmovel(robot_ur5, va)
        return

    def toolxx(self):
        va = 3
        URuse.UR_Control.toolmovel(robot_ur5, va)
        return

    def toolyy(self):
        va = 4
        URuse.UR_Control.toolmovel(robot_ur5, va)
        return

    def toolzz(self):
        va = 5
        URuse.UR_Control.toolmovel(robot_ur5, va)
        return

    def MoveJoint(self):  # 六軸移動, number
        self.ui.pushButton_base.pressed.connect(lambda: self.movebase2())
        self.ui.pushButton_base2.pressed.connect(lambda: self.movebase())
        self.ui.pushButton_shouder.pressed.connect(
            lambda: self.moveshoulder2())
        self.ui.pushButton_shoulder2.pressed.connect(
            lambda: self.moveshoulder())
        self.ui.pushButton_elbow.pressed.connect(lambda: self.moveelbow2())
        self.ui.pushButton_elbow2.pressed.connect(lambda: self.moveelbow())
        self.ui.pushButton_wrist1.pressed.connect(lambda: self.movewrist12())
        self.ui.pushButton_wrist12.pressed.connect(lambda: self.movewrist1())
        self.ui.pushButton_wrist2.pressed.connect(lambda: self.movewrist22())
        self.ui.pushButton_wrist22.pressed.connect(lambda: self.movewrist2())
        self.ui.pushButton_wrist3.pressed.connect(lambda: self.movewrist32())
        self.ui.pushButton_wrist32.pressed.connect(lambda: self.movewrist3())
        # 拜託停下來
        self.ui.pushButton_base.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_base2.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_shouder.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_shoulder2.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_elbow.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_elbow2.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_wrist1.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_wrist12.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_wrist2.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_wrist22.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_wrist3.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_wrist32.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        return

    def movebase(self):
        var = 0
        URuse.UR_Control.MoveJoint(robot_ur5, var)
        return

    def moveshoulder(self):
        var = 1
        URuse.UR_Control.MoveJoint(robot_ur5, var)
        return

    def moveelbow(self):
        var = 2
        URuse.UR_Control.MoveJoint(robot_ur5, var)
        return

    def movewrist1(self):
        var = 3
        URuse.UR_Control.MoveJoint(robot_ur5, var)
        return

    def movewrist2(self):
        var = 4
        URuse.UR_Control.MoveJoint(robot_ur5, var)
        return

    def movewrist3(self):
        var = 5
        URuse.UR_Control.MoveJoint(robot_ur5, var)
        return

    def movebase2(self):
        var = 6
        URuse.UR_Control.MoveJoint(robot_ur5, var)
        return

    def moveshoulder2(self):
        var = 7
        URuse.UR_Control.MoveJoint(robot_ur5, var)
        return

    def moveelbow2(self):
        var = 8
        URuse.UR_Control.MoveJoint(robot_ur5, var)
        return

    def movewrist12(self):
        var = 9
        URuse.UR_Control.MoveJoint(robot_ur5, var)
        return

    def movewrist22(self):
        var = 10
        URuse.UR_Control.MoveJoint(robot_ur5, var)
        return

    def movewrist32(self):
        var = 11
        URuse.UR_Control.MoveJoint(robot_ur5, var)
        return

    def movetcp(self):  # TCP方向
        self.ui.pushButton_movedown.pressed.connect(lambda: self.movery())
        self.ui.pushButton_moveup.pressed.connect(lambda: self.moveryback())
        self.ui.pushButton_moveright.pressed.connect(lambda: self.moverxback())
        self.ui.pushButton_moveleft.pressed.connect(lambda: self.moverx())
        self.ui.pushButton_movep.pressed.connect(lambda: self.moverz())
        self.ui.pushButton_movep_2.pressed.connect(lambda: self.moverzback())
        # 拜託停下來
        self.ui.pushButton_moveup.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_movedown.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_moveright.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_moveleft.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_movep.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        self.ui.pushButton_movep_2.released.connect(
            lambda: URuse.UR_Control.stoprob(robot_ur5))
        return

    def moverx(self):
        var = 0
        URuse.UR_Control.tcpmove(robot_ur5, var)
        return

    def movery(self):
        var = 1
        URuse.UR_Control.tcpmove(robot_ur5, var)
        return

    def moverz(self):
        var = 2
        URuse.UR_Control.tcpmove(robot_ur5, var)
        return

    def moverxback(self):
        var = 3
        URuse.UR_Control.tcpmove(robot_ur5, var)
        return

    def moveryback(self):
        var = 4
        URuse.UR_Control.tcpmove(robot_ur5, var)
        return

    def moverzback(self):
        var = 5
        URuse.UR_Control.tcpmove(robot_ur5, var)
        return

    def URspeed(self):  # 調整速度
        self.ui.horizontalSlider_speed.setRange(1, 100)
        speed = self.ui.horizontalSlider_speed.value
        self.ui.horizontalSlider_speed.valueChanged.connect(
            lambda: URuse.UR_Control.speed(robot_ur5, speed))
        return speed

    def cap1(self):
        self.table_item_clicked()
        print("Total move point:", len(self.move_list))

        # 確定距離
        # p = URuse.UR_Control.move_to_point(robot_ur5,point_z)
        # time.sleep(1.5)
        # filename = self.ui.textEdit_filename.toPlainText()

        for i in range(0, len(self.move_list)):  # 一組 拍兩輪 比較手臂點的位置差異
            # 移動到第一點拍攝 + 紀錄點位置
            # print(i, ":", self.move_list[i])

            time.sleep(0.5)
            point = URuse.UR_Control.move_to_point(robot_ur5, self.move_list[i])
            # time.sleep(0.5)
            # get_main(filename=filename, num=i+1)
            # image = self.getimg(num=i+1)
            
            # show label img
            # self.showImg(image,self.ui.label_img)
            cv2.waitKey(5) 
            # self.detect(image)
            # self.save(num)
            time.sleep(0.5)

        # URuse.UR_Control.set2Home(robot_ur5)
        return

    def getimg(self, num):
        cap = EasyPySpin.VideoCapture(0)
        ret, frame = cap.read()
        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)
        if self.ui.checkBox_saveorigin.isChecked():
            if self.save_root == " ":
                self.save_root = "./Defect"
            now = datetime.now()
            filename = str(self.save_root) + "/" + now.strftime('%Y_%m_%d_%H_%M_%S_%f') + ".bmp"
            cv2.imwrite(filename, frame)
            print(filename)  
        cap.release()
        return frame

    def cap_open(self):
        cap = EasyPySpin.VideoCapture(0)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,840)
        # ret, frame = cap.read()
    
        while(True):
        # 從攝影機擷取一張影像
            ret, frame = cap.read()
            # 顯示圖片
            # frame = cv2.resize(frame, (1280, 960))
            # cv2.imshow('frame', frame)
            self.showImg(frame,self.ui.label_img)
            # 若按下 q 鍵則離開迴圈
            if cv2.waitKey(1) & 0xFF == ord('q'):
            # if event.key() == Qt.Key_Any: #任意鍵停止
                # print("press s")
                break
        cap.release()
        # self.ui.label_img.setText("defect img")
        cv2.destroyAllWindows()
        return


########### DIP #############
    def openSlot_pattern(self, label): #open golden sample
        imgName, imgType = QFileDialog.getOpenFileName(
            self, 'Open Image', 'Image', '*.png *.jpg *.jpeg *.bmp')
        if imgName == '':
            return
        self.img = cv2.imread(imgName, 1)
        if self.img.size == 1:
            return
        self.showImg(self.img, label)
        self.ui.label_info_pattern.setText(imgName)
        self.pattern_path = imgName
        self.ui.label_info_save_done.setText("pattern image is selected")
        self.timer.start(2000)
        print("【pattern_path】", self.pattern_path)

    def openSlot_chooseDir(self): #choose save dir
        dir = QFileDialog.getExistingDirectory(self, "選擇儲存文件夾", "/")
        if dir == "":
            dir = "./Defect/"
            if not os.path.isdir(dir):
                os.mkdir(dir)
            return
        self.save_root = dir
        self.ui.label_info_save_done.setText("save file dir is selected")
        self.timer.start(2000)
        print("【save_root】", self.save_root)

    def showImg(self, image, label):
        if len(image.shape) == 3:
            height, width, channel = image.shape
        elif len(image.shape) == 2:
            height, width = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            channel = 3
        else:
            print("image loading failed !")
            return
        bytesPerline = channel * width
        self.qImg = QImage(image.data, width, height,
                           bytesPerline, QImage.Format_RGB888).rgbSwapped()
        pil_image = self.m_resize(
            label.width(), label.height(), self.qImg)
        pixmap = QPixmap(pil_image)
        label.setPixmap(pixmap)

    def m_resize(self, w_box, h_box, pil_image):  # 參數是：要適應的窗口寬、高、Image.open後的圖片
        w, h = pil_image.width(), pil_image.height()  # 獲取圖像的原始大小
        f1 = 1.0 * w_box/w
        f2 = 1.0 * h_box/h
        factor = min([f1, f2])
        width = int(w*factor)
        height = int(h*factor)
        return pil_image.scaled(width, height)

    def Valuechange_angle(self):
        self.angle_size = self.ui.spinBox_angle.value()

    def Valuechange_scale(self):
        self.scale_size = self.ui.doubleSpinBox_scale.value()

    def golden(self,image): #1 原本的adjust pattern
        ######用predict的######
        # tag = DIP.predict_num(image)
        # self.pattern_path = "./golden_sample/black_" + str(tag) + ".jpeg"
        #######################
        best_pattern, self.golden_sample, img_draw, best_angle, best_scale, total_time = DIP.make_golden(
            image, self.pattern_path, self.angle_size, self.scale_size)

        self.ui.label_info_angle.setText(str(best_angle))
        self.ui.label_info_scale.setText(str(best_scale))
        self.ui.label_info_time.setText(str(round(total_time, 3)))
        # self.showImg(img_draw, self.ui.label_img)
        self.showImg(best_pattern, self.ui.label_pattern)
        self.ui.label_info_save_done.setText("【adjust pattern】is done")
        self.timer.start(2000)

    def absdiff(self, image): #2
        # defect_img = cv2.imread(self.defect_path, -1)
        self.best_diff = DIP.img_diff(image, self.golden_sample)
        # self.showImg(self.best_diff, self.ui.label_img)
        self.ui.label_info_save_done.setText("【absdiff】is done")
        self.timer.start(2000)
        print("best_diff is done")

    def opening(self): #3
        opening_img = DIP.opening(
            self.best_diff, kernel_size=5, iterations=2)
        # self.showImg(self.opening_img, self.ui.label_img)
        self.ui.label_info_save_done.setText("【opening】is done")
        self.timer.start(2000)
        print("opening_img is done")
        return opening_img

    def detect(self, image): #4
        # defect_img = cv2.imread(self.defect_path, -1)
        self.golden(image)
        self.absdiff(image)
        if self.ui.checkBox_opening.isChecked():
            self.opening_img = self.opening()
        else:
            self.opening_img = self.best_diff

        self.predict_img = DIP.find_max_contour(
            self.opening_img, defect_img, side_border=0)
        self.showImg(self.predict_img, self.ui.label_img)
        self.ui.label_info_save_done.setText("【detect】is done")
        self.ui.label_info_img_name.setText("【detect result】")
        self.timer.start(2000)
        print("predict_img is done")

    def save(self,num):
        if self.save_root == ' ':
            self.ui.label_info_save_done.setText(
                "【save】Warning: Choose a save file")
            self.timer.start(2000)
            return
        DIP.save_predictImg(self.predict_img, self.save_root, num)
        self.ui.label_info_save_done.setText("save successfully!")
        
        self.timer.start(2000)

    def run(self):
        self.ui.label_info_save_done.setText(" ")
        self.timer.stop()


class SubDialog(QDialog):  # 物件選擇
    def __init__(self, parent=URpractice):
        super(SubDialog, self).__init__()
        self.dia1 = Ui_Dialog()
        self.dia1.setupUi(self)
        self.setWindowTitle("選擇抓取物件")
        self.dia1.pushButton_diaok.clicked.connect(lambda: self.get())
        self.dia1.pushButton_3.clicked.connect(lambda: self.buttoncancel())

    def calldia(self):  # 呼叫其他自定義訊息框
        image = QtGui.QPixmap('C:/Users/user/Desktop/choose.jpg')
        self.dia1.label_3.setPixmap(image)
        self.show()
        return

    def buttoncancel(self):  # 關閉訊息框
        num = None
        self.close()
        return num

    def get(self):  # 取得數值
        num = int(self.dia1.plainTextEdit.toPlainText())
        self.close()
        return num


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(".\icon\microscope.png"))
    main = URpractice()
    main.show()
    sys.exit(app.exec_())
    count = 0
    while (count < 1):
        time.sleep(2)

        count = count + 1
        print("The count is:", count)
    print("Program finish")
data = r.recv(1024)
r.close()
